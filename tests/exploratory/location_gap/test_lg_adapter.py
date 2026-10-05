import importlib.util
import io
import types
import unittest
import torch
from torch.utils.data import DataLoader
from racepv.models import RaCEPVConfig
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from racepv.training.research_runtime import ResearchTrainer

torch.set_num_threads(1)


def control(seed=29):
    torch.manual_seed(seed)
    return ReliabilityControlRaCEPV(RaCEPVConfig(history_features=14,
        future_features=9, site_features=6, future_hour_sin_index=4,
        future_hour_cos_index=5, backbone='gru'))


def batch():
    g = torch.Generator().manual_seed(13)
    h = .15 + .65 * torch.rand(2, 192, 14, generator=g)
    f = .2 + .4 * torch.rand(2, 24, 9, generator=g)
    f[:, ::5, 0] = 0
    f[..., 4] = torch.linspace(0, 1, 24).sin()
    f[..., 5] = torch.linspace(0, 1, 24).cos()
    target_mask = torch.ones(2, 24, dtype=torch.bool)
    target_mask[:, 2] = False
    return dict(history_values=h, history_mask=torch.ones_like(h, dtype=torch.bool),
        future_known=f, site_metadata=torch.rand(2, 6, generator=g),
        targets=torch.linspace(.1, .8, 48).reshape(2, 24),
        state_labels=torch.arange(48).reshape(2, 24) % 3,
        target_mask=target_mask, daylight_mask=f[..., 0] >= .01)


def inputs(b):
    return {k: b[k] for k in ('history_values', 'history_mask', 'future_known', 'site_metadata')}


class ScientificAdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('racepv.exploratory.location_gap.lg_adapter'),
            'C-LG scientific adapter is not implemented')
        from racepv.exploratory.location_gap.lg_adapter import LocationGapScientificAdapter, LocationGapResearchTrainer
        self.Adapter, self.Trainer = LocationGapScientificAdapter, LocationGapResearchTrainer
        self.c = control()
        self.a = self.Adapter(self.c, backend='cpu-gru')
        self.b = batch()

    def test_shared_state_buffers_repeated_rows_rng_and_hash_proof(self):
        from racepv.exploratory.location_gap.lg_adapter import shared_initialization_proof
        for seed in (29, 47):
            c = control(seed)
            rng = torch.get_rng_state().clone()
            a = self.Adapter(c, backend='cpu-gru')
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            proof = shared_initialization_proof(c, a)
            self.assertEqual(proof['control_shared_sha256'], proof['candidate_shared_sha256'])
            self.assertNotEqual(proof['control_full_sha256'], proof['candidate_full_sha256'])
            self.assertTrue(proof['repeated_three_role_rows_exact'])
            self.assertGreater(proof['shared_buffer_count'], 0)
            # Check every shared parameter AND buffer independently of proof code.
            cs, ads = c.state_dict(), a.base.state_dict()
            keys = ('fusion_router.contextual_network.7.weight', 'fusion_router.contextual_network.7.bias')
            for name, value in cs.items():
                expected = value.repeat(3, 1) if name == keys[0] else value.repeat(3) if name == keys[1] else value
                self.assertTrue(torch.equal(expected, ads[name]), name)
            self.assertFalse({id(p) for p in c.parameters()} & {id(p) for p in a.parameters()})
            with torch.no_grad():
                a.base.fusion_router.contextual_network[-1].bias[3] += 1
            with self.assertRaisesRegex(ValueError, 'projection'):
                shared_initialization_proof(c, a)

    def test_tied_eval_and_train_outputs_rng_and_single_graph(self):
        for training in (False, True):
            self.c.train(training); self.a.train(training)
            rng = torch.get_rng_state().clone()
            old = self.c(**inputs(self.b)); after = torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            counts = {n: 0 for n in ('stable_expert', 'transient_expert', 'fusion_router', 'state_head')}
            hooks = []
            def counter(name):
                def count(*args): counts[name] += 1
                return count
            for name in counts: hooks.append(getattr(self.a.base, name).register_forward_hook(counter(name)))
            try: new = self.a(**inputs(self.b))
            finally:
                for h in hooks: h.remove()
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            self.assertTrue(all(v == 1 for v in counts.values()))
            torch.testing.assert_close(old['quantiles'], new['quantiles'], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(old['state_logits'], new['state_logits'], rtol=0, atol=0)
            self.assertEqual(new['role_gate_weights'].shape, (2, 24, 3, 3))
            self.assertNotIn('gate_weights', new)
            self.assertNotIn('fusion_logits', new)

    def test_role_perturbations_change_only_matching_quantile_side(self):
        self.a.eval()
        initial = self.a(**inputs(self.b))
        for role, unchanged in ((0, None), (1, slice(3, 7)), (2, slice(0, 4))):
            a = self.Adapter(self.c, backend='cpu-gru').eval()
            with torch.no_grad(): a.base.fusion_router.contextual_network[-1].bias[role*3:role*3+3] += torch.tensor([3., -1., -2.])
            result = a(**inputs(self.b))
            torch.testing.assert_close(result['state_logits'], initial['state_logits'], rtol=0, atol=0)
            torch.testing.assert_close(result['expert_quantiles'], initial['expert_quantiles'], rtol=0, atol=0)
            self.assertGreater((result['quantiles'] - initial['quantiles']).abs().sum().item(), 0)
            if unchanged is not None:
                torch.testing.assert_close(result['quantiles'][..., unchanged], initial['quantiles'][..., unchanged], rtol=0, atol=0)
            self.assertTrue((result['quantiles'].diff(dim=-1) >= 0).all())
            self.assertTrue((result['quantiles'][~self.b['daylight_mask']] == 0).all())

    def test_loss_uses_projected_quantiles_masks_and_disjoint_gradients(self):
        self.a.eval()
        out = self.a(**inputs(self.b)); objective = self.a.make_objective()
        args = (self.b['targets'], self.b['state_labels'])
        kw = dict(daylight_mask=self.b['daylight_mask'], target_mask=self.b['target_mask'])
        losses = objective(out, *args, **kw)
        changed = objective(dict(out, role_raw_quantiles=torch.full_like(out['quantiles'], 100)), *args, **kw)
        torch.testing.assert_close(losses['forecast_loss'], changed['forecast_loss'], rtol=0, atol=0)
        projected = objective(dict(out, quantiles=out['quantiles'] + .2), *args, **kw)
        self.assertNotEqual(losses['forecast_loss'].item(), projected['forecast_loss'].item())
        ps = list(self.a.parameters()); states = {id(p) for p in self.a.state_head.parameters()}
        sg = torch.autograd.grad(losses['state_loss'], ps, allow_unused=True, retain_graph=True)
        fg = torch.autograd.grad(losses['forecast_loss'], ps, allow_unused=True)
        for p, s, f in zip(ps, sg, fg):
            self.assertIsNone(f if id(p) in states else s)
            for grad in (s, f):
                if grad is not None: self.assertTrue(torch.isfinite(grad).all())
        projection = self.a.base.fusion_router.contextual_network[-1].weight
        grad = fg[next(i for i, p in enumerate(ps) if p is projection)]
        self.assertTrue((grad.reshape(3, 3, 64).abs().sum((1, 2)) > 0).all())

    def test_real_research_trainer_one_epoch_validation_and_optimizer_coverage(self):
        trainer = self.Trainer(self.a, self.a.make_objective(), model_id='C-LG', device='cpu',
            mixed_precision=False, separate_state_optimizer=True, max_epochs=1,
            selection_metric='daylight_qcrps')
        self.assertIsInstance(trainer, ResearchTrainer)
        groups = [trainer.forecast_parameters, trainer.state_parameters]
        ids = [{id(p) for p in group} for group in groups]
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual(ids[0] | ids[1], {id(p) for p in self.a.parameters() if p.requires_grad})
        self.assertEqual(ids[1], {id(p) for p in self.a.state_head.parameters()})
        initial = [[p.clone() for p in group] for group in groups]
        steps = [0, 0]; handles = []
        def after(index):
            def count(*args): steps[index] += 1
            return count
        for i, optimizer in enumerate((trainer.forecast_optimizer, trainer.state_optimizer)):
            handles.append(optimizer.register_step_post_hook(after(i)))
        rows = [{k: v[i] for k, v in self.b.items()} for i in range(2)]
        loader = DataLoader(rows, batch_size=2, shuffle=False)
        try: result = trainer.fit(loader, loader)
        finally:
            for h in handles: h.remove()
        self.assertEqual(steps, [1, 1])
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.best_epoch, 0)
        self.assertEqual(result.history[0].validation_metrics['valid_count'], 36.)
        for group, old in zip(groups, initial):
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(group, old)))
        self.assertEqual(trainer.model_id, 'C-LG')
        output = trainer.predict_batch(self.b)
        self.assertNotIn('gate_weights', output)
        altered = dict(self.b, targets=self.b['targets'].clone(), state_labels=self.b['state_labels'].clone())
        altered['targets'][:, 2] = 1000; altered['state_labels'][:, 2] = 2
        first, _ = trainer._criterion(output, self.b); second, _ = trainer._criterion(output, altered)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_state_dict_weights_only_roundtrip_exact_eval(self):
        self.a.eval()
        expected = self.a(**inputs(self.b))
        stream = io.BytesIO(); torch.save(self.a.state_dict(), stream); stream.seek(0)
        fresh = self.Adapter(control(47), backend='cpu-gru').eval()
        fresh.load_state_dict(torch.load(stream, map_location='cpu', weights_only=True), strict=True)
        actual = fresh(**inputs(self.b))
        for key, value in expected.items():
            if isinstance(value, torch.Tensor): torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        self.assertFalse(any('lock' in key for key in fresh.__dict__))

    def test_missing_power_and_failed_forward_cleanup(self):
        self.a.eval()
        b = batch(); b['history_values'][..., 0] = float('nan'); b['history_mask'][..., 0] = False
        output = self.a(**inputs(b))
        expected = b['future_known'][..., 0, None].expand(2, 24, 7)
        torch.testing.assert_close(output['quantiles'], expected, rtol=1e-6, atol=1e-6)
        with self.assertRaises(ValueError): self.a(**dict(inputs(b), history_values=b['history_values'][:, :5]))
        self.assertEqual(len(self.a.base.fusion_router._forward_hooks), 0)
        self.assertEqual(self.a(**inputs(b))['quantiles'].shape, (2, 24, 7))

    def test_identity_backend_and_training_guards(self):
        self.assertEqual(self.a.model_metadata['model_id'], 'C-LG')
        self.assertFalse(self.a.model_metadata['scientific_training_ready'])
        self.assertEqual(self.a.model_metadata['initialization'], 'fresh_control_shared_state_repeated_role_projection')
        self.assertEqual(sum(p.numel() for p in self.a.parameters()) - sum(p.numel() for p in self.c.parameters()), 390)
        with self.assertRaises(ValueError): self.Adapter(self.c, backend='cuda-mamba')
        with self.assertRaises(ValueError): self.a(**inputs(self.b), future_weather=torch.zeros(2, 24, 7))
        with self.assertRaises(ValueError):
            self.Trainer(self.a, self.a.make_objective(), model_id='C-LG', device='cpu', separate_state_optimizer=False)

    def test_worker_build_helpers_preserve_pair_training_start_rng(self):
        from racepv.exploratory.location_gap.lg_adapter import build, trainer
        from racepv.exploratory.location_gap.lg_policy import candidates
        from racepv.training.research_baseline_bridge import masked_research_loss
        signature = dict(geometry=dict(lookback=192, horizon=24, stride=4),
            clock=dict(sampling_interval_ns=900_000_000_000),
            history_columns=['power_norm', 'clear_sky_power_norm', 'clear_sky_residual', 'clear_sky_index'] + [f'h{i}' for i in range(10)],
            future_known_columns=['clear_sky_power_norm', 'f1', 'f2', 'f3', 'hour_sin', 'hour_cos', 'f6', 'f7', 'f8'],
            site_metadata_names=[f's{i}' for i in range(6)])
        for seed in (29, 47):
            outputs, states, proofs = [], [], []
            for row in [r for r in candidates() if r['seed'] == seed]:
                torch.manual_seed(seed)
                model, objective, proof = build(signature, row, backend='cpu-gru')
                runner = trainer(model, objective, row, device='cpu', checkpoint_path=None, metadata={})
                states.append(torch.get_rng_state().clone()); proofs.append(proof)
                model.eval(); outputs.append(runner.predict_batch(self.b))
                self.assertEqual(runner.max_epochs, 1)
                self.assertTrue(runner.separate_state_optimizer)
                self.assertEqual(runner.model_id, row['architecture'])
            self.assertTrue(torch.equal(*states))
            self.assertEqual(proofs[0]['canonical_shared_sha256'], proofs[1]['canonical_shared_sha256'])
            self.assertTrue(all(p['repeated_projection_verified'] for p in proofs))
            old = masked_research_loss(outputs[0]['quantiles'], self.b, model_id='tcn_qr')
            new = masked_research_loss(outputs[1]['quantiles'], self.b, model_id='tcn_qr')
            torch.testing.assert_close(old['loss'], new['loss'], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    print(f'LOCAL CPU-GRU SYNTHETIC ONLY; torch={torch.__version__}; CUDA build={torch.version.cuda}', flush=True)
    if sys.platform == 'win32' and torch.version.cuda is not None: raise RuntimeError('local tests require CPU-only torch')
    unittest.main(verbosity=2)
