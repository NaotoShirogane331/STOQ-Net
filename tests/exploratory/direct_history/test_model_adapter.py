"""Real frozen-source GRU tests; not Mamba, CUDA, datasets or scientific results."""
from dataclasses import replace
import importlib.util
import io
import unittest
import torch
from .fixtures import make_config, make_inputs, paired_models, reset_rng, rng_state, restore_rng
from racepv.reliability_experiment import ReliabilityControlRaCEPV
from racepv.training.engine import EarlyStoppingTrainer
from .fixtures import make_batch


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('racepv.exploratory.direct_history.model_adapter'), 'C direct-history adapter not implemented')
        from racepv.exploratory.direct_history.model_adapter import DirectHistoryC
        self.Candidate = DirectHistoryC
        self.control, self.model = paired_models(DirectHistoryC)
        self.inputs = make_inputs()

    def assert_rng_equal(self, left, right):
        self.assertEqual(left[0], right[0])
        self.assertEqual(left[1][0], right[1][0])
        self.assertTrue((left[1][1] == right[1][1]).all())
        self.assertEqual(left[1][2:], right[1][2:])
        self.assertTrue(torch.equal(left[2], right[2]))

    def test_shared_init_and_rng(self):
        for seed in (29, 47):
            reset_rng(seed)
            original = ReliabilityControlRaCEPV(make_config())
            old_rng = rng_state()
            reset_rng(seed)
            candidate = self.Candidate(make_config())
            self.assert_rng_equal(old_rng, rng_state())
            shared = original.state_dict()
            for k, v in shared.items():
                self.assertTrue(torch.equal(v, candidate.state_dict()[k]), k)
            self.assertEqual(set(candidate.state_dict()) - set(shared),
                             {'direct_history_head.weight', 'direct_history_head.bias'})
            self.assertEqual(sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in original.parameters()), 776)
        original29, _ = paired_models(self.Candidate, 29)
        self.assertTrue(any(not torch.equal(v, original.state_dict()[k])
                            for k, v in original29.state_dict().items()))

    def test_zero_forward_eval_and_train(self):
        for training in (False, True):
            self.control.train(training)
            self.model.train(training)
            before = rng_state()
            old = self.control(**self.inputs)
            old_rng = rng_state()
            restore_rng(before)
            new = self.model(**self.inputs)
            self.assert_rng_equal(old_rng, rng_state())
            for k, v in old.items():
                if isinstance(v, torch.Tensor):
                    torch.testing.assert_close(new[k], v, rtol=0, atol=0, msg=k)
                else:
                    self.assertEqual(new[k], v)
            self.assertTrue((new['direct_history_delta'] == 0).all())
            self.assertFalse(new['direct_history_delta'].requires_grad)

    def test_one_forward_per_original_component(self):
        counts = dict(stable_expert=0, transient_expert=0, fusion_router=0, state_head=0)
        handles = []
        for name in counts:
            def increment(_mod, _args, _out, name=name):
                counts[name] += 1
            handles.append(getattr(self.model, name).register_forward_hook(increment))
        try:
            self.model(**self.inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, dict.fromkeys(counts, 1))

    def test_nonzero_locality_order_and_projection(self):
        self.model.eval()
        with torch.no_grad():
            before = self.model(**self.inputs)
            self.model.direct_history_head.bias.copy_(torch.tensor([1., -1., 2., -2., 3., -3., 100., -100.]))
            after = self.model(**self.inputs)
        q = after['quantiles']
        self.assertEqual(q.shape, (2, 24, 7))
        self.assertEqual(q.dtype, before['quantiles'].dtype)
        self.assertEqual(q.device, before['quantiles'].device)
        self.assertTrue((q.diff(dim=-1) >= 0).all())
        self.assertTrue(((q >= 0) & (q <= 1.05)).all())
        self.assertTrue(torch.equal(q[:, 8:], before['quantiles'][:, 8:]))
        self.assertTrue((after['direct_history_delta'][:, 8:] == 0).all())
        self.assertTrue((q[:, ::5] == 0).all())
        expected = (before['quantiles'][:, :8] + after['direct_history_delta'][:, :8, None]).clamp(0, 1.05)
        torch.testing.assert_close(q[:, :8], expected, rtol=0, atol=0)
        self.assertGreater((q - before['quantiles']).abs().sum().item(), 0.)
        for key in before:
            if key != 'quantiles' and not key.startswith('direct_history_'):
                torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(q.clamp(0, 1.05), q))

    def test_recent_valid_zero_not_missing(self):
        self.model.eval()
        self.inputs['history_values'][:, -24:, 0] = 0
        with torch.no_grad():
            self.model.direct_history_head.bias.fill_(.5)
            available = self.model(**self.inputs)
            self.inputs['history_mask'][:, -24:, 0] = False
            missing = self.model(**self.inputs)
        self.assertTrue(available['direct_history_valid'].all())
        self.assertGreater(available['direct_history_delta'][:, 1].min().item(), 0)
        self.assertFalse(missing['direct_history_valid'].any())
        self.assertTrue((missing['direct_history_delta'] == 0).all())
        self.assertTrue(torch.equal(missing['quantiles'], missing['base_quantiles']))

    def test_actual_lower_and_upper_clipping_boundaries(self):
        self.model.eval()
        heads = (self.model.stable_expert.quantile_head,
                 self.model.transient_expert.up_quantile_head,
                 self.model.transient_expert.down_quantile_head)
        for sign in (-1., 1.):
            with self.subTest(sign=sign), torch.no_grad():
                # Change only disposable fixture weights. All original expert calls remain real.
                for head in heads:
                    head.projection.weight.zero_()
                    head.projection.bias.zero_()
                    head.projection.bias[0] = 10. * sign
                self.model.direct_history_head.bias.zero_()
                baseline = self.model(**self.inputs)['quantiles']
                self.model.direct_history_head.bias.fill_(sign)
                output = self.model(**self.inputs)
                pre_projection = baseline[:, :8] + output['direct_history_delta'][:, :8, None]
                crosses = pre_projection < 0. if sign < 0 else pre_projection > 1.05
                self.assertTrue(crosses.any(), 'fixture must actually require clipping')
                q = output['quantiles']
                expected = pre_projection.clamp(0., 1.05)
                torch.testing.assert_close(q[:, :8], expected, rtol=0, atol=0)
                self.assertTrue((q[:, :8][crosses] == (0. if sign < 0 else 1.05)).all())
                self.assertTrue((q.diff(dim=-1) >= 0).all())
                self.assertTrue(torch.equal(q[:, 8:], baseline[:, 8:]))
                self.assertTrue((q[:, ::5] == 0).all())

    def test_input_immutability_and_nonfinite_future_rejection(self):
        self.model.eval()
        self.inputs['future_known'][:, 1, 0] = float('nan')
        for model in (self.control, self.model):
            with self.assertRaisesRegex(ValueError, 'future_known must be finite'):
                model(**self.inputs)
        self.inputs['future_known'][:, 1, 0] = 0.
        self.inputs['history_values'][:, -1, 0] = float('inf')
        snapshots = {k: v.clone() for k, v in self.inputs.items()}
        with torch.no_grad():
            self.model.direct_history_head.bias.fill_(1)
            out = self.model(**self.inputs)
        self.assertTrue(torch.isfinite(out['quantiles']).all())
        self.assertTrue((out['quantiles'][:, 1] == 0).all())
        for key in self.inputs:
            torch.testing.assert_close(self.inputs[key], snapshots[key], rtol=0, atol=0, equal_nan=True)

    def test_strict_state_roundtrip(self):
        self.model.eval()
        with torch.no_grad():
            self.model.direct_history_head.weight.fill_(.001)
        stream = io.BytesIO()
        torch.save(self.model.state_dict(), stream)
        stream.seek(0)
        restored = self.Candidate(make_config()).eval()
        restored.load_state_dict(torch.load(stream, map_location='cpu', weights_only=True), strict=True)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        with torch.no_grad():
            torch.testing.assert_close(restored(**self.inputs)['quantiles'], self.model(**self.inputs)['quantiles'], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(self.control.state_dict(), strict=True)

    def test_metadata_and_backend_rejection(self):
        meta = self.model.model_metadata
        self.assertEqual(meta['candidate_id'], 'C-DH-linear')
        self.assertEqual(meta['parent_model_revision'], self.control.model_metadata['model_revision'])
        self.assertNotEqual(meta['model_revision'], meta['parent_model_revision'])
        self.assertEqual(meta['actual_backbone'], 'causal_conv_gru')
        self.assertFalse(meta['scientific_training_ready'])
        self.assertTrue(meta['cpu_contract_only'])
        for delta in ({'backbone': 'auto'}, {'horizon': 12}, {'router_temperature': 2.},
                      {'tail_residual_mode': 'sparse_ramp_up'}, {'history_clear_sky_index': 2},
                      {'dropout': .2}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.Candidate(replace(make_config(), **delta))
        with self.assertRaises(ValueError):
            self.model(**self.inputs, future_weather=torch.zeros(2, 24, 7))

    def test_head_fp32_boundary_and_half_rejection(self):
        from racepv.exploratory.direct_history.history_head import extract_recent_features
        x, valid = extract_recent_features(self.inputs['history_values'], self.inputs['history_mask'])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            delta, _ = self.model.direct_history_head(x, valid)
        self.assertEqual(delta.dtype, torch.float32)
        self.model.half()
        with self.assertRaises((TypeError, ValueError)):
            self.model(**self.inputs)

    def test_optimizer_partition(self):
        trainer = EarlyStoppingTrainer(self.model, self.model.make_objective(), device='cpu',
            mixed_precision=False, separate_state_optimizer=True, learning_rate=.001,
            weight_decay=.0001, gradient_clip_norm=1.)
        forecast = [id(p) for p in trainer.forecast_parameters]
        state = [id(p) for p in trainer.state_parameters]
        all_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        self.assertFalse(set(forecast) & set(state))
        self.assertEqual(set(forecast) | set(state), all_ids)
        self.assertEqual(len(forecast) + len(state), len(all_ids))
        self.assertEqual(set(state), {id(p) for p in self.model.state_head.parameters()})
        for p in self.model.direct_history_head.parameters():
            self.assertEqual(forecast.count(id(p)), 1)
        for optimizer in (trainer.forecast_optimizer, trainer.state_optimizer):
            self.assertEqual(optimizer.param_groups[0]['lr'], .001)
            self.assertEqual(optimizer.param_groups[0]['weight_decay'], .0001)

    def test_original_objective_and_gradient_isolation(self):
        self.control.eval()
        self.model.eval()
        batch = make_batch()
        objective = self.model.make_objective()
        old, new = self.control(**self.inputs), self.model(**self.inputs)
        args = (batch['targets'], batch['state_labels'])
        kwargs = dict(daylight_mask=batch['daylight_mask'], target_mask=batch['target_mask'])
        a, b = objective(old, *args, **kwargs), objective(new, *args, **kwargs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        params = list(self.model.parameters())
        state_ids = {id(p) for p in self.model.state_head.parameters()}
        forecast_grads = torch.autograd.grad(b['forecast_loss'], params, allow_unused=True, retain_graph=True)
        state_grads = torch.autograd.grad(b['state_loss'], params, allow_unused=True)
        some_base_gradient = False
        for p, fg, sg in zip(params, forecast_grads, state_grads):
            if id(p) in state_ids:
                self.assertIsNone(fg)
            else:
                self.assertIsNone(sg)
                if id(p) not in {id(q) for q in self.model.direct_history_head.parameters()} and fg is not None:
                    some_base_gradient |= bool((fg != 0).any())
            for grad in (fg, sg):
                if grad is not None:
                    self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(some_base_gradient)
        for hp in self.model.direct_history_head.parameters():
            grad = forecast_grads[next(i for i, p in enumerate(params) if p is hp)]
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_original_criterion_target_mask_and_real_cpu_steps(self):
        self.model.eval()
        batch = make_batch()
        batch['target_mask'][:, 2:4] = False
        trainer = EarlyStoppingTrainer(self.model, self.model.make_objective(), device='cpu',
            mixed_precision=False, separate_state_optimizer=True, learning_rate=.001, weight_decay=.0001)
        output = self.model(**self.inputs)
        loss, terms = trainer._criterion(output, batch)
        batch['targets'][:, 2:4] = float('nan')
        changed_loss, changed_terms = trainer._criterion(output, batch)
        torch.testing.assert_close(changed_loss, loss, rtol=0, atol=0)
        saved = {k: p.clone() for k, p in self.model.named_parameters()}
        trainer._step_separated_losses(forecast_loss=terms['forecast_loss'], state_loss=terms['state_loss'])
        self.assertFalse(torch.equal(saved['direct_history_head.weight'], self.model.direct_history_head.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
        for optimizer in (trainer.forecast_optimizer, trainer.state_optimizer):
            steps = [int(v['step']) for v in optimizer.state.values() if 'step' in v]
            self.assertTrue(steps)
            self.assertTrue(all(s == 1 for s in steps))

