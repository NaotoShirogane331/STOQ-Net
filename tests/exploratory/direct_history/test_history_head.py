"""Catch channel/time/mask corruption, random initialization and detached learning."""
import importlib.util
import unittest
import torch


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('racepv.exploratory.direct_history.history_head'), 'history extractor not implemented')
        from racepv.exploratory.direct_history.history_head import extract_recent_features
        self.extract = extract_recent_features
        self.h = torch.zeros(2, 192, 14)
        self.m = torch.ones_like(self.h, dtype=torch.bool)

    def test_channel_major_and_last24(self):
        self.h[:, -24:, 0] = torch.arange(24) / 32
        self.h[:, -24:, 1] = .25
        x, valid = self.extract(self.h, self.m)
        self.assertEqual(x.shape, (2, 96))
        self.assertEqual(x.dtype, torch.float32)
        torch.testing.assert_close(x[0, :24], torch.arange(24) / 32, rtol=0, atol=0)
        torch.testing.assert_close(x[0, 24:48], torch.arange(24) / 32 - .25, rtol=0, atol=0)
        self.assertTrue(torch.equal(x[:, 48:], torch.ones(2, 48)))
        self.assertTrue(valid.all())
        changed = self.h.clone()
        changed[:, :168] = 99
        changed[:, :, 2:] = -99
        torch.testing.assert_close(self.extract(changed, self.m)[0], x, rtol=0, atol=0)

    def test_masks_nan_inf_and_observed_zero(self):
        self.h[:, -24:, 0] = .5
        self.h[:, -24:, 1] = .25
        self.h[0, -24, 0] = float('nan')
        self.h[0, -23, 1] = float('inf')
        self.m[0, -22, 0] = False
        self.m[0, -21, 1] = False
        self.h[0, -20, 0] = 0
        self.h[0, -19, 0] = -float('inf')
        self.m[1, -24:, 0] = False
        x, valid = self.extract(self.h, self.m)
        self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(x[0, :6], torch.tensor([0., .5, 0., .5, 0., 0.]), rtol=0, atol=0)
        torch.testing.assert_close(x[0, 24:30], torch.tensor([0., 0., 0., 0., -.25, 0.]), rtol=0, atol=0)
        torch.testing.assert_close(x[0, 48:54], torch.tensor([0., 1., 0., 1., 1., 0.]), rtol=0, atol=0)
        torch.testing.assert_close(x[0, 72:78], torch.tensor([0., 0., 0., 0., 1., 0.]), rtol=0, atol=0)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue((x[1] == 0).all())

    def test_invalid_contract(self):
        cases = [(self.h[:0], self.m[:0]), (self.h[:, :191], self.m[:, :191]),
                 (self.h[:, :, :13], self.m[:, :, :13]), (self.h, self.m[:, :5]),
                 (self.h.long(), self.m), (self.h.to(torch.complex64), self.m),
                 (self.h, torch.full_like(self.h, 2)), (self.h, torch.full_like(self.h, float('nan')))]
        for h, m in cases:
            with self.subTest(shape=h.shape, dtype=h.dtype), self.assertRaises((TypeError, ValueError)):
                self.extract(h, m)

    def test_inputs_are_not_mutated(self):
        before_h, before_m = self.h.clone(), self.m.clone()
        self.extract(self.h, self.m)
        self.assertTrue(torch.equal(self.h, before_h))
        self.assertTrue(torch.equal(self.m, before_m))


class HeadTests(unittest.TestCase):
    def setUp(self):
        from racepv.exploratory.direct_history import history_head
        self.assertTrue(hasattr(history_head, 'BoundedHistoryHead'), 'bounded head not implemented')
        self.Head = history_head.BoundedHistoryHead

    def test_zero_parameters_do_not_consume_rng(self):
        rng = torch.get_rng_state()
        head = self.Head()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 776)
        delta, terms = head(torch.ones(2, 96), torch.tensor([True, False]))
        self.assertEqual(delta.shape, (2, 8))
        self.assertTrue((delta == 0).all())
        self.assertTrue(all((p == 0).all() for p in head.parameters()))

    def test_bound_sign_fallback_and_terms(self):
        head = self.Head()
        with torch.no_grad():
            head.weight[0, 0] = 2
            head.weight[0, 48] = 3
            head.bias[0] = 1
            head.bias[1] = -1
            head.bias[2] = 100
        x = torch.zeros(2, 96)
        x[:, 0], x[:, 48] = .25, 1.
        delta, terms = head(x, torch.tensor([True, False]))
        self.assertTrue((delta.abs() <= .08).all())
        torch.testing.assert_close(delta[0, :3], .08 * torch.tanh(torch.tensor([4.5, -1., 100.])))
        self.assertTrue((delta[1] == 0).all())
        self.assertEqual(terms['value_term'][0, 0].item(), .5)
        self.assertEqual(terms['mask_term'][0, 0].item(), 3.)
        self.assertEqual(terms['bias_term'][0, 0].item(), 1.)
        self.assertTrue(all(not t.requires_grad for t in terms.values()))

    def test_head_gradient_and_fp32_under_cpu_autocast(self):
        head = self.Head()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            delta, terms = head(torch.ones(2, 96), torch.ones(2, dtype=torch.bool))
        self.assertEqual(delta.dtype, torch.float32)
        delta.sum().backward()
        for p in head.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertTrue((p.grad != 0).all())
        self.assertTrue(all(t.dtype == torch.float32 for t in terms.values()))

    def test_invalid_head_inputs_and_parameter_cast(self):
        head = self.Head()
        for x, valid in ((torch.ones(2, 95), torch.ones(2, dtype=torch.bool)),
                         (torch.ones(2, 96), torch.ones(2)),
                         (torch.ones(2, 96), torch.ones(3, dtype=torch.bool)),
                         (torch.full((2, 96), float('nan')), torch.ones(2, dtype=torch.bool))):
            with self.subTest(shape=x.shape), self.assertRaises((TypeError, ValueError)):
                head(x, valid)
        with self.assertRaises((TypeError, ValueError)):
            head.half()(torch.ones(2, 96), torch.ones(2, dtype=torch.bool))

