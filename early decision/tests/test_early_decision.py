import math
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from early_decision import (ScoreBounds, fit_bilinear, mass_candidates,
                            mass_lower_bound, quantize_projection, topk_candidates)


class EarlyDecisionTests(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(71)

    def rand(self, *shape):
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def example(self, rank=3, mode="tail", d_input=12, d_head=6, transform=None):
        wq, wk = self.rand(d_input, d_head), self.rand(d_input, d_head)
        xq, xk = self.rand(d_input), self.rand(19, d_input)
        model = fit_bilinear(wq, wk, rank, score_transform=transform)
        cache = model.new_cache(mode)
        cache.append(xk)
        q = xq @ wq
        if transform is not None:
            q = q @ transform
        exact = q @ (xk @ wk).T / math.sqrt(d_head)
        return model, cache, xq, exact

    def test_full_rank_recovers_bilinear_scores(self):
        for dim, head in ((12, 6), (4, 7)):
            _, cache, xq, exact = self.example(min(dim, head), d_input=dim, d_head=head)
            torch.testing.assert_close(cache.bounds(xq).estimate, exact)

    def test_both_interval_modes_cover_exact_scores_at_all_ranks(self):
        for mode in ("tail", "norm"):
            for rank in range(7):
                _, cache, xq, exact = self.example(rank, mode)
                bounds = cache.bounds(xq)
                self.assertTrue(bool((bounds.lower <= exact).all()))
                self.assertTrue(bool((exact <= bounds.upper).all()))

    def test_topk_superset_for_all_ranks(self):
        for rank in range(7):
            _, cache, xq, exact = self.example(rank)
            bounds = cache.bounds(xq)
            for k in (1, 3, 19):
                decision = topk_candidates(bounds, k)
                self.assertTrue(bool(decision.mask[torch.topk(exact, k).indices].all()))

    def test_mass_bound_and_selection_hold(self):
        for rank in range(7):
            _, cache, xq, exact = self.example(rank)
            bounds = cache.bounds(xq)
            for p in (0.1, 0.8, 0.95, 1.0):
                decision = mass_candidates(bounds, p)
                actual = float(exact.softmax(-1)[decision.mask].sum())
                self.assertGreaterEqual(actual + 1e-12, decision.mass_lower_bound)
                self.assertGreaterEqual(decision.mass_lower_bound + 1e-12, p)

    def test_log_space_avoids_large_logit_overflow(self):
        scores = torch.tensor([10000., 9999., -10000.], dtype=torch.float64)
        bounds = ScoreBounds(scores, scores - .1, scores + .1)
        decision = mass_candidates(bounds, .9)
        self.assertTrue(math.isfinite(decision.mass_lower_bound))
        self.assertGreaterEqual(float(scores.softmax(0)[decision.mask].sum()), .9)

    def test_ties_are_conservative_for_topk(self):
        zero = torch.zeros(5, dtype=torch.float64)
        bounds = ScoreBounds(zero, zero, zero)
        self.assertEqual(int(topk_candidates(bounds, 1).mask.sum()), 5)
        self.assertEqual(int(mass_candidates(bounds, .8).mask.sum()), 4)

    def test_tail_mode_tighter_than_input_norm(self):
        model, tail, xq, _ = self.example()
        xk = self.rand(19, 12)
        tail = model.new_cache("tail")
        norm = model.new_cache("norm")
        tail.append(xk)
        norm.append(xk)
        self.assertTrue(bool((tail.bounds(xq).upper <= norm.bounds(xq).upper + 1e-12).all()))

    def test_append_retains_only_features_and_norms(self):
        model, _, xq, _ = self.example()
        xk = self.rand(19, 12)
        one, chunks = model.new_cache(), model.new_cache()
        one.append(xk)
        chunks.append(xk[:7])
        chunks.append(xk[7:])
        torch.testing.assert_close(one.bounds(xq).estimate, chunks.bounds(xq).estimate)
        self.assertEqual(chunks.storage_bytes, 19 * (3 + 1) * 8)
        self.assertNotIn("x_keys", vars(chunks))

    def test_fixed_score_transform_is_supported(self):
        transform, _ = torch.linalg.qr(self.rand(6, 6))
        _, cache, xq, exact = self.example(6, transform=transform)
        torch.testing.assert_close(cache.bounds(xq).estimate, exact)

    def test_rope_cannot_be_silently_ignored(self):
        wq = wk = torch.eye(2, dtype=torch.float64)
        xq = torch.tensor([1., 0.], dtype=torch.float64)
        xk = torch.tensor([[1., 0.]], dtype=torch.float64)
        cache = fit_bilinear(wq, wk, 2).new_cache()
        cache.append(xk)
        rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        real_rope = (xq @ wq) @ (xk @ wk @ rotation.T).T / math.sqrt(2)
        self.assertGreater(float((cache.bounds(xq).estimate - real_rope).abs().max()), .5)
        with self.assertRaises(ValueError):
            fit_bilinear(wq, wk, 1, score_transform=rotation[None])

    def test_quantized_reference_is_covered(self):
        wq, wk = quantize_projection(self.rand(12, 6), 4), quantize_projection(self.rand(12, 6), 8)
        model = fit_bilinear(wq, wk, 3)
        cache = model.new_cache()
        xq, xk = self.rand(12), self.rand(19, 12)
        cache.append(xk)
        exact = (xq @ wq) @ (xk @ wk).T / math.sqrt(6)
        bounds = cache.bounds(xq)
        self.assertTrue(bool(((bounds.lower <= exact) & (exact <= bounds.upper)).all()))

    def test_invalid_shapes_nonfinite_and_arguments(self):
        w = self.rand(12, 6)
        for bad in (-1, 7, True, 1.2):
            with self.assertRaises(ValueError):
                fit_bilinear(w, w, bad)
        for bad in (float("nan"), -1):
            with self.assertRaises(ValueError):
                fit_bilinear(w, w, 2, guard=bad)
        with self.assertRaises(ValueError):
            fit_bilinear(w, self.rand(12, 5), 2)
        with self.assertRaises(ValueError):
            fit_bilinear(w * float("nan"), w, 2)
        cache = fit_bilinear(w, w, 2).new_cache()
        with self.assertRaises(ValueError):
            cache.bounds(self.rand(12))
        with self.assertRaises(ValueError):
            cache.append(self.rand(2, 11))
        cache.append(self.rand(2, 12))
        bounds = cache.bounds(self.rand(12))
        for p in (0, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                mass_candidates(bounds, p)
        for k in (0, 3, False):
            with self.assertRaises(ValueError):
                topk_candidates(bounds, k)

    def test_empty_and_full_subset_mass(self):
        _, cache, xq, _ = self.example()
        bounds = cache.bounds(xq)
        self.assertEqual(mass_lower_bound(bounds, torch.zeros(19, dtype=torch.bool)), 0)
        self.assertEqual(mass_lower_bound(bounds, torch.ones(19, dtype=torch.bool)), 1)

    def test_zero_weights_and_zero_input(self):
        weights = torch.zeros(8, 4, dtype=torch.float64)
        cache = fit_bilinear(weights, weights, 2).new_cache()
        cache.append(self.rand(9, 8))
        bounds = cache.bounds(torch.zeros(8, dtype=torch.float64))
        torch.testing.assert_close(bounds.estimate, torch.zeros(9, dtype=torch.float64))
        self.assertEqual(int(topk_candidates(bounds, 2).mask.sum()), 9)
        self.assertGreaterEqual(mass_candidates(bounds, .9).mass_lower_bound, .9)

    def test_shared_weights_do_not_imply_fixed_token_ranking(self):
        identity = torch.eye(2, dtype=torch.float64)
        cache = fit_bilinear(identity, identity, 2).new_cache()
        cache.append(torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64))
        positive = cache.bounds(torch.tensor([1., 0.], dtype=torch.float64))
        negative = cache.bounds(torch.tensor([-1., 0.], dtype=torch.float64))
        self.assertNotEqual(int(positive.estimate.argmax()), int(negative.estimate.argmax()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
