import ast
import importlib.util
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("comparison", ROOT / "early decision/compare_attention.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
ed = compare.adapter


class IntegrationTest(unittest.TestCase):
    def test_report_roundtrip_plain_and_gzip(self):
        import gzip
        import json
        import tempfile
        data = {"steps": [{"score": 0.1234567890123456}], "label": "完整结果"}
        with tempfile.TemporaryDirectory(dir=ROOT / "early decision" / "tests") as folder:
            Path(folder).resolve().relative_to((ROOT / "early decision" / "tests").resolve())
            plain, zipped = Path(folder)/"report.json", Path(folder)/"report.json.gz"
            compare.write_report(data, plain)
            compare.write_report(data, zipped)
            self.assertEqual(json.loads(plain.read_bytes()), data)
            self.assertEqual(gzip.decompress(zipped.read_bytes()), plain.read_bytes())

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.w = torch.randn(12, 4, dtype=torch.float64)
        self.x = torch.randn(20, 12, dtype=torch.float64)
        self.v = torch.randn(20, 4, dtype=torch.float64)

    def engine(self, **kwargs):
        return ed.EarlyDecisionAttention(self.w, self.w, rank=kwargs.pop("rank", 4), budget=kwargs.pop("budget", 20), **kwargs)

    def test_full_keep_and_full_rank(self):
        for dtype in (torch.float32, torch.float64):
            e = ed.EarlyDecisionAttention(self.w.to(dtype), self.w.to(dtype), rank=4, budget=20)
            e.append(self.x.to(dtype), values=self.v.to(dtype))
            r = compare.evaluate(e, self.x[-1:].to(dtype), self.x[-1:].to(dtype) @ self.w.to(dtype))
            self.assertTrue(r["ref_result_allclose"])
            self.assertAlmostEqual(r["output_cosine"], 1., places=6)
            self.assertAlmostEqual(r["row_logits_cosine"], 1., places=6)

    def test_persistence_and_no_dense_qk_in_selector(self):
        e = self.engine(rank=2, budget=3)
        model = e.model
        e.append(self.x[:10], values=self.v[:10])
        features = e.cache.features.clone()
        e.append(self.x[10:11], values=self.v[10:11])
        self.assertIs(e.model, model)
        self.assertEqual(e.append_calls, 2)
        self.assertTrue(torch.equal(e.cache.features[:10], features))
        # Selector is independent of all actual K/V values and full projection weights.
        m1, _ = e.select(self.x[10:11])
        e.keys.fill_(float("nan"))
        e.wq.fill_(float("nan"))
        m2, _ = e.select(self.x[10:11])
        self.assertTrue(torch.equal(m1, m2))
        e.reset()
        self.assertEqual(len(e.cache.norms), 0)

    def test_bounds_superset_and_same_budget(self):
        e = self.engine(rank=1, budget=3, mode="topk_candidates")
        e.append(self.x, values=self.v)
        r = compare.evaluate(e, self.x[-1:], self.x[-1:] @ self.w)
        self.assertEqual(r["oracle_logits_requested_k"]["recall"], 1.)
        self.assertGreaterEqual(r["selected"], 3)
        self.assertIn("oracle_cosine_qk_same_budget", r)

    def test_rejections_and_zero(self):
        e = self.engine()
        with self.assertRaises(ValueError):
            e.append(self.x.float(), values=self.v)
        with self.assertRaises(ValueError):
            e.append(torch.empty(1, 12, device="meta", dtype=self.w.dtype), values=self.v[:1])
        with self.assertRaises(ValueError):
            e.select(self.x[:1])
        with self.assertRaises(ValueError):
            self.engine(rope_mode="rope")
        e.append(torch.zeros_like(self.x), values=torch.zeros_like(self.v))
        with self.assertRaises(ValueError):
            e.attend(self.x[:1], bias=torch.zeros(20))
        with self.assertRaises(ValueError):
            e.attend(self.x[:2])
        r = compare.evaluate(e, torch.zeros_like(self.x[:1]), torch.zeros(1, 4, dtype=self.w.dtype))
        self.assertEqual(r["output_cosine"], 1.)
        self.assertEqual(r["output_relative_l2"], 0.)

    def test_tensor_input_comparison(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "early decision/tests") as directory:
            fixture = Path(directory) / "trace.pt"
            torch.save({"wq": self.w, "wk": self.w, "x": self.x, "values": self.v}, fixture)
            result = compare.run(types.SimpleNamespace(seed=42, tensor_input=str(fixture)))
            self.assertEqual(result["input"], "user_tensor")
            self.assertEqual(len(result["input_file_sha256"]), 64)
            self.assertNotIn(str(fixture), str(result))
            self.assertTrue(all(r["ref_result_allclose"] for c in result["cases"] if c["budget"] == 20 for r in c["steps"]))

    def test_rope_explicit_and_full_keep(self):
        e = self.engine(rope_mode="pre_rope_heuristic")
        with self.assertRaises(ValueError):
            e.append(self.x, values=self.v)
        keys = compare.rotate(self.x @ self.w, torch.arange(20))
        e.append(self.x, keys=keys, values=self.v)
        with self.assertRaises(ValueError):
            e.attend(self.x[-1:])
        q = compare.rotate(self.x[-1:] @ self.w, torch.tensor([19]))
        self.assertTrue(compare.evaluate(e, self.x[-1:], q)["ref_result_allclose"])

    def test_dispatch_before_dense_or_old_selectors(self):
        tree = ast.parse((ROOT / "twilight/pyimpl/attention.py").read_text())
        forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "attention_forward")
        self.assertIsInstance(forward.body[0], ast.If)
        ns = {"torch": torch, "Optional": __import__("typing").Optional, "Tuple": __import__("typing").Tuple}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), "dispatch", "exec"), ns)
        mock = types.SimpleNamespace(_early_decision_forward=lambda *a, **k: "early")
        self.assertEqual(ns["attention_forward"](mock, self.x), "early")

    def test_runtime_stub_gqa_causal_reset(self):
        # Tests runtime adapter plumbing without claiming Transformers compatibility.
        def rotary(q, k, cos, sin, positions):
            q = torch.stack([compare.rotate(row, positions[0]) for row in q[0]])[None]
            k = torch.stack([compare.rotate(row, positions[0]) for row in k[0]])[None]
            return q, k
        stub = types.ModuleType("transformers.models.llama.modeling_llama")
        stub.apply_rotary_pos_emb = rotary
        stub.repeat_kv = lambda x, groups: x.repeat_interleave(groups, dim=1)
        class Cache:
            def __init__(self): self.n = 0
            def get_usable_length(self, n, layer): return self.n
            def update(self, k, v, layer, args): self.n += k.shape[-2]
        module = types.SimpleNamespace(num_heads=2, num_key_value_groups=2, num_key_value_heads=1,
                                       head_dim=4, layer_idx=0, sliding_window=None)
        module.q_proj = torch.nn.Linear(12, 8, bias=False).double()
        module.k_proj = torch.nn.Linear(12, 4, bias=False).double()
        module.v_proj = torch.nn.Linear(12, 4, bias=False).double()
        module.o_proj = torch.nn.Linear(8, 12, bias=False).double()
        module.rotary_emb = lambda v, p: (None, None)
        with self.assertRaises(ValueError):
            ed.enable_early_decision(module, rank=4, budget=100, rope_mode="none")
        module.config = types.SimpleNamespace(sliding_window=64)
        with self.assertRaises(ValueError):
            ed.enable_early_decision(module, rank=4, budget=100, rope_mode="pre_rope_heuristic")
        module.config.sliding_window = None
        ed.enable_early_decision(module, rank=4, budget=100, rope_mode="pre_rope_heuristic")
        old_heads = module._early_decision_heads
        with self.assertRaises(ValueError):
            ed.enable_early_decision(module, rank=100, budget=100, rope_mode="pre_rope_heuristic")
        self.assertIs(module._early_decision_heads, old_heads)
        cache = Cache()
        with patch.dict(sys.modules, {stub.__name__: stub}):
            out = module._early_decision_forward(self.x[:3][None], position_ids=torch.arange(3)[None], past_key_value=cache, use_cache=True)[0]
            self.assertEqual(out.shape, (1, 3, 12))
            # Independently compute causal two-head GQA post-rotation dense output.
            xx = self.x[:3]
            k = compare.rotate(module.k_proj(xx), torch.arange(3))
            v = module.v_proj(xx)
            q = module.q_proj(xx).view(3, 2, 4)
            reference = []
            for t in range(3):
                hs = []
                for h in range(2):
                    qq = compare.rotate(q[t:t+1, h], torch.tensor([t]))
                    hs.append(((qq @ k[:t+1].T / 2).softmax(-1, dtype=torch.float32).double() @ v[:t+1]))
                reference.append(torch.cat(hs, -1))
            expected = module.o_proj(torch.cat(reference))[None]
            torch.testing.assert_close(out, expected)
            self.assertEqual(cache.n, 3)
            module._early_decision_forward(self.x[3:4][None], position_ids=torch.tensor([[3]]), past_key_value=cache, use_cache=True)
            self.assertEqual(len(module._early_decision_heads[0].cache.norms), 4)
            with self.assertRaises(ValueError):
                module._early_decision_forward(self.x[:1][None], position_ids=torch.tensor([[0]]), past_key_value=cache, use_cache=True)
            with self.assertRaises(ValueError):
                module._early_decision_forward(self.x[:1][None].expand(2, -1, -1), position_ids=torch.tensor([[4]]))
            module.early_decision_reset()
            a = module._early_decision_forward(self.x[:2][None], position_ids=torch.tensor([[0, 1]]))[0]
            b = module._early_decision_forward(self.x[:2][None], position_ids=torch.tensor([[0, 1]]))[0]
            torch.testing.assert_close(a, b)
            self.assertEqual(len(module._early_decision_heads[0].cache.norms), 0)


if __name__ == "__main__":
    unittest.main()
