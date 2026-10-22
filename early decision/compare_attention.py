"""Reproducible CPU component comparison; no model/checkpoint dependency."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import statistics
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ed_attention", ROOT / "twilight/early_decision_attention.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = float(a.norm()), float(b.norm())
    return float(a @ b / (na * nb)) if na and nb else float(na == nb)


def overlap(mask, scores, k):
    oracle = torch.zeros_like(mask)
    oracle[torch.topk(scores, k).indices] = True
    hit, b = int((mask & oracle).sum()), int(mask.sum())
    p, r = hit / b, hit / k
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.,
            "jaccard": hit/(b+k-hit)}


def rotate(x, positions):
    d = x.shape[-1]
    phase = positions[:, None].to(x) / (10000 ** (torch.arange(d//2, dtype=x.dtype) / (d//2)))
    a, b = x[:, :d//2], x[:, d//2:]
    return torch.cat((a*phase.cos()-b*phase.sin(), b*phase.cos()+a*phase.sin()), -1)


def evaluate(engine, xq, query, *, repeats=7, warmup=2):
    # Oracle phase starts only after RESULT has selected and executed attention.
    for _ in range(warmup):
        engine.attend(xq, query=query)
    timings, selection, execution = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        result, mask, bounds = engine.attend(xq, query=query)
        timings.append((time.perf_counter()-start)*1000)
        selection.append(engine.last["selection_ms"])
        execution.append(engine.last["selected_qk_av_ms"])
    warm = statistics.median(timings)
    engine.last["selection_ms"] = statistics.median(selection)
    engine.last["selected_qk_av_ms"] = statistics.median(execution)
    engine.last["selected"] = int(mask.sum())
    dense_times = []
    for i in range(warmup+repeats):
        start = time.perf_counter()
        scores = (query @ engine.keys.T / math.sqrt(query.shape[-1]))[0]
        p = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        ref = p[None] @ engine.values
        if i >= warmup:
            dense_times.append((time.perf_counter()-start)*1000)
    dense_ms = statistics.median(dense_times)
    denom = query.norm() * engine.keys.norm(dim=-1)
    cosqk = torch.where(denom > 0, (engine.keys @ query[0])/denom.clamp_min(1e-30), 0.)
    sparse_p = torch.zeros_like(p)
    sparse_p[mask] = scores[mask].softmax(-1, dtype=torch.float32).to(query.dtype)
    err = (result-ref).double()
    k, b = min(engine.budget, len(mask)), int(mask.sum())
    stats = {"n": len(mask), "requested_k": k, "selected": b,
             "selected_indices": torch.nonzero(mask).flatten().tolist(),
             "token_cosine_qk": cosqk.tolist(),
             "output_cosine": cosine(result, ref),
             "output_relative_l2": float(err.norm()/ref.double().norm().clamp_min(1e-30)),
             "output_nrmse": float(err.square().mean().sqrt()/ref.double().square().mean().sqrt().clamp_min(1e-30)),
             "output_max_abs": float(err.abs().max()),
             "probability_tv": float((p-sparse_p).abs().sum()/2),
             "retained_true_softmax_mass": float(p[mask].sum()),
             "row_logits_cosine": cosine(bounds.estimate, scores),
             "row_logits_relative_l2": float((bounds.estimate-scores).norm()/scores.norm().clamp_min(1e-30)),
             "oracle_logits_requested_k": overlap(mask, scores, k),
             "oracle_logits_same_budget": overlap(mask, scores, b),
             "oracle_cosine_qk_requested_k": overlap(mask, cosqk, k),
             "oracle_cosine_qk_same_budget": overlap(mask, cosqk, b),
             "cpu_warm_step_ms": warm, "cpu_dense_qk_av_ms": dense_ms,
             "cpu_warm_samples_ms": timings, "cpu_dense_samples_ms": dense_times,
             "phase_cpu_ms": dict(engine.last), "costs": engine.costs(),
             "ref_result_allclose": bool(torch.allclose(ref, result, atol=1e-6, rtol=1e-5))}
    return stats


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cases = []
    input_hash = None
    if args.tensor_input:
        input_hash = hashlib.sha256(Path(args.tensor_input).read_bytes()).hexdigest()
        bundle = torch.load(args.tensor_input, map_location="cpu", weights_only=True)
        sources = [("user_tensor", bundle["wq"], bundle["wk"], bundle["x"], bundle["values"])]
    else:
        D, d, N = 64, 16, 256
        basisq, _ = torch.linalg.qr(torch.randn(D, d, dtype=torch.float64))
        basisk, _ = torch.linalg.qr(torch.randn(D, d, dtype=torch.float64))
        x, values = torch.randn(N+4, D, dtype=torch.float64), torch.randn(N+4, d, dtype=torch.float64)
        sources = []
        for spectrum in ("favorable", "flat"):
            sigma = torch.logspace(1, -5, d, dtype=torch.float64) if spectrum == "favorable" else torch.ones(d, dtype=torch.float64)
            sources.append((spectrum, basisq*sigma.sqrt(), basisk*sigma.sqrt(), x, values))
    for source, rawq, rawk, x, values in sources:
        d = rawq.shape[1]
        if d % 2 or len(x) <= 4:
            raise ValueError("comparison requires even head dimension and more than four input rows")
        n = len(x)-4
        for bits in (None, 4, 8):
            wq = rawq if bits is None else adapter.core.quantize_projection(rawq, bits).to(rawq.dtype)
            wk = rawk if bits is None else adapter.core.quantize_projection(rawk, bits).to(rawk.dtype)
            for rank in (max(1, d//4), d):
                for budget in (max(1, n//8), max(1, n//2), len(x)):
                    for mode in ("estimate_topk", "topk_candidates"):
                        for rope in ("none", "pre_rope_heuristic"):
                            engine = adapter.EarlyDecisionAttention(wq, wk, rank=rank, budget=budget, mode=mode, rope_mode=rope)
                            keys = x @ wk
                            if rope != "none":
                                keys = rotate(keys, torch.arange(len(x)))
                            start = time.perf_counter()
                            engine.append(x[:n], keys=keys[:n], values=values[:n])
                            build_ms = (time.perf_counter()-start)*1000
                            rows = []
                            for t in range(n, len(x)):
                                engine.append(x[t:t+1], keys=keys[t:t+1], values=values[t:t+1])
                                q = x[t:t+1] @ wq
                                if rope != "none":
                                    q = rotate(q, torch.tensor([t]))
                                rows.append(evaluate(engine, x[t:t+1], q))
                            D = wq.shape[0]
                            extra_history_mac = n*D*rank
                            warm_mean = statistics.mean(r["cpu_warm_step_ms"]+r["phase_cpu_ms"]["append_ms"] for r in rows)
                            cases.append({"source": source, "weight_bits": bits or "FP", "rank": rank,
                                          "budget": budget, "mode": mode, "rope_mode": rope,
                                          "score_interval_applies": rope == "none",
                                          "topk_superset_claim": rope == "none" and mode == "topk_candidates",
                                          "numerical_bound_status": "analytical real-arithmetic interval with float64 guard; not directed-rounding certification",
                                          "init_ms": engine.init_ms, "initial_cache_build_ms": build_ms,
                                          "steps": rows,
                                          "warm_append_and_attention_cpu_ms_mean": warm_mean,
                                          "amortized_cpu_ms_per_decode_step": warm_mean+(engine.init_ms+build_ms)/len(rows),
                                          "amortized_logical_mac_excluding_factorization": statistics.mean(r["costs"]["warm_step_logical_mac"] for r in rows)+extra_history_mac/len(rows),
                                          "factorization_work": "QR of two Dxd matrices + SVD of dxd core + basis products; init_ms measured, not converted to MAC"})
    return {"protocol": "ED-R002", "seed": args.seed, "torch": torch.__version__, "device": "cpu",
            "platform": platform.platform(), "threads": 1, "decode_steps": 4, "warmup": "torch QR and input construction before cases; per-case cold initialization measured",
            "timing": "per-step 2 warmups + 7 repeated attend/dense calls, median; repeated attend does not append or refit. Four persistent decode steps; initial build and each actual append timed once; Python validation and float64 selector included; diagnostic oracle excluded from RESULT",
            "input": "user_tensor" if args.tensor_input else "synthetic D64 d16 initialN256, N+4 persistent decode; independent orthogonal Uq/Uk; Wq=Uq sqrt(sigma), Wk=Uk sqrt(sigma); iid standard Gaussian X and V; current causal token included",
            "input_file_sha256": input_hash,
            "zero_norm_convention": "output/logit cosine both zero=1, one zero=0; token qk zero denominator=0",
            "accounting": "shared original Q/K/V projections and common original cache writes excluded from BOTH incremental comparisons; output projection excluded; metadata and selected gather/cat included in CPU time; torch.cat copies reported separately. Frozen weight clones and actual KV caches included in byte ledger. Runtime GQA duplicates K/V and factors per query head; no shared-GQA memory optimization. Weight QDQ is float execution, KV cache stays floating. Logical MAC is not speedup. Amortized timing includes factorization and initial cache build over four decode steps; logical MAC excludes separately reported QR/SVD setup.",
            "cases": cases}


def write_report(result, output):
    """Write full records, optionally gzip compressed, without losing fields."""
    dest = Path(output)
    data = json.dumps(result, indent=2).encode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(gzip.compress(data, mtime=0) if dest.suffix == ".gz" else data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--tensor-input", help="torch.save dict wq,wk [D,d], x [N+4,D], values [N+4,d]; even d")
    parser.add_argument("--output", default="early decision/measurements/2026-10-22-attention-comparison.json")
    args = parser.parse_args()
    if args.tensor_input and Path(args.tensor_input).resolve() == Path(args.output).resolve():
        parser.error("input and output must be different files")
    result = run(args)
    write_report(result, args.output)
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"]), "steps": sum(len(c["steps"]) for c in result["cases"])}))
