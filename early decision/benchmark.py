"""CPU numerical experiment, not a native kernel or an LLM accuracy benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import torch

from early_decision import fit_bilinear, mass_candidates, quantize_projection, topk_candidates


def mean(values):
    return sum(values) / len(values)


def synthetic_cases(seed):
    g = torch.Generator().manual_seed(seed)
    d_input, d_head = 64, 16
    def orth(rows, cols):
        return torch.linalg.qr(torch.randn(rows, cols, generator=g, dtype=torch.float64))[0]
    uq, uk, rotation = orth(d_input, d_head), orth(d_input, d_head), orth(d_head, d_head)
    xq = torch.randn(12, d_input, generator=g, dtype=torch.float64)
    xk = torch.randn(128, d_input, generator=g, dtype=torch.float64)
    spectra = {
        "fast_decay": torch.tensor([8., 6., 4., 2.] + [1e-6] * 12, dtype=torch.float64),
        "flat_spectrum": torch.full((d_head,), 3., dtype=torch.float64),
    }
    return [(name, {"W_q": (uq * sigma.sqrt()) @ rotation.T,
                    "W_k": (uk * sigma.sqrt()) @ rotation.T,
                    "x_query": xq, "x_keys": xk}) for name, sigma in spectra.items()]


def measure_case(name, payload, ranks, seed):
    required = ("W_q", "W_k", "x_query", "x_keys")
    if not all(k in payload and isinstance(payload[k], torch.Tensor) for k in required):
        raise ValueError("input needs tensor W_q, W_k, x_query [M,D], x_keys [N,D]")
    wq_base, wk_base, xq, xk = [payload[k].detach().to(dtype=torch.float64, device="cpu") for k in required]
    if xq.ndim != 2 or xk.ndim != 2 or not xq.shape[0] or not xk.shape[0]:
        raise ValueError("query and history must be nonempty matrices")
    transform = payload.get("score_transform")
    if transform is not None:
        transform = transform.to(dtype=torch.float64, device="cpu")
    if payload.get("per_token_rope", False):
        raise ValueError("per-token RoPE is not supported by this shared-transform prototype")
    n, d_input = xk.shape
    d_head = wq_base.shape[1]
    k = min(8, n)
    rows = []
    for bits in (None, 4, 8):
        wq = wq_base if bits is None else quantize_projection(wq_base, bits)
        wk = wk_base if bits is None else quantize_projection(wk_base, bits)
        # Oracle exists ONLY in this evaluation driver, never in cache.bounds/selection.
        q = xq @ wq
        if transform is not None:
            q = q @ transform
        exact = q @ (xk @ wk).T / math.sqrt(d_head)
        for rank in ranks:
            if rank > min(d_input, d_head):
                continue
            model = fit_bilinear(wq, wk, rank, score_transform=transform)
            for mode in ("norm", "tail"):
                cache = model.new_cache(mode)
                cache.append(xk)
                topk_sizes, oracle_sizes = [], []
                miss_count, coverage_violation, max_error, max_width = 0, 0, 0., 0.
                mass_stats = {p: {"sizes": [], "mass": [], "bound": [], "oracle_sizes": []}
                              for p in (.9, .99)}
                for i, query in enumerate(xq):
                    bounds = cache.bounds(query)
                    target = exact[i]
                    violation = (target < bounds.lower - 1e-9) | (target > bounds.upper + 1e-9)
                    coverage_violation += int(violation.sum())
                    max_error = max(max_error, float((target - bounds.estimate).abs().max()))
                    max_width = max(max_width, float((bounds.upper - bounds.lower).max()))
                    decision = topk_candidates(bounds, k)
                    oracle_topk = torch.topk(target, k).indices
                    miss_count += int((~decision.mask[oracle_topk]).sum())
                    topk_sizes.append(int(decision.mask.sum()))
                    oracle_sizes.append(k)
                    probabilities = target.softmax(0)
                    sorted_p = torch.sort(probabilities, descending=True).values
                    cumulative = sorted_p.cumsum(0)
                    for p, stats in mass_stats.items():
                        result = mass_candidates(bounds, p)
                        actual = float(probabilities[result.mask].sum())
                        if actual + 1e-9 < p or actual + 1e-9 < result.mass_lower_bound:
                            raise AssertionError("mass bound failed against oracle")
                        stats["sizes"].append(int(result.mask.sum()))
                        stats["mass"].append(actual)
                        stats["bound"].append(result.mass_lower_bound)
                        stats["oracle_sizes"].append(int(torch.searchsorted(cumulative, p)) + 1)
                tail_dim = model.left_tail.shape[1] if mode == "tail" else 0
                full_rank_reusable = rank == model.singular_values.numel()
                rows.append({
                    "case": name, "seed": seed, "weight_bits": bits,
                    "bound_mode": mode, "rank": rank,
                    "shape": {"D": d_input, "d": d_head, "N": n, "queries": xq.shape[0]},
                    "reference": "linear FP64 over frozen QDQ weights; no per-token RoPE or activation QDQ",
                    "residual_sigma": model.residual_sigma,
                    "singular_values": model.singular_values.tolist(),
                    "interval_violations": coverage_violation,
                    "max_estimate_abs_error": max_error, "max_interval_width": max_width,
                    "topk": {"k": k, "mean_B0_size": mean(topk_sizes),
                             "min_B0_size": min(topk_sizes), "max_B0_size": max(topk_sizes),
                             "missed_oracle_tokens": miss_count},
                    "mass": {str(p): {"mean_B0_size": mean(stats["sizes"]),
                                     "mean_oracle_top_p_size": mean(stats["oracle_sizes"]),
                                     "min_actual_mass": min(stats["mass"]),
                                     "min_mass_bound": min(stats["bound"])}
                             for p, stats in mass_stats.items()},
                    "storage": {"input_cache_actual_fp64_bytes": cache.storage_bytes,
                                "full_K_fp64_bytes_for_comparison": n * d_head * 8,
                                "factors_and_tail_bases_actual_fp64_bytes": model.storage_bytes},
                    "logical_MAC_counts": {
                        "offline_QR_SVD": "excluded from online counts; must amortize separately",
                        "extra_history_feature_build": n * d_input * rank,
                        "extra_history_tail_norm_projection": n * d_input * tail_dim,
                        "query_feature_projection": d_input * rank,
                        "query_tail_norm_projection": d_input * tail_dim,
                        "B0_score_scan_per_query": n * rank,
                        "baseline_full_QK_per_query": n * d_head,
                        "full_QK_on_topk_B0_unreused": mean(topk_sizes) * d_head,
                        "full_QK_on_topk_B0_if_full_rank_scores_reused": (
                            0. if full_rank_reusable else mean(topk_sizes) * d_head),
                        "notes": "no selection/exp/sort/norm/metadata transactions or bit-cost weighting; no speedup claim",
                    },
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="torch.save dict with W_q,W_k,x_query,x_keys; row-layout weights")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", default="0,2,4,8,16")
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ranks = [int(s) for s in args.ranks.split(",")]
    if args.input is not None and args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if args.input is None:
        cases = synthetic_cases(args.seed)
    else:
        payload = torch.load(args.input, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("input must contain a tensor dictionary")
        cases = [("provided_tensors", payload)]
    rows = []
    for name, payload in cases:
        rows.extend(measure_case(name, payload, ranks, args.seed if args.input is None else None))
    if not rows:
        raise ValueError("no valid ranks produced measurements")
    report = {"experiment": "early-decision-bilinear-v1", "date": "2026-10-22",
              "kind": "synthetic numerical validation" if args.input is None else "provided tensor validation",
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "threads": 1, "dtype": "float64"},
              "scope": "no LLM/PPL, no native INT kernel, no per-token RoPE, no latency or energy measurement",
              "interval_violations": sum(row["interval_violations"] for row in rows),
              "topk_missed_tokens": sum(row["topk"]["missed_oracle_tokens"] for row in rows),
              "configurations": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("configurations", "interval_violations", "topk_missed_tokens")}))


if __name__ == "__main__":
    main()
