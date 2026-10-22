"""Validate and summarize measured records; no new attention execution."""
import argparse
import gzip
import json
from pathlib import Path
import statistics


def load_report(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def summarize(report):
    cases = report["cases"]
    full_keep, checked, bounds_checked = 0, 0, 0
    rows = []
    for case in cases:
        for step in case["steps"]:
            n, b = step["n"], step["selected"]
            indices = step["selected_indices"]
            assert 0 < b <= n and len(set(indices)) == b == len(indices)
            assert all(0 <= i < n for i in indices)
            assert len(step["token_cosine_qk"]) == n
            assert -1 - 1e-12 <= step["output_cosine"] <= 1 + 1e-12
            assert 0 <= step["retained_true_softmax_mass"] <= 1 + 1e-6
            assert abs(step["probability_tv"] - (1 - step["retained_true_softmax_mass"])) < 5e-7
            if case["budget"] >= n:
                assert step["ref_result_allclose"]
                full_keep += 1
            if case["rope_mode"] == "none" and case["mode"] == "topk_candidates":
                assert step["oracle_logits_requested_k"]["recall"] == 1
                bounds_checked += 1
            checked += 1
        if case["weight_bits"] != "FP" or case["rank"] != 4 or case["budget"] != 32:
            continue
        steps = case["steps"]
        avg = lambda key: statistics.mean(s[key] for s in steps)
        nested = lambda key, field: statistics.mean(s[key][field] for s in steps)
        dense_mac = statistics.mean(s["costs"]["dense_qk_av_logical_mac"] for s in steps)
        row = {"spectrum": case["source"], "mode": case["mode"], "rope": case["rope_mode"],
               "selected_mean": avg("selected"), "output_cosine": avg("output_cosine"),
               "relative_l2": avg("output_relative_l2"), "retained_mass": avg("retained_true_softmax_mass"),
               "warm_mac_ratio": statistics.mean(s["costs"]["warm_step_logical_mac"] for s in steps)/dense_mac,
               "amortized_mac_ratio_excluding_factorization": case["amortized_logical_mac_excluding_factorization"]/dense_mac,
               "warm_cpu_ms": avg("cpu_warm_step_ms"), "dense_cpu_ms": avg("cpu_dense_qk_av_ms"),
               "amortized_cpu_ms": case["amortized_cpu_ms_per_decode_step"]}
        for oracle in ("oracle_logits_requested_k", "oracle_cosine_qk_requested_k"):
            row[oracle] = {metric: nested(oracle, metric) for metric in ("precision", "recall", "f1", "jaccard")}
        rows.append(row)
    return {"protocol": "ED-R002", "configurations": len(cases), "steps_checked": checked,
            "full_keep_steps_allclose": full_keep, "no_rope_bound_steps_recall_one": bounds_checked,
            "table_scope": "FP, rank4, requested-k32, means across 4 persistent decode steps",
            "source_kind": "synthetic CPU component measurements, no LLM checkpoint",
            "rows": rows}


def markdown(summary):
    lines = ["# ED-R002 实测摘要", "", "由完整记录自动生成；固定 FP、rank=4、requested-k=32，四个 decode step 的均值。逻辑 MAC 比值不是实测加速。", "",
             "| 谱／模式／RoPE | 候选数 | 输出cos | 相对L2 | 真实mass | logits召回 | token-cos召回 | warm MAC比 | 含4步初始化摊销MAC比* | CPU ms / dense ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["rows"]:
        lines.append(f"| {r['spectrum']}/{r['mode']}/{r['rope']} | {r['selected_mean']:.2f} | {r['output_cosine']:.4f} | {r['relative_l2']:.4f} | {r['retained_mass']:.4f} | {r['oracle_logits_requested_k']['recall']:.4f} | {r['oracle_cosine_qk_requested_k']['recall']:.4f} | {r['warm_mac_ratio']:.3f} | {r['amortized_mac_ratio_excluding_factorization']:.3f} | {r['warm_cpu_ms']:.4f}/{r['dense_cpu_ms']:.4f} |")
    lines += ["", "* 摊销 MAC 含初始 metadata 构建；QR/SVD 用实际初始化时间另列，没有虚构其等价 MAC。实际 CPU 的摊销时间字段包含这些初始化时间。", "",
              f"共检查 {summary['configurations']} 配置、{summary['steps_checked']} steps；{summary['full_keep_steps_allclose']} 个全保留 step 与 dense allclose；{summary['no_rope_bound_steps_recall_one']} 个 no-RoPE 保守 step 的 requested-k recall=1。",
              "", "本摘要不支持完整模型质量或 GPU 加速结论；逐 token RoPE 使用 pre-RoPE 筛选启发式，线性区间保证不适用。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--json-output", required=True)
    p.add_argument("--markdown-output", required=True)
    args = p.parse_args()
    if len({Path(v).resolve() for v in (args.input, args.json_output, args.markdown_output)}) != 3:
        p.error("input and outputs must be distinct")
    result = summarize(load_report(args.input))
    Path(args.json_output).write_bytes(json.dumps(result, indent=2).encode("utf-8"))
    Path(args.markdown_output).write_bytes(markdown(result).encode("utf-8"))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
