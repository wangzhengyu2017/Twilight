"""Read-only distribution diagnostics of existing real TC source traces.

No model, new estimator, or causal counterfactual is executed. CPU FP32 logits
are diagnostic values; exact CUDA replay is supplied by the original reports.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    roots = {
        "095": Path("/output/project/experiments/stp-slice/20261028-v2-smoke"),
        "090": Path("/output/project/experiments/stp-slice-thresholds/p090-smoke"),
        "085": Path("/output/project/experiments/stp-slice-thresholds/p085-smoke"),
    }
    code_rows, sign_rows, logit_rows, inputs = [], [], [], []
    hist = (out / "histograms.jsonl").open("w")
    for tag, root in roots.items():
        p = int(tag) / 100
        for source in ("int", "msb_q4k4", "msb_q8k4"):
            trace_root = root / f"{source}_topp_p{tag}-decode-on" / "trace"
            files = sorted(trace_root.glob("*/actual.pt"))
            if len(files) != 6:
                raise ValueError(f"Expected six real traces: {trace_root}: {len(files)}")
            for path in files:
                inputs.append({"path": str(path), "sha256": sha(path)})
                a = torch.load(path, map_location="cpu", weights_only=True)
                name = path.parent.name
                layer, query, stage = name.split("_")
                base = dict(p=p, source=source, trace=name,
                            layer=int(layer[5:]), query=int(query[5:]), stage=stage)
                for operand in ("q", "k"):
                    x = a[f"{operand}_codes"].reshape(32, -1, 128).to(torch.int32)
                    scales = a[f"{operand}_scales"].float().reshape(32, -1, 1)
                    for head in range(32):
                        xx = x[head].flatten()
                        hh = torch.div(xx, 16, rounding_mode="floor")
                        residual = (xx - 16 * hh).double()
                        code_hist = torch.bincount((xx + 127).long(), minlength=255)
                        prefix_hist = torch.bincount((hh + 8).long(), minlength=16)
                        nonzero = code_hist > 0
                        probs = code_hist[nonzero].double() / xx.numel()
                        hp = prefix_hist[prefix_hist > 0].double() / xx.numel()
                        per_prefix, weighted_variance, collisions = [], 0.0, 0
                        for prefix in range(-8, 8):
                            sel = hh == prefix
                            count = int(sel.sum())
                            unique_codes = int(xx[sel].unique().numel())
                            variance = float(residual[sel].var(unbiased=False)) if count else None
                            if count:
                                weighted_variance += count * variance / xx.numel()
                                collisions += max(0, unique_codes - 1)
                            per_prefix.append(dict(prefix=prefix, count=count,
                                                   unique_source_codes=unique_codes,
                                                   residual_variance=variance))
                        row = dict(**base, head=head, operand=operand)
                        hist.write(json.dumps(dict(**row, code_min=-127,
                            source_histogram=code_hist.tolist(), prefix_min=-8,
                            prefix_histogram=prefix_hist.tolist(), conditional=per_prefix)) + "\n")
                        code_rows.append(dict(**row, count=xx.numel(),
                            negative_fraction=float((xx < 0).double().mean()),
                            positive_fraction=float((xx > 0).double().mean()),
                            source_zero_fraction=float((xx == 0).double().mean()),
                            prefix_zero_fraction=float((hh == 0).double().mean()),
                            endpoint_fraction=float((xx.abs() == 127).double().mean()),
                            endpoint_note="endpoint occupancy, not proof of clipping",
                            unique_codes=int(nonzero.sum()), unique_prefixes=int((prefix_hist > 0).sum()),
                            excess_unique_codes_per_prefix=collisions,
                            conditional_residual_variance=weighted_variance,
                            residual_mean=float(residual.mean()),
                            source_entropy_bits=float(-(probs * probs.log2()).sum()),
                            prefix_entropy_bits=float(-(hp * hp.log2()).sum())))
                        error = (16 * hh - xx).double()
                        real_error = error * scales[head].expand_as(x[head]).flatten().double()
                        for sign, selection in (("negative", xx < 0), ("zero", xx == 0), ("positive", xx > 0)):
                            n = int(selection.sum())
                            sr = dict(**row, sign=sign, count=n)
                            for units, e in (("code", error), ("dq", real_error)):
                                for metric, values in (("bias", e), ("mae", e.abs()), ("mse", e.square())):
                                    sr[f"{units}_{metric}"] = float(values[selection].mean()) if n else None
                            sign_rows.append(sr)
                # Same-input original full INT reference; CPU diagnostic, not CUDA replay.
                q, k = a["q_dq"].float(), a["k_dq"].float()
                ref = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).reshape(32, -1)
                pred = a["estimator_logits"].float().reshape(32, -1)
                allowed = a["allowed_mask"].expand_as(a["estimator_logits"]).reshape(32, -1)
                selected = a["selected_mask"].reshape(32, -1)
                for head in range(32):
                    legal = allowed[head]
                    r, z, s = ref[head][legal], pred[head][legal], selected[head][legal]
                    prob = r.softmax(-1)
                    order = torch.argsort(r, descending=True, stable=True)
                    cumulative = prob[order].cumsum(-1)
                    crossing = torch.searchsorted(cumulative, torch.tensor(p), right=False).clamp(max=r.numel()-1)
                    rs = torch.zeros_like(s).scatter(0, order[:int(crossing)+1], True)
                    error = (z-r).double()
                    rank_z = torch.argsort(torch.argsort(z, stable=True), stable=True).double()
                    rank_r = torch.argsort(torch.argsort(r, stable=True), stable=True).double()
                    rank_z -= rank_z.mean(); rank_r -= rank_r.mean()
                    retained = float(prob[s].sum())
                    logit_rows.append(dict(**base, head=head, legal_keys=r.numel(),
                        logit_bias=float(error.mean()), logit_mae=float(error.abs().mean()),
                        logit_mse=float(error.square().mean()),
                        centered_logit_rmse=float((error-error.mean()).square().mean().sqrt()),
                        stable_rank_correlation=float((rank_z*rank_r).sum()/(rank_z.norm()*rank_r.norm()).clamp_min(1e-30)),
                        predictor_tie_fraction=1-z.unique().numel()/z.numel(),
                        reference_tie_fraction=1-r.unique().numel()/r.numel(),
                        selected_count=int(s.sum()), reference_selected_count=int(rs.sum()),
                        retained_reference_mass=retained, mass_shortfall=max(0, p-retained),
                        recall=float((s & rs).sum()/rs.sum()), precision=float((s & rs).sum()/s.sum()),
                        jaccard=float((s & rs).sum()/(s | rs).sum())))
    hist.close()
    write_csv(out / "source_statistics.csv", code_rows)
    write_csv(out / "signed_errors.csv", sign_rows)
    write_csv(out / "same_input_logits.csv", logit_rows)
    manifest = dict(status="verified", scope="read-only CPU diagnostics; no new estimator/model run",
        script_sha256=sha(Path(__file__)), inputs=inputs, traces=len(inputs),
        source_head_operand_rows=len(code_rows), sign_rows=len(sign_rows), logit_rows=len(logit_rows),
        collision_definition="sum over prefixes max(distinct observed source codes-1,0)",
        variance_definition="E_prefix[Var(source_code-16*prefix | prefix)], population variance",
        numerical_scope="CPU FP32 diagnostic full logits; original independent CUDA replay remains separate",
        outputs=[dict(path=str(f), sha256=sha(f)) for f in sorted(out.iterdir()) if f.is_file()])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("inputs", "outputs")}))


if __name__ == "__main__":
    main()
