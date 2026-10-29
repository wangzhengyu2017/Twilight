# Native RS attention study

The ten RS jobs are implemented and integrated. CPU checks do not establish GPU or pinned-model acceptance; all ten actual measurements and all 60 actual captures remain unexecuted. Execution requires an independent review bound to the current source and executable matrix. Full PPL additionally requires the complete RS smoke gate. Existing dense/full-format and historical predictor jobs keep their own interfaces and frozen evidence.

The [approved SPEC](../../../experiments/stp-format-study/rs-expanded-spec/SPEC.md) defines `rsint8_native_source_scale`: two signed four-bit slices, native range [-136,119], packed post-BSU H4/L4, and unchanged original per-head/token FP16 scales. Original source codes120..127 clip to119 before encoding. This changes endpoint information and adopts study scales rather than the paper's LSQ/QAT procedure. The codec is lossless over its native domain; source conversion is lossy at those positive endpoints. RS4 retains the converted signed high slice and reconstructs16H. Original quantization, floating cache, RoPE, GQA and V remain in the shared path.

| Variant | MSB | Top-p | Q/K bits | Final operands | Selectors per chunk |
|---|---|---:|---|---|---:|
| rs_dense_q8k8 | off | off | 8/8 | decoded native RS8 | 0 |
| rs_full_q8k8_topp_p095 | off | .95 | 8/8 | decoded native RS8 | 1 |
| rs_full_q8k8_topp_p090 | off | .90 | 8/8 | decoded native RS8 | 1 |
| rs_full_q8k8_topp_p085 | off | .85 | 8/8 | decoded native RS8 | 1 |
| rs_msb_q8k4_topp_p095 | on | .95 | 8/4 | original source Q8/K8 | 1 |
| rs_msb_q8k4_topp_p090 | on | .90 | 8/4 | original source Q8/K8 | 1 |
| rs_msb_q8k4_topp_p085 | on | .85 | 8/4 | original source Q8/K8 | 1 |
| rs_msb_q4k4_topp_p095 | on | .95 | 4/4 | original source Q8/K8 | 1 |
| rs_msb_q4k4_topp_p090 | on | .90 | 4/4 | original source Q8/K8 | 1 |
| rs_msb_q4k4_topp_p085 | on | .85 | 4/4 | original source Q8/K8 | 1 |

Native full Top-p reuses the actual native logits computed for selection. MSB jobs recompute selected original-source logits once. Neither applies a second selector. Their PPL difference includes different final formats and cannot be attributed solely to prefix selection. At D128 the required FP16 scale makes logical decoding budgets8.125/4.125 bits per element; the prototype also retains uint8 containers, int16 decoded integers, FP32 DQ, diagnostics and original floating caches. Reports distinguish these allocations and aliases. No native INT4 throughput is claimed.

## Unified selector: all ten cells

Run from the repository root with `/output/project/quality-venv/bin/python`. These commands produce plans without model loading or execution. Every selected smoke contains its own window, trace-off/on and replay commands; it cannot mint the ten-job cohort gate.

```bash
PY=/output/project/quality-venv/bin/python
$PY -m benchmark.attention_benchmark --encoding rsint8 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/dense
$PY -m benchmark.attention_benchmark --encoding rsint8 --top-p on --p .95 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/full095
$PY -m benchmark.attention_benchmark --encoding rsint8 --top-p on --p .90 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/full090
$PY -m benchmark.attention_benchmark --encoding rsint8 --top-p on --p .85 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/full085
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .95 --q-bits 8 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q8k4095
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .90 --q-bits 8 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q8k4090
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .85 --q-bits 8 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q8k4085
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .95 --q-bits 4 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q4k4095
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .90 --q-bits 4 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q4k4090
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .85 --q-bits 4 --k-bits 4 --profile smoke --output-root /output/project/experiments/stp-rs-study/planned/q4k4085
```

A full selected cell uses the same flags with `--profile full`, a new output root, and the matching complete `--smoke-root`, `--review-report` and `--baselines-manifest` paths. For example:

```bash
$PY -m benchmark.attention_benchmark --encoding rsint8 --msb on --top-p on --p .95 --q-bits 8 --k-bits 4 --profile full --output-root /output/project/experiments/stp-rs-study/planned/full-q8k4095 --smoke-root /output/project/experiments/stp-rs-study/smoke --review-report /output/project/experiments/stp-rs-study/review/accepted.json --baselines-manifest /output/project/experiments/stp-rs-study/preparation/baselines_manifest.json
```

## Ten-job cohort driver

The driver defaults to dry run. Once independent acceptance and GPU scheduling permit execution, `--execute` runs serial separate processes. A selected-cell plan does not replace cohort smoke evidence.

```bash
MATRIX=benchmark/attention_rs/matrix.json
HANDOFF=benchmark/attention_rs
$PY -m benchmark.attention_rs.run --profile smoke --matrix "$MATRIX" --handoff-dir "$HANDOFF" --model-path /hyperai/input/input0 --asset-dir /output/project/assets/tq-wt2-v1 --output-root /output/project/experiments/stp-rs-study/planned/cohort-smoke --events-path /output/project/experiments/stp-rs-study/events.jsonl --controls-manifest /output/project/experiments/stp-rs-study/preparation/controls_manifest.json --review-report /output/project/experiments/stp-rs-study/review/accepted.json
$PY -m benchmark.attention_rs.run --profile full --matrix "$MATRIX" --handoff-dir "$HANDOFF" --model-path /hyperai/input/input0 --asset-dir /output/project/assets/tq-wt2-v1 --output-root /output/project/experiments/stp-rs-study/planned/cohort-full --smoke-root /output/project/experiments/stp-rs-study/smoke --events-path /output/project/experiments/stp-rs-study/events.jsonl --baselines-manifest /output/project/experiments/stp-rs-study/preparation/baselines_manifest.json --review-report /output/project/experiments/stp-rs-study/review/accepted.json
```

Cohort smoke requires ten4096-token/4095-target windows, ten trace-off/on companions, six actual captures per job at layers0/16/31 and queries127/128, exact trace visibility and independent replay. Dense actual all-legal execution sets are separate from offline native-full important sets at .95/.90/.85, labelled unused in forward. Replay preserves the original chunk's index width, valid padding, repeated selected source operands and -inf invalid logits even when the captured row selects fewer keys. Sixty captures provide1920 actual head-target rows. Full scoring requires289075 targets per job,2890750 total, with complete tails.

Both pairing kinds are required. `same_input_estimator` derives original FP/source INT, native full and actual predictor references from the same captured inputs. `cross_run_trajectory` uses actual compatible runs. Historical TC `full_codes` and SM `original_intflash_codes` are explicit schema spellings for original-source final arithmetic; native, prefix and unknown origins are rejected. The twelve TC/SM groups use driver595.84. Original dense FP/INT controls use595.112.07, so those strict cross-run pairings remain unavailable. Historical dense PPL anchors remain measured but unverified because matching full gates are unavailable. No old run is repeated or relabelled.

CPU checks: `benchmark.attention_rs.validate`, `regression`, `trace_check`, `integration_check`, `contract_check` and `control_audit`. Synthetic CPU records are never counted as actual model captures. Failed/unexecuted jobs retain null NLL, target count and PPL in the ten-row summary.
