# Full-format TopP benchmarks and unified selector

The selector resolves one numeric format and attention profile into its exact existing configuration, source/config hashes and subprocess commands. It prints a plan by default. `--execute` requires a current independent implementation review; full execution also requires the matching complete smoke gate. Existing entry points remain available.

This package implements six new jobs: full SM8 Q8K8 and full MXINT8 Q8K8, each at p=.95/.90/.85. They follow the four dense-format jobs. The earlier eighteen format-predictor measurements remain frozen. RS is a separate study and is not dispatched by this selector yet.

Run the examples from the source checkout being inspected, using the pinned environment. The five selector/batch examples below omit `--execute`, so they print plans without loading the model, allocating CUDA tensors, or creating output directories. Use new output paths. Execution must use the same frozen checkout as its accepted review and smoke gate; do not edit an executing checkout.

```bash
cd /output/project/Twilight-STP_test_Format
```

## Select one benchmark

Full MXINT8 Q8K8 TopP at p=.90, with one-config smoke:

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_benchmark \
  --encoding mxint8 --msb off --top-p on --p .90 \
  --q-bits 8 --k-bits 8 --final-operands format \
  --profile smoke \
  --output-root /output/project/experiments/stp-dense-formats/full-topp-v1/selector-mxint8-p090-smoke
```

The printed `execution_commands` contain the numerical validator and exactly this configuration's windows, decode-off, decode-on, exact comparison and replay commands. Choosing one format never expands into other format measurements. A one-config smoke does not create the cohort gate required for full execution; `gate_prerequisites` states that requirement.

Dense native FP4 is a distinct numeric format, with zero selector passes:

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_benchmark \
  --encoding mxfp4 --msb off --top-p off \
  --q-bits 4 --k-bits 4 \
  --profile smoke \
  --output-root /output/project/experiments/stp-dense-formats/full-topp-v1/selector-native-fp4-dense-smoke
```

After the complete six-job smoke has passed, plan one full MXINT8 p=.90 cell using its gate:

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_benchmark \
  --encoding mxint8 --msb off --top-p on --p .90 \
  --final-operands format --profile full \
  --smoke-root /output/project/experiments/stp-dense-formats/full-topp-v1/smoke \
  --review-report /output/project/experiments/stp-dense-formats/review/full-topp-implementation/REVIEW.json \
  --output-root /output/project/experiments/stp-dense-formats/full-topp-v1/selector-mxint8-p090-full
```

This resolves to one `benchmark.ppl` full job with the selected JSON config and `--full-format-gate-report`. Appending `--execute` runs that single cell only after review, source, model, assets, configuration and gate checks pass. Do not execute a cell already completed by the authorized batch.

## Run the six-job batch

The dedicated batch runner is separate from the one-config selector. Its smoke plan contains forty serial commands: numerical validation, six preserved small controls and their comparison, three scoring profiles plus comparison/replay for each of six variants, pairing, and the release gate.

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_full_format.run \
  --profile smoke \
  --matrix benchmark/attention_full_format/matrix.json \
  --model-path /hyperai/input/input0 \
  --asset-dir /output/project/assets/tq-wt2-v1 \
  --output-root /output/project/experiments/stp-dense-formats/full-topp-v1/smoke \
  --dense-smoke-root /output/project/experiments/stp-dense-formats/execution-v1/smoke \
  --review-report /output/project/experiments/stp-dense-formats/review/full-topp-implementation/REVIEW.json \
  --events-path /output/project/experiments/stp-dense-formats/events.jsonl
```

Once that smoke gate and independent acceptance pass, plan the six full PPL jobs:

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_full_format.run \
  --profile full \
  --matrix benchmark/attention_full_format/matrix.json \
  --model-path /hyperai/input/input0 \
  --asset-dir /output/project/assets/tq-wt2-v1 \
  --smoke-root /output/project/experiments/stp-dense-formats/full-topp-v1/smoke \
  --output-root /output/project/experiments/stp-dense-formats/full-topp-v1/full \
  --review-report /output/project/experiments/stp-dense-formats/review/full-topp-implementation/REVIEW.json \
  --events-path /output/project/experiments/stp-dense-formats/events.jsonl
```

Append `--execute` only to the intended approved plan. The batch stops on failure and preserves logs; it does not restart completed work. Full runs use 282 windows, 289075 effective targets, C=2048 and S=1024. The frozen 289076-token WikiText assets and model are reused; no dataset regeneration occurs.

Optional inputs are explicit. `--dense-smoke-root` supplies authentic dense traces for cross-run pairing. A `--controls-manifest` for this package maps `095`, `090`, `085` to authentic source-INT TopP decode-on run directories. `--baselines-manifest` maps `fp` and `source_int` to the frozen full dense controls. `--dense-summary-manifest` supplies a hash-bound dense `summary_manifest.json`, enabling `summary/NON_RS_BASELINE.csv` and `non_rs_manifest.json` for the ten non-RS jobs plus old controls. These manifests do not authorize rerunning historical measurements.

## Operand and selection profiles

| Requested profile | Predictor/full scores | Final operands | Selector passes |
|---|---|---|---:|
| SM8/MXINT8 dense, MSB off, TopP off | Actual full stored-byte decode over legal keys | Decoded full format | 0 |
| SM8/MXINT8 Q8K8, MSB off, TopP on | Actual full stored-byte decode, one stable TopP | The same retained format logits, renormalized | 1 |
| Historical INT8/SM8/MXINT8 MSB TopP | Reviewed Q4K4 or Q8K4 prefix predictor | Original source full INT8 inside its predicted mask | 1 |
| Native/mixed FP4 dense | Native E2M1 full code, independent Q/K metadata | Decoded full native format | 0 |
| Historical native/mixed FP4 predictor TopP | Native low-bit predictor | Original source full INT8; requires explicit `--final-operands source-int8` | 1 |

`--final-operands auto` resolves to `source-int8` for MSB and `format` otherwise. The resolved plan and invocation record show the actual `final_operand_profile`, `final_operand_origin`, widths and selection count. The new six configurations record `decoded_full_format`; they never substitute original source-INT scores for MXINT8 final scores. SM8 has exact same-arithmetic source-INT identity, but all three requested SM8 runs are still measured.

The model/projections and original floating K/V cache use BF16. Post-RoPE source Q/K use the existing FP16 per-token source quantizer and scales. SM stores sign/magnitude bytes; MXINT8 stores signed bytes and separate block32 E8M0 metadata. Reconstruction, QK scaling, softmax and PV use FP32; original V and the established output cast/projection path are preserved. Native BF16-to-FP16 underflow-to-zero behavior is retained. This is torch format-quality evaluation, with no native low-bit kernel or speedup claim.

## Important-key evidence and results

Every new TopP variant requires the six authentic targets from `benchmark/attention_format/trace_targets.json`: layers 0/16/31, prefill query 127 and decode query 128. Thirty-six real traces are required for the six-job gate. Trace-on/off logits and NLL must be identical.

Each decode-on directory contains `selection_sets.jsonl` and a hash-bound `trace/index.json`. `actual.pt` records source codes/scales, actual format bytes/exponents/flags and decoded Q/K, allowed mask, full-format logits/probabilities, selected IDs/counts/padded validity and mask, retained logits, renormalized probabilities, original V, per-head PV and recorder-boundary outputs. These selected sets have `unused_in_forward=false`: they are the actual forward sets. Dense traces instead expose separate diagnostic p=.95/.90/.85 sets tagged `unused_in_forward=true`, while their actual execution mask includes all legal keys.

`derived.pt` contains labeled offline controls. Pairing keeps same-input source-INT TopP and format-dense comparisons separate from authentic cross-run trajectories. Jaccard/intersection, precision/recall, retained reference mass, argmax survival, counts/fractions, centered-logit errors and PV errors use the matching causal key domain and keep p separate. An unavailable real reference is labeled unavailable rather than fabricated.

The gate requires CPU/CUDA numerical checks, exact preserved controls, trace invisibility, byte reconstruction, exact stable selected support, retained-score equality, output replay, pairing and matching source/model/asset/config identities. Model-row replay preserves the original linear M dimension when reconstructing a captured prefill row. Missing evidence fails the gate.

Runs write invocation/provenance, local command/window logs, status, coverage, selection statistics and costs. `summary/BASELINE.csv`, `BASELINE.md` and `summary_manifest.json` retain null NLL/PPL for failed or unexecuted measurements. Runner completion is not independent acceptance. Only compact start/failure/summary events enter the existing registered monitoring stream; per-window progress stays local, and independent auditors own verified-completion events.

## Rejected or historical-only choices

- Operational p is restricted to .95/.90/.85. Arbitrary thresholds, including p=1, need their own approved configuration and gate; p=1 is never used as a dense alias.
- `--p` with TopP off, MSB without TopP, FP MSB, native/mixed FP4 MSB, and incompatible widths fail before model load.
- Default MSB widths are recorded Q8K4; Q4K4 is supported explicitly. Native FP4 Q4K4 is never called an eight-bit MSB prefix.
- Native/mixed FP4 full-format TopP is not part of the implemented six-job matrix. Explicit source-INT final operands resolve its historical predictor configurations only.
- `rsint8` fails until the separate RS implementation and independent gate are integrated.
- Existing FP/INT and eighteen format-predictor configurations resolve without changing their semantics. Under the current no-rerun authorization, the selector prints their plans and rejects `--execute`; their original CLI entry points remain unchanged.
- Output directories must be new. Missing/stale implementation reviews or mismatched full gates fail; the selector cannot mint or bypass a cohort gate.

## CPU checks

These use no real model forward. Run from the repository root with the pinned interpreter:

```bash
/output/project/quality-venv/bin/python -m benchmark.attention_full_format.validate \
  --matrix benchmark/attention_full_format/matrix.json \
  --output /tmp/stp-full-topp-cpu-validation.json --device cpu
/output/project/quality-venv/bin/python -m benchmark.attention_full_format.tests.test_selector
/output/project/quality-venv/bin/python -m benchmark.attention_full_format.tests.test_trace_capture
```

CPU-only numerical validation reports `cpu_only` with CPU checks passed and CUDA not executed. The selector checks cover 42 currently supported study profiles, real single-config argv/config dispatch, invalid combinations, historical semantics, and gate refusal. Recorder checks are synthetic CPU integration evidence and do not count as real study traces. See [HANDOFF.md](HANDOFF.md) for implementation and validation status.
