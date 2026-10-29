# Full-format TopP and selector handoff

Implementation is complete for the six full SM8/MXINT8 Q8K8 TopP jobs and the unified selector. Independent Medium CPU review passed the six numerical variants, 42 selector profiles, synthetic recorder integration, shared dense checks and historical format behavior. No six-job real-model smoke, CUDA numerical validation, full PPL measurement or real trace collection has been executed by this implementation task. GPU work remains with root orchestration after the four dense jobs.

[README.md](README.md) contains executable plan-only single-config smoke/full and six-job batch examples. The selector resolves one benchmark. Its standalone smoke does not mint the cohort gate; full execution requires the matching complete accepted smoke gate. Current p values are .95/.90/.85. Unsupported numeric/native-FP4/RS combinations and historical reruns fail explicitly.

## Source scope

New modules:

- `benchmark/attention_benchmark.py`: unified resolution, serialization and single-config dispatch.
- `benchmark/attention_full_format/`: six exact configs and matrix, independent formal checks/oracle, numerical validation, trace replay, pairing, immutable gate, six-job runner and null-preserving/unified summaries.
- `twilight/pyimpl/quant_attention/full_format_backend.py`: actual full stored-byte scores, one selector and reuse of the retained format logits.
- `twilight/pyimpl/full_format_trace.py`: real forward selected sets and separate same-input controls.

Shared changes are limited to algorithm config loading, model/state dispatch, scorer manifests/gates/recorder selection, pluggable serial runner and full-cell filtering, reusable cache tests, and original-M model-row replay. Old codec/backend definitions and existing named configs are preserved.

Explicit modified paths are `benchmark/attention_baseline/common.py`, `benchmark/ppl.py`, `twilight/pyimpl/attention.py`, `twilight/pyimpl/quant_attention/forward.py`, and `benchmark/attention_dense_format/{run,validate,model_replay}.py`. The new package includes `tests/test_selector.py` and `tests/test_trace_capture.py`. No RS code is merged here.

## Validation evidence

Implementer CPU evidence under `/output/project/experiments/stp-dense-formats/preparation/`:

- `full-topp-cpu-validation-v2.json`: all six numerical/cache variants passed on CPU; exact SM/source-INT identity, actual bytes, one selector, retained score reuse, selected support/IDs, renormalization/PV and mask rejection.
- `selector-cpu-validation.json`: 42 real profiles, single-config dispatch, invalid choices, plan-only no-model/no-CUDA behavior and missing-gate refusal passed.
- `full-topp-trace-cpu-validation.json`: exact recorder invisibility and saved code/score/mask/PV boundaries passed on synthetic CPU tensors; these are not authentic study captures.
- `full-topp-smoke-plan.json`: forty-command serial batch plan; no model execution.

Fresh independent Medium evidence lives under `/output/project/experiments/stp-dense-formats/review/full-topp-implementation/`: `independent-cpu-validation.json`, `independent-dense-regression-cpu.json`, `independent-historical-format-cpu.json`, `REVIEW.md`, `REVIEW.json` and `reviewed-source.json`. Use the pinned `/output/project/quality-venv/bin/python`; system Python has a different Transformers cache API. Final review/receipt must bind the committed source before execution.

The shared model-row replay correction retains original prefill M=128 and authentic row index for all frozen-weight linear operations; decode retains M=1. Its evidence and incident/restoration audit are separately retained at `/output/project/experiments/stp-dense-formats/review/replay-amendment-v1/`. The active four-dense execution checkout remains separate and immutable. This main-checkout helper change is for the future six-job source; it does not modify or overwrite the active dense evidence.

## Root release steps

1. Review the exact whitelist; commit these source/config/test files and the two explicit package Markdown files only. Preserve unrelated/historical paths. No commit has been made by this task.
2. Create a separate detached six-job execution checkout from that commit. Verify identical source/config identities there and refresh the independent implementation receipt against its HEAD/code hash.
3. After dense GPU work completes, run the six-job smoke plan from that frozen checkout. It includes the necessary CUDA checks, six small preserved controls, 36 authentic traces, exact trace invisibility, independent replay and pairing before gate construction.
4. Independently accept the passed gate and run the six full jobs serially, each 282 windows/289075 targets. Do not edit the execution checkout or repeat completed jobs for monitoring/publication.
5. Keep same-input and cross-run metrics separate; import accepted dense results with `--dense-summary-manifest` for the ten-job non-RS table. Failed/missing metrics stay null. Root independently audits measured outputs and handles final figures/index/publication.

RS remains a separate staged integration and ten-job gate. The selector intentionally rejects it until that work has its own independent acceptance. The old eighteen results/source and the four-dense execution snapshot are not reinterpreted by this handoff.
