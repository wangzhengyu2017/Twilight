# Four full-attention format benchmarks

All four full-corpus runs passed independent acceptance. Model: Llama-3.1-8B-Instruct (BF16 weights); task: frozen WikiText-2 raw test; protocol: C2048/S1024,282 windows,289075 targets. These are actual full-format decoded Q/K attention with all legal keys and zero TopP selection passes. They are not the earlier format-predictor experiments whose final scores used original source INT8. Arithmetic is FP32 torch Q/DQ, not a native low-bit kernel.

| Format | PPL | Total NLL | Effective targets |
|---|---:|---:|---:|
| SM8 Q8/K8 | 6.486693431358 | 540498.8236694336 | 289075 |
| MXINT8 Q8/K8 | 6.490220331854 | 540655.9548339844 | 289075 |
| MXINT8-Q8 / MXFP4-K4 | 7.062489489105 | 565083.1185913086 | 289075 |
| Native MXFP4 Q4/K4 | 8.659092745342 | 624000.1722412109 | 289075 |

Historical measured FP dense PPL is6.484759140305276 (NLL540412.6104736328); historical original INT Q/DQ dense PPL is6.486693431357931 (NLL540498.8236694336), both289075targets. Their matching archived full release gate is unavailable, so they remain historical measured anchors rather than newly accepted runs. The new SM8 measurement exactly matches original INT's NLL/PPL, consistent with its lossless full sign-magnitude reconstruction; its result was actually run and independently checked, not copied.

Full MXINT8 stays close to original INT here. Native FP4 full-attention PPL is substantially higher than the earlier FP4 predictor-only TopP results. The earlier predictor used FP4 only to choose keys and then used original full Q8/K8 for final selected scores. Therefore its near-baseline PPL was not evidence that dense native FP4 attention had the same quality. These results come from one frozen corpus/model trajectory each; no statistical-significance or native-kernel speedup claim follows.

## Captures and important-key sets

Each format has six real cached-decode companion captures: layers0,16,31 at query127(prefill) and128(decode),32heads each. All24 records passed production corrected replay and independent byte/mask/score/PV checks. Full PPL runs are trace-off. Actual dense execution indices/masks contain every legal key. Separate diagnostic-only important sets at p=.95/.90/.85 are derived from saved actual full probabilities and explicitly unused in forward; these yield576 head/threshold records per format. Codes/scales/exponents, allowed domains, positions, indices/counts/validity and outputs are preserved. Old FP/INT dense captures did not originally save important sets; do not describe later derived controls as historical collection.

## Verifier correction and preserved failure

Initial SM smoke stopped because its verifier projected a saved prefill row as GEMM M1 instead of the actual M128, changing BF16 CUDA rounding. Restoring original matrix geometry with zero padding around the authentic row gives bitwise identical Q/RoPE,K/V,o_proj and complete layer outputs for all six saved SM records. A separate Medium agent verified the correction. Original failed status/log/replay and all174 original smoke evidence files remain unchanged; successful recovery and amended acceptance are separately recorded. No completed model run was repeated and no tolerance was relaxed. The remaining three formats ran once after this correction.

An accidental temporary edit to the frozen verifier during diagnosis was restored byte-for-byte before recovery; all157 source hashes and clean Git HEAD were independently rechecked, and no original smoke artifact changed during that incident. The incident-restoration record is indexed. The original smoke/status.json intentionally remains failed; the amended gate and independent acceptance below establish subsequent success.

## Evidence and source

- Full table: /output/project/experiments/stp-dense-formats/execution-v1/full/summary/BASELINE.csv
- Exact four-row data: [RESULTS.csv](RESULTS.csv)
- Actual captures: /output/project/experiments/stp-dense-formats/execution-v1/smoke/<variant>-decode-on/trace/index.json
- Important-key listings: /output/project/experiments/stp-dense-formats/execution-v1/smoke/<variant>-decode-on/diagnostic_sets.jsonl
- Pair report: /output/project/experiments/stp-dense-formats/execution-v1/smoke/pairing/pairing_manifest.json
- Amended gate: /output/project/experiments/stp-dense-formats/execution-v1/smoke/gate.json
- Independent24-trace acceptance: [smoke-independent-acceptance.json](../review/independent-results/smoke-independent-acceptance.json)
- Independent full acceptance: [full-independent-audit.json](../review/independent-results/full-independent-audit.json)
- Correction/recovery/incident: /output/project/experiments/stp-dense-formats/review/replay-amendment-v1/
- Complete frozen-output and acceptance index: [ARTIFACT_INDEX.json](ARTIFACT_INDEX.json)

Execution source commit `502081b3d8a8d50871fceb7a813ebf7ab39bb994`, code SHA256 `1d895ebc01bda7ac04ac9d698a3b9504df8465afc3a4241f14346b7366b2e029`. Later `ce6bd8613c77f72f88c6a8b23f3cddf8e2b015af` adds six full-format TopP jobs and the unified CLI; it is not the source used for these four runs. The six TopP and ten RS jobs remain separate until their own acceptance. This report closes only the completed four-dense batch.
