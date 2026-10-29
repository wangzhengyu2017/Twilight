# TC source distribution evidence

The real traces confirm substantial discarded source information and a one-sided TC reconstruction error. They do not establish that a different format will improve model PPL; the upcoming controlled runs test that claim.

Inputs are 54 captured traces: three thresholds, three INT/TC model trajectories, six layer/query points each. Statistics below equally weight head/operand rows; K rows contain more tokens than Q rows. All source-distribution rows diagnose four retained TC bits, including Q from a Q8K4 trajectory. Actual Q8K4 estimator logit statistics use its saved full-Q predictor.

| Source | Q | K |
|---|---:|---:|
| Negative fraction | 0.48549 | 0.49437 |
| Source zero fraction | 0.02604 | 0.01984 |
| Four-bit prefix zero fraction | 0.30755 | 0.25967 |
| Observed source-code entropy (bits) | 5.72091 | 6.66153 |
| Observed prefix entropy (bits) | 2.49900 | 2.73479 |
| Mean residual, code units | 7.47382 | 7.48095 |
| E[Var(residual given prefix)] | 18.11980 | 20.21034 |

TC reconstruction is `16 floor(x/16)`, so residual is in [0,15] and reconstruction error is nonpositive. Mean error for negative Q is -8.60228 code units versus -6.75281 for positive Q; for K, -8.27281 versus -6.98872. Zero source values have zero error. Per-prefix source histograms, distinct-code collisions, residual variance, and sign-specific DQ errors are retained in the linked CSV/JSONL artifacts. Source endpoint occupancy (~0.786%) is reported as occupancy, not evidence of quantizer clipping.

| p | Q4K4 centered logit RMSE | Q8K4 centered logit RMSE | Q4K4 mean reference-mass shortfall | Q8K4 shortfall |
|---|---:|---:|---:|---:|
| .95 | .696055 | .386717 | .009418 | .002911 |
| .90 | .697537 | .379299 | .013496 | .003394 |
| .85 | .669139 | .375183 | .019050 | .002965 |

Shortfall is max(0,p-retained original INT probability mass), computed per head before averaging. Q4K4 mean retained mass can exceed p while individual heads miss important keys; averaging shortfall preserves this distinction. The actual predictor has more exact ties for Q4K4 (about2.2–2.5% duplicate scores) than Q8K4 (about0.14–0.15%). This supports investigating coarse reconstructions and low-tail selection failures. It does not isolate their corpus-level causal contribution because the source trajectories differ.

The previous report derives the omitted cross terms and provides a concrete missed-key example, full PPL comparisons, and same-input versus trajectory distinctions: [main diagnosis](/output/project/experiments/stp-format-study/diagnosis/REPORT.md). Further causal interventions remain proposed in [supplement SPEC](/output/project/experiments/stp-format-study/diagnosis/SUPPLEMENT_SPEC.md), not executed.

Artifacts: [manifest](/output/project/experiments/stp-format-study/diagnosis-distributions/manifest.json), [histograms](/output/project/experiments/stp-format-study/diagnosis-distributions/histograms.jsonl), [source statistics](/output/project/experiments/stp-format-study/diagnosis-distributions/source_statistics.csv), [signed errors](/output/project/experiments/stp-format-study/diagnosis-distributions/signed_errors.csv), [same-input logits](/output/project/experiments/stp-format-study/diagnosis-distributions/same_input_logits.csv). The manifest binds all54 inputs and four data outputs. CPU FP32 logit recomputation is diagnostic; the original CUDA replay reports separately establish exact execution.
