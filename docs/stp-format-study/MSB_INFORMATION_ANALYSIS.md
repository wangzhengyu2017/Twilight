# TC MSB Q4×K4 Top-p: mathematical and observational diagnosis

Status: executed CPU analysis of existing evidence, 2026-10-28. New model runs, new GPU work, and proposed estimator remedies are not part of this report. The p=0.85 and p=0.90 full-window sweeps are now complete and verified; the p=0.95 completed full-window baseline and all three thresholds' completed smoke traces are included. The pre-update report is preserved in `history/REPORT-before-p085-full-completion.md`.

## Finding and evidence boundary

The evidence supports a specific failure chain: taking the high signed two's-complement nibble introduces a one-sided operand error; omitting the resulting mixed terms changes *relative* logits; Top-p acts on the distorted probabilities and sometimes removes keys carrying substantial full-INT attention mass; selected full-code recomputation cannot restore a discarded key. The Q8×K4 control substantially reduces this error and the observed quality loss. This is stronger than a generic statement that four bits lose information, but it does not establish what fraction of corpus NLL degradation is caused by each algebraic term or prove that a proposed correction would fix PPL.

The six actual traces per run cover layers 0, 16, 31, prefill query 127 and decode query 128, all 32 heads: 192 head identities. Three thresholds and three integer source trajectories give 1,728 source/head observations. Evaluating the already-established full INT, Q4×K4 and Q8×K4 estimator definitions on those same operands gives 5,184 diagnostic metric rows. These are correlated traces from a small smoke sequence, not independent samples from the full corpus.

The source is frozen. No existing experiment artifact, model source or Git state was changed. All new files reside here. The script uses CPU PyTorch with two intra-op and two inter-op threads, loads saved operands only, and does not instantiate a model. `manifest.json`, `verification.json` and the CSVs record the scope. A first attempted diagnostic program containing extra midpoint/cross-term-restoration predictions failed before saving output; those proposed estimators were removed before the successful run in response to the instruction requiring approval for supplemental experiments. No remedy results are reported.

## Completed end-to-end evidence

Existing full-window results, each with 289,075 scored targets:

| p | Full INT PPL | Q4×K4 PPL | Q8×K4 PPL | Full INT selected fraction | Q4×K4 fraction | Q8×K4 fraction |
|---|---:|---:|---:|---:|---:|---:|
| .95 | 6.485039 | 6.836825 | 6.500290 | .141888 | .230227 | .151824 |
| .90 | 6.532782 | 7.317663 | 6.563772 | .085027 | .153913 | .092222 |
| .85 | 6.614929 | 7.921542 | 6.667327 | 0.059787 | 0.116829 | 0.065664 |

Sources: `../../stp-slice/20261028-v2-full/comparison.csv`, `../../stp-slice-thresholds/p090-full/comparison.csv`, and `../../stp-slice-thresholds/p085-full/comparison.csv`. The threshold study summary has execution status `verified` and no failures; each full p=.85 variant scores 289,075 targets. FP Top-p at p=.85 has PPL 6.613165. Its integer control is close (6.614929), while Q4×K4 reaches 7.921542 and Q8×K4 reaches 6.667327. The Q4×K4 PPL gap against the same-threshold INT control grows from 0.351786 at p=.95 to 0.784881 at p=.90 and 1.306613 at p=.85. This strengthens the observed quality-risk trend as the threshold decreases, without isolating its individual mathematical causes. All methods use one Top-p pass. Integer final scores use the same full source INT8 codes and dequantization rules. In TC methods the full scores are recomputed only for selected keys. Accordingly the corpus gap is not simply the direct use of four-bit final QK or PV values. Q4×K4 selects more keys overall and still has worse PPL; selected count alone does not measure selection quality.

## Exact mathematics and the role of cancellation

Write source integer vectors as

\[
Q=aH_Q+L_Q,\quad K=bH_K+L_K,
\]

where `H=floor(code/a)`, `L=code-aH`. For four-bit prefixes, a=b=16, H lies in [-8,7] for the INT8 range and L in [0,15]. The original quantizer's actual codes are bounded by ±127. The estimator is

\[
\hat z_j = c_j ab\langle H_Q,H_{K_j}\rangle,
\quad c_j=s_Qs_{K_j}/\sqrt{128}.
\]

The full-INT score satisfies exactly

\[
z_j=\hat z_j+c_j\left[a\langle H_Q,L_{K_j}\rangle+b\langle L_Q,H_{K_j}\rangle+\langle L_Q,L_{K_j}\rangle\right].
\]

The analysis checks this decomposition numerically on every processed head; maximum absolute residual is asserted below 1e-4. The production prefix dot is representable exactly in FP32 at these dimensions. Hence this is not evidence of integer-overflow or prefix-MAC rounding failure.

Equivalently, truncated vectors are Q'=Q-LQ and K'=K-LK, so

\[
\hat z_j-z_j=c_j[-\langle L_Q,K_j\rangle-\langle Q,L_{K_j}\rangle+\langle L_Q,L_{K_j}\rangle].
\]

Residuals are nonnegative even for negative codes: -1 truncates to -16 while +1 truncates to 0. This is floor truncation, not symmetric four-bit rounding. It introduces systematic negative operand offsets. On p=.95 Q4×K4's own trajectory the observed average residual is 7.4561 codes for Q and 7.4882 for K. The mean dequantized operand shift is -0.3645 for Q and -0.5104 for K; scales vary by head/token. Prefix zero fractions are .3091 for Q and .2585 for K, compared with source-code zero fractions .0276 and .0196. Prefixing collapses many distinct small values and changes their signs/magnitudes asymmetrically.

However, a positive mean score bias is not a sufficient explanation. Softmax is invariant to adding the same constant to all keys. The relevant error is e_j minus its row mean. The raw Q4×K4 mean logit error is +2.3296, whereas its mean per-head *centered* RMSE is .6961. Both numbers are reported to avoid confusing a large removable offset with ranking error.

The mixed terms also cancel; adding their separate magnitudes is invalid. For p=.95 Q4×K4's own inputs:

| Omitted term in z minus estimate | Mean score contribution | Mean per-head centered RMS | Mean correlation with total missing centered score |
|---|---:|---:|---:|
| 16 HQ·LK | -2.2789 | .5040 | .6859 |
| 16 LQ·HK | -2.2564 | .5848 | .8173 |
| LQ·LK | +2.2057 | .3224 | -.4053 |
| Sum | -2.3296 | .6961 | 1.0000 |

All entries include the actual scales. Centered RMS values do not sum because terms are correlated. The positive low×low term partially cancels the negative mixed terms.

A useful explanatory split is LQ=7.5+εQ and LK=7.5+εK. As an *algebraic measurement*, the centered RMS of c·7.5·sum(K) is .4234, whereas c·7.5·sum(Q) is .0489. The latter can vary across keys through the per-key scale; it is not necessarily constant. The former depends directly on each key vector's coordinate sum. This asymmetry is consistent with Q-prefixing being especially harmful here. It does not prove that centering prefixes would recover quality: centered residual variation, coordinate correlations, token-dependent scales, and other terms remain. No corrected predictor was run successfully.

## Same-input estimator comparison isolates selection error

The next table evaluates all three established estimator definitions on the *same Q4×K4 trajectory operands* at each threshold. Final PV uses full-INT logits on the selected set and the actual saved V; the reference PV uses full-INT Top-p on those same operands. Relative PV error is the head vector L2 error divided by the reference head vector norm.

| p | Estimator | Centered logit RMSE | Spearman rank correlation | Mean selected keys | Mean retained full-INT probability | Relative PV error |
|---|---|---:|---:|---:|---:|---:|
| .95 | full INT | 0 | 1 | 27.9583 | .95546 | 0 |
| .95 | Q4×K4 | .69606 | .89436 | 41.4688 | .96099 | .11015 |
| .95 | Q8×K4 | .38485 | .96251 | 29.4583 | .95633 | .06222 |
| .90 | full INT | 0 | 1 | 18.8698 | .91868 | 0 |
| .90 | Q4×K4 | .69754 | .89344 | 27.5677 | .92817 | .17803 |
| .90 | Q8×K4 | .38636 | .96138 | 19.5781 | .91976 | .08070 |
| .85 | full INT | 0 | 1 | 15.5885 | .87808 | 0 |
| .85 | Q4×K4 | .66914 | .89012 | 22.7656 | .88918 | .25740 |
| .85 | Q8×K4 | .37593 | .95803 | 16.2292 | .88388 | .09417 |

Retained mass is measured under the full-INT *full distribution*, before restricting to the set. Predicted Top-p mass uses the estimator's distribution and therefore is not a guarantee about retained full-INT mass. Full-INT reference mass itself often exceeds p because the crossing key must be included.

The p=.95 result is also reproduced on the INT trajectory: Q4×K4 centered RMSE .6930 and PV error .1079; Q8×K4 .3831 and .0646. Thus the large Q4 selection error is already visible on inputs produced by the INT control; it cannot be dismissed entirely as a consequence of an already-diverged Q4 trajectory. Conversely this posthoc comparison is not a counterfactual whole-model PPL measurement.

## Diffuse on average, occasionally catastrophically overconfident

At p=.95 on Q4's own inputs, mean estimator entropy is 2.6346 nats versus full INT 1.8670. Mean selected size rises from 27.96 to 41.47. Yet the mean logit standard-deviation ratio is 1.0068, near one: a simple global shrinkage of all logits does not explain this distribution change. Ranking, tail shape, and individual head behavior matter.

Average retained mass .96099 conceals failures: 59/192 heads retain less than .95, four retain less than .90, and two omit the full-INT argmax. Minimum retained mass is .38094. At p=.90, 55/192 fall below .90. At p=.85, 60/192 fall below .85 and minimum mass is .1805. These are empirical smoke-head counts, not corpus rates or confidence intervals.

Concrete captured example: layer 0, query 128 decode, head 2, p=.95. Full INT assigns key 127 probability .404557 and key 128 probability .380937. Q4×K4 assigns them .012872 and .957162 respectively. It selects only key 128, while full INT selects six keys. Full-code recomputation on this singleton necessarily returns probability one at key 128; it cannot recover key 127 or key 126, whose full-INT mass is .143139. Its retained full-INT mass is .380937 and relative head-PV error 1.253888. This head is *too concentrated*, despite the average estimator being more diffuse. `worst_head_example.json` preserves exact scores and key positions.

Another failure, layer 31 query 127 prefill head 7, retains .533637 with three selected keys versus five under full INT and head-PV error .799651. Oversized average sets therefore do not prevent small, badly chosen sets in sensitive heads. A mean retained-mass or mean selected-count statistic cannot diagnose this alone.

## Layer and trajectory effects

p=.95, Q4 estimator on its own inputs, averaging the two trace positions and all heads per layer:

| Layer | Centered RMSE | Spearman | Q4 keys | INT keys | Retained INT mass | Relative head-PV error |
|---|---:|---:|---:|---:|---:|---:|
| 0 | .4418 | .9559 | 60.4219 | 62.2344 | .9318 | .0868 |
| 16 | .7816 | .8770 | 30.9688 | 11.4062 | .9781 | .1051 |
| 31 | .8648 | .8503 | 33.0156 | 10.2344 | .9731 | .1385 |

Late layers show worse rank agreement and much larger sets. Layer 0 has the lowest mean retained mass and contains the worst single-head failure. High average late-layer retained mass does not imply negligible PV error: the INT Top-p reference and the larger Q4 set redistribute probability differently, and V directions and cancellation influence output sensitivity.

Actual cross-run Q/K are equal at layer 0, before any attention divergence. At layer 16, Q4 versus INT Q relative difference averages .0879 and K .1038; at layer 31, .1072 and .1320. Q8×K4 differences are smaller: layer 16 .0332/.0413; layer 31 .0395/.0421. Layer-output relative differences at layers 0/16/31 are .1629/.1427/.3753 for Q4 versus .0086/.0502/.1088 for Q8×K4. Whole-layer quantities are shared across heads in the raw table and are not 32 independent measurements.

Actual selected-set Jaccard against the INT run falls from .8543 at layer 0 to .3320 and .2900 for Q4; Q8×K4 gives .9194, .7307, .6757. These cross-run set comparisons include upstream trajectory changes. They must not be interpreted as estimator error on fixed inputs; the separate same-input analysis supplies that comparison.

## Verification, limitations, and next decision

The existing source quantization and trace replay already implement independent CPU validation. This analysis additionally reproduced all 72 captured Top-p masks across four variants, three thresholds and six targets exactly (`verification.json`). Its exact term identity was checked for all 1,728 integer-source/head observations. Existing full-run metrics, not CPU recomputation, supply the reported PPL.

Evidence supports rejecting the explanation that Q4×K4 merely needs more selected keys, and supports mixed-term/offset-related ranking distortion as a concrete mechanism. It supports Q8×K4 as a successful control for retaining Q precision in this implementation. It does not isolate residual noise from offset effects causally, establish a unique best representation, assign NLL loss to an individual layer/head, or establish hardware speed/area/power. No new remedy, corrected logits, additional PPL or GPU model experiment has been executed by this analysis.

Supplemental causal experiments need a separate approved SPEC after the requested format study. Candidate hypotheses and controls are recorded in `SUPPLEMENT_SPEC.md`; all remain planned and approval-required.

## Artifact index

- `analyze_tc.py`: successful reproducible CPU observational script; uses existing estimator definitions only.
- `head_metrics.csv`, `summary.csv`, `layers.csv`: full per-head and aggregated estimator comparisons.
- `code_statistics.csv`: observed codes, scales, residuals and prefix frequencies.
- `decomposition.csv`: exact component scores, centered RMS and correlations.
- `trajectory.csv`: actual run-to-run operand and selected-set changes.
- `worst_head_example.json`: observed key-level miss.
- `manifest.json`, `verification.json`: execution scope and captured-mask checks.

Reproduce with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /output/project/quality-venv/bin/python analyze_tc.py`. This reads the saved smoke trace directories named in the manifest and writes only this diagnosis directory. It does not start an experiment or load model weights.

## Completion update, 2026-10-28

Added verified p=.85 full-window results after the original diagnosis was written. This update reads completed metrics only; no CPU tensor analysis or GPU run was repeated. Experimental code identity remains the frozen hashes in the existing run manifests; subsequent live checkout commits do not redefine that identity. The diagnostic conclusions and unexecuted supplemental SPEC retain their original scope. See `report_update.json` for hashed source evidence and `artifact_index.json` for this diagnosis bundle.
