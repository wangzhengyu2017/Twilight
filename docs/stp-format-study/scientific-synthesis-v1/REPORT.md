# Completed format-predictor study: results and mechanisms

The eighteen new format-predictor runs passed independent full-run acceptance. Together with fourteen historical FP/INT/TC controls, this report contains thirty-two measured rows; each uses Llama-3.1-8B-Instruct, frozen WikiText-2, 282 windows and 289,075 effective targets. Lower PPL is better. Six historical RS placeholders in the original delivery are not measurements. Newly authorized non-RS and RS experiments are separate, still pending, and are not represented by these results.

The main finding is that four-bit TC prefixes can damage relative attention logits and remove important keys; full-code recomputation cannot recover excluded keys. Native MXFP4 substantially improves Q4 predictor quality here. SM also recovers quality but selects many more keys, so its advantage cannot be attributed solely to preserved information or better ranking. Neither these curves nor selected counts demonstrate hardware speedup or establish a unique causal share of corpus loss.

## Measured results

| Field | Baseline |
|---|---|
| Model | Llama-3.1-8B-Instruct |
| Task | WikiText-2 raw test |
| FP dense PPL | 6.484759 |
| INT dense (torch Q/DQ) PPL | 6.486693 |

Historical dense FP/INT values are measured, but their matching archived full release gate is missing; the delivery marks these two rows verified=no. The other thirty completed rows are verified=yes. This limitation is retained, not replaced by a new pass claim.

| Method | p=.95 PPL | p=.90 PPL | p=.85 PPL |
|---|---:|---:|---:|
| FP TopP | 6.482529 | 6.530140 | 6.613165 |
| INT TopP | 6.485039 | 6.532782 | 6.614929 |
| TC MSB Q4K4 | 6.836825 | 7.317663 | 7.921542 |
| TC MSB Q8K4 | 6.500290 | 6.563772 | 6.667327 |
| SM MSB Q4K4 | 6.497612 | 6.557247 | 6.675108 |
| SM MSB Q8K4 | 6.498413 | 6.574305 | 6.697492 |
| MXINT8-TC4 Q4K4 | 6.673347 | 6.921718 | 7.238384 |
| MXINT8-TC4 Q8K4 | 6.503503 | 6.571011 | 6.675618 |
| Native MXFP4 Q4K4 predictor | 6.492194 | 6.544645 | 6.638278 |
| MXINT8-Q8 / MXFP4-K4 predictor | 6.484025 | 6.530161 | 6.603273 |

[All exact NLL/PPL/target values, statuses and run paths](MEASURED_RESULTS.csv) are copied from the accepted delivery table, not recomputed from rounded PPL. For every row, NLL is the sum over the same 289,075 scored next-token targets; mean NLL=NLL/targets and PPL=exp(mean NLL). Shared target counts establish matched coverage, not identical model probabilities.

## Quality and selection budget

The following fractions count selected key pairs divided by all causal-allowed key pairs across the complete scored-window forwards, including all layers/heads and context positions. They are computation proxies, not timed sparse-kernel measurements.

| Method at p=.95 | PPL | Selected / allowed | Fraction / INT TopP fraction |
|---|---:|---:|---:|
| INT TopP | 6.485039 | 0.141888 | 1.000 |
| TC MSB Q4K4 | 6.836825 | 0.230227 | 1.623 |
| TC MSB Q8K4 | 6.500290 | 0.151824 | 1.070 |
| SM MSB Q4K4 | 6.497612 | 0.489443 | 3.450 |
| SM MSB Q8K4 | 6.498413 | 0.297910 | 2.100 |
| MXINT8-TC4 Q4K4 | 6.673347 | 0.193648 | 1.365 |
| MXINT8-TC4 Q8K4 | 6.503503 | 0.142893 | 1.007 |
| Native MXFP4 Q4K4 predictor | 6.492194 | 0.189258 | 1.334 |
| MXINT8-Q8 / MXFP4-K4 predictor | 6.484025 | 0.163698 | 1.154 |

SM Q4's broad selection is an important alternative explanation for low PPL. At identical p, its distribution is more diffuse, so the cumulative threshold can require many more keys. Fixed p is not a fixed key-count budget. Native MXFP4 Q4 improves PPL over MXINT8-TC4 Q4 while using slightly fewer global key pairs at all three thresholds, which cannot be explained by a larger aggregate selected count alone. It still does not isolate local rank/calibration, metadata, trajectory or error-distribution contributions. Mixed MXINT8-Q8/MXFP4-K4 slightly outperforms original INT TopP in these single-corpus runs while selecting more global pairs; no significance or universal superiority is inferred.

[Verified data and readable figures](../analysis-final-readable/RENDER.manifest.json): [PPL versus threshold](../analysis-final-readable/ppl_vs_top_p.png) and [PPL versus selected fraction](../analysis-final-readable/ppl_vs_selected_fraction.png). PDF versions sit beside the PNGs. Threshold marker shapes replace overlapping point annotations; values and axes are unchanged from the fully verified first rendering.

## Why TC Q4 can fail

For a source integer x, the ordinary two's-complement prefix reconstructs T(x)=16 floor(x/16), leaving x-T(x) in {0,...,15}. In particular, -1 reconstructs as -16 while +1 reconstructs as 0. This is a many-to-one, asymmetric error pattern around zero. With qhat=q-rq and khat=k-rk, the exact real-arithmetic logit difference is

`zhat_j-z_j = (-rq·k_j - q·rk_j + rq·rk_j)/sqrt(D)`.

A row-constant bias cancels under softmax; key-dependent errors change ranks and cumulative mass. Q8K4 removes query residual terms, but cancellation prevents a universal theorem that every head improves. The saved same-input p=.95 analysis measured centered-logit RMSE .69606 for TC4 versus .38485 for TC8. This is direct local evidence, not a decomposition of total PPL.

At layer0/query128/head2, original probabilities at keys127/128/126 are .404557/.380937/.143139. TC4 assigns key128 predictor probability .957162 and selects only that key at p=.95. It retains original mass .380937, misses argmax127, and gives relative PV error1.253888. Full Q8/K8 recomputation within the singleton cannot bring keys127/126 back. Same-input MXINT8-TC4 selects the same singleton. SM4 selects60 keys and retains .996383 original mass despite still ranking128 above127; SM8 selects29 keys and retains .987668. This distinguishes recovered coverage from recovered ordering.

If P is the full-score distribution, Q the predictor distribution and Q(S)>=p, then `P(S)>=max(0,p-TV(P,Q))`; the predicted threshold alone does not guarantee original mass p. If the maximum centered logit error is at most epsilon, `P(S)>=p*exp(-2*epsilon)`. An aggregate RMSE cannot certify that maximum-error premise. With m=P(S), full attention A and renormalized selected attention A_S satisfy `A-A_S=(1-m)*(A_out-A_S)` and `||A-A_S|| <= (1-m)*diam(V)`. Missing probability mass is therefore relevant but is not numerically identical to relative PV error or to NLL degradation.

## Information, calibration and trajectory limits

The stored prefix entropy/conditional variance establish lost scalar information. They do not provide a lower bound on attention error or PPL because scales, vector context, correlations and error cancellation matter. On byte-identical layer0 inputs (64 deduplicated query/head observations), SM4's mean predictor/original logit-standard-deviation ratio is .660946 and entropy increase is .614649 nats. Its original argmax is not predicted rank1 in27/64 heads versus21/64 for TC4. SM changes both spread and rank, not just a uniform temperature. The paired report separates same-input estimates from later-layer cross-run trajectories, which can contain different activations.

MXINT8-TC4 first requantizes source DQ into block32 MXINT8 and then keeps a TC prefix. Native MXFP4 uses its actual E2M1 four-bit code; it is not another eight-bit MSB truncation. Both carry block exponent metadata, but neither a common bit count nor this matched study proves a universal pointwise error ordering. All original eighteen new jobs use their format only to predict the mask and then score selected keys with original source full Q8/K8 exactly once. Their names must not be mistaken for the newly requested dense full-format benchmarks.

The supplied BADA paper has now enabled a separate RS design and independent SPEC review. Native RS is range-shifted signed-slice coding; the adopted source-scale adaptation explicitly clips source120..127 to119. Its new experiments will report this saturation confound rather than present recoding as whole-domain lossless. No RS measurement is included in this completed-study table.

## Evidence, remaining tasks and publication

The mathematical bounds and saved-tensor observations are supported by [the mathematical note](../diagnosis-math/NOTE.md), [calibration observations](../diagnosis-math/LOGIT_CALIBRATION_OBSERVATIONS.md), [metric definitions](../diagnosis-math/METRIC_DEFINITIONS.md), and [independent observation review](../review/calibration-observations-review.md). Legacy ordinal-rank correlation is not average-tie Spearman; follow the recorded metric definition. Evidence uses real saved tensors, not newly fitted causal interventions.

[Accepted delivery](../results/REPORT.md), [paired-set table](../results/PAIRED_SETS.csv), [terminal audit](../results/INDEPENDENT_AUDIT.json), and [closed execution index](/output/project/experiments/stp-format-study/execution-v1/artifact-index.json) preserve trace and run provenance. The eighteen new jobs contain108 real traces. Their paired table has72 variant/reference summaries; individual paired rows remain in the indexed manifests. Original FP/INT controls retain the historical provenance limitations described in the delivery report.

The initial result publication is commit `baf8b70c0d794313b1d68df4a17a712fb47dd696` on `STP_test_Format`, verified against the SSH remote. It is a result/documentation commit; the eighteen runs used source commit `124b78ae3e542bba3dc66bb35e58048373892ced`. This repository copy publishes the later synthesis and improved figures after independent document review; see [publication status](PUBLICATION_STATUS.md). The [other-cause supplement v2](/output/project/experiments/stp-format-study/supplement-proposal-v2/SPEC.md) remains proposed and unexecuted pending user approval. The separately authorized non-RS10 and RS10 jobs proceed without treating this initial publication as completion of the expanded task.
