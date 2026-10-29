# Saved layer0 logits: SM is more diffuse and spends more selected keys

Date: 2026-10-28. Status: descriptive analysis of existing actual captures only. Full-window format PPL remains pending. No logits or probabilities were regenerated or altered; no temperature was fitted; no new Top-k/Top-p set, predictor, counterfactual, model execution, or GPU work was performed. Closed reports, gates, and frozen inputs are unchanged. “Calibration” here describes observed distribution spread, not a measured calibration score against held-out outcomes.

On exactly common layer0 inputs, SM has narrower saved logits and higher saved probability entropy than original INT. It selects more keys and retains more original mass, while often moving the original argmax away from predicted rank one. Thus improved retained mass cannot by itself establish improved ranking. These are local observations; they do not identify the cause of whole-model PPL changes.

## Inputs, deduplication, and definitions

Six predictors are included: historical TC Q4K4/Q8K4 and actual SM/MXINT8-TC4 Q4K4/Q8K4, using p=.95 captures at layer0, prefill query127 and decode query128, each with 32 heads. This gives **64 head/query observations per predictor, 384 rows total**. We verified identical dtype/shape and contiguous uint8 bytes against the TC Q4K4 reference for q/k codes, scales, post-RoPE Q/K, DQ q/k, V, legal mask, and key positions. Saved derived original-INT logits and probabilities are also byte-identical across all six predictors at each query position.

For all four format/shape combinations, their p=.90 and .85 captures have byte-identical original inputs **and predictor logits/probabilities** to p=.95 at these layer0 positions. They are therefore deduplicated rather than counted as additional logit evidence. Their actual selected masks can differ with p; the descriptive selection counts and masses below use only the saved p=.95 masks. Historical TC8's common-input equality was checked directly rather than assumed from the earlier TC4 comparison.

All spread calculations use saved legal-key logits, already in real scaled attention-logit units including 1/√D. Standard deviation is the population value (`correction=0`), computed in FP64; reported ratios are means of each head's predictor/original standard-deviation ratio. Entropy is −Σp ln p over positive **saved** probabilities, in nats, without re-softmax or renormalization. Original mass sums the saved original probabilities over the saved selected mask. Original top-key rank is a one-based descending stable ordinal rank: equal logits use ascending key position. It is a rank observation, not a newly selected set; [metric definitions](METRIC_DEFINITIONS.md) explain tie handling.

[Manifest](logit_calibration_manifest.json) binds all input files and output artifacts, records byte-equality checks and threshold deduplication checks. [Script](/output/project/experiments/stp-format-study/diagnosis-math/observe_saved_logit_calibration.py), [384 head rows](layer0_logit_calibration_heads.csv), [six-row summary](layer0_logit_calibration_summary.csv), and [worst-head key values](layer0_worst_head_key_probabilities.csv) preserve the calculations. Existing actual predictor probabilities are used directly; tiny differences from old CPU-recomputed diagnostic probabilities are not corrections to historical evidence.

## Distribution spread, original argmax rank, and budget

Common original-INT means are logit standard deviation 1.831107 and entropy 3.624620 nats. Every table row equally weights the same 64 head/query identities.

| Predictor | Mean std ratio | Mean entropy change | Mean original-argmax predicted rank | Original argmax not rank1 /64 | Mean selected count | Mean original retained mass |
|---|---:|---:|---:|---:|---:|---:|
| TC Q4K4 | 1.027762 | −.074919 | 1.531250 | 21 | 60.421875 | .931828 |
| TC Q8K4 | 1.007405 | +.010748 | 1.296875 | 15 | 61.906250 | .950370 |
| SM Q4K4 | .660946 | +.614649 | 3.406250 | 27 | 87.562500 | .983226 |
| SM Q8K4 | .847279 | +.318306 | 1.703125 | 20 | 73.984375 | .970863 |
| MXINT8-TC4 Q4K4 | 1.021940 | −.085190 | 1.406250 | 15 | 60.515625 | .932505 |
| MXINT8-TC4 Q8K4 | 1.011942 | −.027660 | 1.250000 | 11 | 61.468750 | .948498 |

SM Q4K4 has std ratio below one in 64/64 heads and entropy above original in 64/64; SM Q8K4 has the respective counts 64/64 and 63/64. This is stronger evidence of a diffuse local SM distribution than selected counts alone. Meanwhile original argmax rank is worse on average than both same-width TC and MX. A rank change alone need not remove a key: both SM shapes keep the original argmax in all 64 saved masks, while TC4 and MX4 each omit it in one head. Counts and rank/mass statistics answer different questions.

These layer0 results must not be substituted for the earlier all-layer averages. For example, the previous TC diagnosis reports higher average predictor entropy over its full six-capture layer0/16/31 scope, whereas the deduplicated common-input layer0 TC4 mean entropy change here is negative. Layer scope and trajectory scope both matter.

## Captured query128/head2: flattening, changed ordering, and actual inclusion

The original probabilities at keys127/128/126 are .404557/.380937/.143139. The original argmax is key127. Each cell below shows **saved predictor probability (descending rank)**; all values refer to exactly identical original operands and reference probabilities.

| Predictor | Key127 | Key128 | Key126 | Saved count | Retained original mass |
|---|---:|---:|---:|---:|---:|
| TC Q4K4 | .012872 (3) | .957162 (1) | .013823 (2) | 1 | .380937 |
| TC Q8K4 | .174905 (2) | .631645 (1) | .115439 (3) | 7 | .955313 |
| SM Q4K4 | .120459 (2) | .193898 (1) | .087040 (3) | 60 | .996383 |
| SM Q8K4 | .305132 (1) | .256858 (2) | .183645 (3) | 29 | .987668 |
| MXINT8-TC4 Q4K4 | .008163 (3) | .965485 (1) | .009742 (2) | 1 | .380937 |
| MXINT8-TC4 Q8K4 | .185671 (2) | .665198 (1) | .090044 (3) | 5 | .943952 |

TC4/MX4 select only key128, excluding the original argmax127 and key126. Both SM shapes and both Q8 controls include all three keys. SM4 still ranks128 above127, but its diffuse distribution and 60-key mask include both. SM8 also restores their original ordering in this head; its selection still spends 29 keys versus the original INT Top-p count of six. Thus the observed successful mass retention can accompany both rank changes and a large budget increase; this comparison does not isolate their separate causal contributions.

## What follows, and what does not

The [temperature theorem](SMOKE_SELECTION_BUDGET.md) proves that reducing a positive uniform logit scale while preserving ranks can increase the Top-p crossing count. The measured SM std ratios and entropy changes are consistent with a flattening component, but **do not prove uniform scaling**. In fact, original-argmax rank changes in 27 SM4 and 20 SM8 heads preclude representing every observed head as a positive scaling of its original logits plus a constant. Coordinatewise SM magnitude truncation can alter relative logits as well as spread; no temperature parameter was estimated here.

The descriptive evidence supports comparing retained mass together with distribution shape, rank relevance, and selected-key expenditure. It establishes neither a matched-budget quality advantage nor a corpus-level causal attribution. The [existing matched-budget proposal](/output/project/experiments/stp-format-study/supplement-proposal/SPEC.md) remains unexecuted and requires its stated approval. This analysis creates no additional selector or intervention evidence.
