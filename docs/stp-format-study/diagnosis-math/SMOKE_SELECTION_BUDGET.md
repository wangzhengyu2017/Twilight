# Actual smoke selection: SM retains more mass while spending more keys

Date: 2026-10-28. Status: descriptive aggregation of existing saved actual smoke evidence, not a new experiment. Full-window format PPL is pending; no quality conclusion is drawn here. No predictor, counterfactual, model, or GPU job was evaluated. A separately requested read-only comparison checks exact equality of existing captured inputs and selected membership. Existing NOTE/review, source, specifications, frozen execution roots, and completed diagnoses were not changed.

The SM smoke results support treating **selection budget as a competing explanation** for their high retained mass. At p=.95, SM Q4K4 selects 64.2865 keys per head against its own-input original-INT Top-p reference's 26.9115: 2.3888 times the reference count. MX Q4K4 selects 32.7344 against 26.9271: 1.2157 times. SM retains more original probability (.986988 versus .954827) and almost all reference-set keys, while having lower set precision and lower saved rank correlation. These observations do not establish that SM improves model quality or that extra keys are its unique mechanism.

## Evidence and aggregation scope

All 12 cells have 192 own-source head/query rows: layers 0/16/31 × prefill query127/decode query128 × 32 heads. Each row compares the actual predictor-selected set S against original full-INT Top-p set T on **that candidate's own saved source operands**, labelled `own_source_int_full` / `same_input_estimator`. The source pairing manifests contain six passed candidate replay captures per cell. Every own-source identity is unique and its saved predicted count equals the replay statistic's saved selected count.

[Own-source CSV](smoke_selection_own_source.csv) records all means, tails, per-head count ratios, and saved same-input logit statistics. [Cross-TC CSV](smoke_selection_cross_tc.csv) separately records actual saved historical TC-set pairings. [Hash manifest](smoke_selection_manifest.json) binds the 24 source JSONL/manifest files, output CSVs, note, and [aggregation script](/output/project/experiments/stp-format-study/diagnosis-math/summarize_smoke_selection.py). The script checks JSONL SHA256 against each original pairing manifest; it does not rehash tensors or rerun replay. The table aggregation uses saved scalar statistics and sets only; the additional equality check below reads existing captured tensors without recomputing logits or probabilities.

Rows receive equal weight. Count ratio below means sum(|S|)/sum(|T|), equivalently ratio of mean counts; it is **not** the mean of per-head ratios, which is separately provided in CSV. Recall is |S∩T|/|T|, precision is |S∩T|/|S|, Jaccard is |S∩T|/|S∪T|. These describe unweighted set membership, while retained mass is P(S) under the full-original-INT distribution on the candidate's own inputs. Shortfall is max(0,p−P(S)). Reported table precision is rounded; CSV preserves computed floats.

## Retained mass and selected-key budget

| Format | Shape | p | Mean selected | Mean reference | Count ratio | Retained mass | Mean shortfall | Heads below p /192 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SM | Q4K4 | .95 | 64.2865 | 26.9115 | 2.3888 | .986988 | .000205 | 1 |
| SM | Q4K4 | .90 | 46.8698 | 19.7865 | 2.3688 | .967946 | .000206 | 1 |
| SM | Q4K4 | .85 | 36.5729 | 15.5365 | 2.3540 | .942732 | .001756 | 6 |
| MXINT8-TC4 | Q4K4 | .95 | 32.7344 | 26.9271 | 1.2157 | .954827 | .010359 | 68 |
| MXINT8-TC4 | Q4K4 | .90 | 23.7344 | 20.2031 | 1.1748 | .911411 | .018154 | 70 |
| MXINT8-TC4 | Q4K4 | .85 | 18.3125 | 15.9323 | 1.1494 | .869264 | .024072 | 67 |
| SM | Q8K4 | .95 | 43.7969 | 28.0833 | 1.5595 | .976158 | .000036 | 1 |
| SM | Q8K4 | .90 | 30.0417 | 19.8594 | 1.5127 | .948408 | .000231 | 4 |
| SM | Q8K4 | .85 | 23.4375 | 15.7448 | 1.4886 | .914585 | .003878 | 7 |
| MXINT8-TC4 | Q8K4 | .95 | 27.8333 | 27.7396 | 1.0034 | .951187 | .005662 | 96 |
| MXINT8-TC4 | Q8K4 | .90 | 19.4583 | 19.4323 | 1.0013 | .910909 | .007716 | 76 |
| MXINT8-TC4 | Q8K4 | .85 | 15.5052 | 15.3854 | 1.0078 | .872260 | .009440 | 74 |

The large difference between count ratio and recall matters. Selecting every legal key gives mass and recall one, although it says nothing about predictive ranking and spends the entire budget. SM does not literally select all keys; the example explains why higher mass/recall alone cannot isolate ranking quality. Conversely extra keys are not guaranteed to improve output relative to the INT Top-p reference: full-code normalization inside S changes weights, as shown in [NOTE §2](NOTE.md).

## Set overlap and saved same-input logit diagnostics

| Format | Shape | p | Recall | Precision | Jaccard | Centered RMSE | Rank correlation |
|---|---|---:|---:|---:|---:|---:|---:|
| SM | Q4K4 | .95 | .991312 | .354893 | .352188 | .917977 | .868659 |
| SM | Q4K4 | .90 | .982219 | .368259 | .363852 | .895328 | .864037 |
| SM | Q4K4 | .85 | .967804 | .409566 | .393581 | .872935 | .863206 |
| MXINT8-TC4 | Q4K4 | .95 | .924617 | .635780 | .580277 | .677581 | .903634 |
| MXINT8-TC4 | Q4K4 | .90 | .895551 | .651395 | .568677 | .674037 | .903166 |
| MXINT8-TC4 | Q4K4 | .85 | .894009 | .663671 | .585622 | .655541 | .903915 |
| SM | Q8K4 | .95 | .986418 | .520262 | .514478 | .609134 | .921017 |
| SM | Q8K4 | .90 | .981945 | .573913 | .565947 | .594660 | .920337 |
| SM | Q8K4 | .85 | .976382 | .619053 | .604753 | .593250 | .918224 |
| MXINT8-TC4 | Q8K4 | .95 | .884766 | .890175 | .783716 | .430362 | .953454 |
| MXINT8-TC4 | Q8K4 | .90 | .896832 | .896115 | .802102 | .422159 | .953808 |
| MXINT8-TC4 | Q8K4 | .85 | .898090 | .887147 | .799035 | .414403 | .954987 |

Centered RMSE and `stable_rank_correlation` are means of the already saved candidate replay diagnostics against its own original-INT logits. No new logits or ranks were computed. SM's high mass coexists with higher centered RMSE and lower rank correlation than MX in all six shape/threshold comparisons. This weakens interpreting its higher mass as evidence of superior ranking. Comparisons **between** candidates still involve different model trajectories: same-input here means candidate versus its own original-INT oracle, not that all formats were reevaluated on one shared hidden-state trajectory. Also, rank correlation over all legal keys does not by itself resolve important top-key ordering.

Historical TC pairings supply a separate budget context. Mean actual TC counts at p=.95/.90/.85 are 41.4688/27.5677/22.7656 for Q4K4 and 28.7500/20.2083/16.1667 for Q8K4. SM's actual mean count is larger at every matched shape/p; MX's is smaller. These are saved **cross-run trajectory** comparisons, not new TC predictions on SM/MX operands, and not evidence of fixed-input computational savings or measured hardware speed.

## Exact common-input contrast at layer 0

The [layer0 equality manifest](/output/project/experiments/stp-format-study/diagnosis-math/layer0_input_equality_manifest.json) records an additional read-only check against the historical TC Q4K4 p=.95 trace at the same query position. Across **all 24 layer0 candidate captures** (12 cells × two query positions), `torch.equal` succeeds for q/k codes, q/k scales, post-RoPE Q (`q_post_rope`), visible post-RoPE K (`k_visible`), q/k DQ tensors, V, legal mask, and key positions. Thus these layer0 captures support a local same-input format contrast, even though later layers involve differing trajectories. This checks captured operands; it does not assume equality merely from identical input token IDs.

For the original worst TC head, layer0/query128/decode/head2, the actual p=.95 saved selections are:

| Predictor | Selected keys | Retained original mass | Original argmax key127 included? | Keys128 /126 included? |
|---|---:|---:|---|---|
| Historical TC Q4K4 | 1 | .380937 | No | Yes / No |
| MXINT8-TC4 Q4K4 | 1 | .380937 | No | Yes / No |
| SM Q4K4 | 60 | .996383 | Yes | Yes / Yes |
| MXINT8-TC4 Q8K4 | 5 | .943952 | Yes | Yes / Yes |
| SM Q8K4 | 29 | .987668 | Yes | Yes / Yes |

The original INT Top-p set contains six keys. The historical TC singleton is verified directly from its saved actual mask; its mass is the existing [captured worst-head](/output/project/experiments/stp-format-study/diagnosis/worst_head_example.json) original probability of key128. Candidate masses are the saved own-source pairing statistics, and candidate memberships are checked against their saved actual masks. [Head CSV](/output/project/experiments/stp-format-study/diagnosis-math/layer0_query128_head2.csv) preserves all twelve format/shape/threshold observations, including exact selected key lists. The [comparison script](/output/project/experiments/stp-format-study/diagnosis-math/compare_saved_layer0.py) computes equality and membership only; it evaluates no new estimator.

On these exactly equal inputs, SM includes the missed high-mass keys while selecting 10 times the original INT Top-p count for Q4K4 (60 versus 6). MX Q4K4 reproduces the same singleton failure as TC in this head. This strengthens the observation that successful mass retention and selected-key expenditure must be interpreted together. It does not prove which of SM's ranking, probability, or budget changes is necessary: no matched-budget counterfactual was executed. Nor can this one head establish a whole-model quality ordering.

## A temperature theorem explains a possible budget mechanism, not SM causality

Fix logits z sorted in descending order and a fixed top-k set, with 0<k<n. Let

\[
F_k(\beta)=\sum_{i\le k}\operatorname{softmax}(\beta z)_i,\qquad\beta>0.
\]

Differentiating the partition sums gives

\[
\frac{dF_k}{d\beta}
=F_k(1-F_k)
 \left(E_{\beta}[z\mid i\le k]-E_{\beta}[z\mid i>k]\right)\ge0,
\]

because every top-k logit is at least every outside logit. All equal logits yield equality; a boundary tie alone does not force equality. Positive scaling preserves the sorted order (with a fixed tie rule). Therefore the minimal crossing count \(N_\tau(\beta)=\min\{k:F_k(\beta)\ge\tau\}\) is nonincreasing in β: reducing positive β can only leave it unchanged or increase it. It need not strictly increase. Pure temperature flattening can thus increase retained original mass by buying more keys even with identical ranking.

SM's coordinatewise odd magnitude shrinkage is **not** a uniform positive scaling of logits. Its coordinate errors depend on code magnitude, sign, query/key coordinates, and actual scales, and it can change ranks. These saved replay statistics contain no predictor/reference logit-standard-deviation field. The theorem illustrates a mechanism and does not establish that SM realizes a temperature change, that temperature explains its budget, or that its PPL will improve.

The existing [causal supplement proposal](/output/project/experiments/stp-format-study/supplement-proposal/SPEC.md), especially its count-matched full-INT ranking oracle and budget controls, addresses the broader matched-budget hypothesis. It remains unexecuted and requires its stated explicit approval. The present evidence supports reporting mass, rank diagnostics, and selected-key budget together while awaiting full-window format results; it does not activate a remedy or a supplemental experiment.
