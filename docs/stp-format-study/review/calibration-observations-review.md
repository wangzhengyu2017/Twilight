# Calibration observations: independent frozen-artifact review

2026-10-28. **Pass for the stated descriptive scope; no blocking finding.** The saved layer0 evidence supports narrower SM logits, higher entropy, and higher retained original mass accompanied by larger selections. It does not establish improved ranking, measured calibration against outcomes, a matched-budget advantage, or a cause of PPL changes. The note makes these limits explicit.

## Verification

I independently loaded the existing tensors on CPU, checked their hashes and bytes, and recalculated descriptive statistics using Python `statistics.pstdev`, `math.fsum`, and stable descending sorting. I did not run the observation script, import/execute predictor code, regenerate logits/probabilities, select a new mask, fit temperature, or run a model/GPU experiment. Only this review was added under the experiment tree; no source/specification/Git/frozen-input changes were made.

- All **40 source files** and **five bound artifacts** match the calibration manifest's SHA256 values. The manifest itself is identified below.
- All **364 field comparisons** pass dtype, shape, and contiguous-byte equality: 12 predictor/query comparisons × (11 original-input fields + two original-reference fields), plus 16 threshold-duplicate comparisons × 13 fields. This directly includes historical TC Q8K4 and original reference logits/probabilities. Key positions are ascending contiguous positions in every checked capture.
- All **384 unique head/query rows**, six summary rows, and 18 worst-head/key rows agree with independently read saved values. Maximum absolute difference across independently recomputed numeric head fields is **2.6645352591003757e-15**. Reported six-decimal table values round correctly.
- The population is six predictors × two layer0 queries × 32 heads, equally weighted. The .90/.85 deduplication applies to the four SM/MX format/shape combinations and their saved inputs and predictor logits/probabilities; it does not claim identical masks or additional independent observations.
- Population standard deviation uses legal logits in attention units including 1/√128. Entropy uses positive saved probabilities without softmax or renormalization; their predictor sums range from 0.9999998681 to 1.0000001092. The ratio column is the mean of per-head ratios. These definitions are implemented as documented.

The SM4/SM8 standard-deviation counts are 64/64 below original; entropy is higher in 64/64 and 63/64. Their original-argmax ranks average 3.406250/1.703125, with 27/20 departures from rank one. Both include that argmax in all saved masks; TC4/MX4 each omit it once. The query128/head2 key probabilities, ranks, membership, counts and masses all match. Its original INT Top-p count of six is confirmed from the existing saved reference mask and count, without selecting a set.

The temperature theorem is correctly restricted to a positive uniform logit scale and fixed tie rule: decreasing scale cannot decrease the Top-p crossing count, but need not increase it. SM's observed rank changes rule out an exact positive affine transformation of the original logits in those heads. A lower standard deviation alone would not prove uniform scaling. The note also correctly separates these layer0 common-input results from earlier all-layer/trajectory averages and leaves the proposed matched-budget intervention unexecuted.

## Conditional MXFP4 comparison: arithmetic only

The frozen `predictor_formats.py` determines the following conventions. Let a finite nonzero block of 32 binary32 source values have maximum absolute value A, e=⌊log₂ A⌋, b=2^(e−2), and x=v/b. Assume both branches accept the block and all reconstructions remain finite. Their shared-exponent range requirements intersect at **−125 ≤ e ≤ 127**; this condition alone does not guarantee acceptance of every endpoint reconstruction. Zero blocks take a separate exponent-zero path.

For a **four-bit MXINT8-TC4 operand**, the code stores an INT8 sidecar with scale b/16, rounds ties to even, clamps to [−127,127], then uses signed floor division by 16:

\[
T(v)=b\left\lfloor\frac{\operatorname{clamp}(\operatorname{RNE}(16x),-127,127)}{16}\right\rfloor.
\]

Thus the predictor alphabet is b·{−8,…,7}. Floor is essential: this is neither sign-magnitude truncation nor nearest-integer four-bit rounding. Away from saturation, level kb has normalized cell [k−1/32,k+31/32), with ties at the left endpoint included. Its reconstruction error is predominantly downward but can be positive by up to b/32. Across |x|<8 the error obeys **−b < T(v)−v ≤ b/32**; the positive endpoint's INT8 clamp extends the final level-seven cell. This is a pointwise bound, not an expected bias under an assumed input distribution. For Q8K4, the Q operand instead retains the full INT8 sidecar reconstruction; the formula above applies to its four-bit K operand.

The conditional FP4 branch uses exponent e−2 and the E2M1 magnitude table {0,.5,1,1.5,2,3,4,6}. It selects the nearest table value, resolves exact ties by even magnitude encoding, applies the sign, and canonicalizes zero's sign:

\[
F(v)=b\,\operatorname{RNE}_{E2M1}(x).
\]

For example, magnitudes .25→0, .75→1, 2.5→2, 3.5→4, and 5→4. It saturates to magnitude six above six, with a symmetric signed alphabet. `bits` does not further truncate this FP4 branch: full and predictor reconstructions coincide. Binary32 conversion, bounded `ldexp`, finite E8M0 checks, and final reconstruction checks remain part of the implementation contract.

There is **no universal pointwise accuracy ordering**. At x=.5, FP4 is exact while TC4 gives zero; at x=5, TC4 is exact while FP4 gives four; at x=7.5, TC4 gives seven while FP4 gives six. FP4 error is at most b for |x|≤6 and less than 2b over |x|<8, with the larger tail error arising from saturation. These are algebraic examples and bounds, not predictor experiments.

These conventions establish different operand error structures. They do not establish that FP4 will sharpen or flatten attention logits, improve ranks, spend fewer keys, or improve PPL. Signed products, Q/K pairing, cancellation, block scales, saturation incidence, and the Q8K4 versus Q4K4 layout all matter. No conditional MXFP4 execution or causal performance prediction is included in this review.

## Source identity

SHA256 values below identify the reviewed versions. Paths in the first seven rows are relative to `diagnosis-math/`; the manifest binds all 40 verified source tensor files.

| Source | SHA256 |
|---|---|
| `LOGIT_CALIBRATION_OBSERVATIONS.md` | `f4f4f83ee0fe5c153a979c8d3ebe85800d86e0d708e3af26f4886a51fda006bc` |
| `logit_calibration_manifest.json` | `009695d400262ef91d7fdad3967709cc69c3ecad3fb7cf552503e4673c81030b` |
| `observe_saved_logit_calibration.py` | `2316ca504d23231da2e2547ab161ebfb47434fd7fd91ba545710a5a759a9c463` |
| `layer0_logit_calibration_heads.csv` | `5f748178295962545b902ebfe515cfe77277e5bec5360e318eddff12ad268629` |
| `layer0_logit_calibration_summary.csv` | `6928b97d97415da4d5232a63a6e679c69068fb66a76a5f22466e3407606a46ad` |
| `layer0_worst_head_key_probabilities.csv` | `ca7fb32a13190c377cffe699619bafb99193e72d1bae65db5de480258025e5d7` |
| `METRIC_DEFINITIONS.md` | `926c813d1215cb54b37fee96b6007a3db7208469eba4c1df4394afff3e4623dc` |
| `diagnosis-math/SMOKE_SELECTION_BUDGET.md` | `5e7eeb1d49184dd21784a35d233bb47a897871bacaad8bf95d12bc9dfd59aa46` |
| `execution-v1/freeze/source/twilight/pyimpl/quant_attention/predictor_formats.py` | `2c9ab253372aaab4b8171432a542c09d26a502c2d02674bd65dd7aee52ef3627` |
| `execution-v1/freeze/source/twilight/pyimpl/quant_attention/format_backend.py` | `0cdd19f353ef57929f96552efeefc2ed17b39365ea625420894287d434782350` |
