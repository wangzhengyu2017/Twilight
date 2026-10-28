# MX execution and conditional MXFP4 decision — preparation v1

Date: 2026-10-28. State: planned; no format implementation, format PPL, or GPU validation is claimed. This decision is fixed before inspecting any MX format full-run result. The user authorized autonomous technical choices, MXINT8 first, and MXFP4 conditionally if MXINT8 is poor. This note proposes a concrete interpretation for incorporation into the frozen SPEC; it does not launch a run or release the current source/GPU freeze. It refines the arithmetic shorthand in SPEC §4.3. Existing TC/SM arithmetic remains unchanged.

## 1. Predeclared conditional rule

Let H[f,w,p] = full NLL / verified target count for format f, width w, threshold p. Use p∈{.95,.90,.85}; all cells must use identical frozen tokens, model, coverage and scoring protocol. Define mean cross-entropy over the three engineering points:

`Hbar[f,w] = (H[f,w,.95] + H[f,w,.90] + H[f,w,.85]) / 3`.

`exp(Hbar)` is the geometric mean of the three PPLs. This is an equal-weight summary of three chosen operating points, not the PPL of a new corpus. Original INT-full Top-p depends on p but has no predictor-width distinction; reuse the same matched INT reference for both width summaries. Compute from unrounded NLL/targets in CPU float64, never displayed rounded PPL.

**Primary decision uses Q4K4. Declare MXINT8 poor and activate MXFP4 if either:**

1. `Hbar[MXINT8,Q4K4] >= Hbar[TC,Q4K4]` (no aggregate improvement over the matched existing TC predictor), or
2. `Hbar[MXINT8,Q4K4] - Hbar[original_INT_full] > log(1.01)` (more than1% aggregate PPL above matched original INT-full Top-p).

Equality with TC triggers; equality with the1% INT boundary does not. The1% margin is an explicitly chosen engineering tolerance, not a statistical confidence bound, significance threshold, correctness property or literature-established standard. Any strictly lower value than TC satisfies the first part regardless of its size; report tiny changes as tiny changes. No rounding epsilon, post-result margin change, best-p choice or replacement by dense/FP reference is allowed. Archive the signed CE differences, aggregate PPL ratios and each Boolean before scheduling the conditional matrix.

Q8K4 is a fully reported secondary result: publish the same two CE comparisons and all three per-p PPLs, alongside every Q4K4 per-p comparison and selection count. Its result does not reverse or independently trigger the primary decision. This prevents choosing the preferred width after observing results. A heterogeneous result across p must be described, even if its aggregate passes. This rule concerns model quality only; it does not claim that fewer selected keys or lower PPL implies measured hardware efficiency.

Wait for all six valid MXINT8 full cells and their required matched TC/INT references before evaluating the rule. Missing, failed, incompatible or nonfinite cells yield `decision_unavailable`; investigate/repair the correctness or evidence issue. They are not assigned fabricated PPL or called poor quality. A validated higher PPL is a valid negative result and remains in reports. Do not repeatedly rerun valid cells to search for a favorable value.

## 2. Scope when the rule fires

If either primary condition fires, record `activate_mxfp4=true` and add all six conditional cells: two predictor shapes × three p values. Apply the same original full-code final execution, one Top-p, smoke, six actual traces/cell, independent replay and full scoring gates. This adds36 traces and six full runs; it does not replace the initial18 candidates. If neither fires, record `activate_mxfp4=false`; no MXFP4 model run is scheduled. The later MXFP4 results are conditional follow-up evidence; do not pool them as an unconditional original matrix or claim unbiased best-format selection.

Native E2M1 has four bits, so the Q8 variant needs an explicit mixed-format definition:

| Conditional ID pattern | Predictor Q | Predictor K | Final selected scores |
|---|---|---|---|
| `format_mxfp4_q4k4_topp_p{095,090,085}` | Native block32/E8M0 E2M1 sidecar4 | Native block32/E8M0 E2M1 sidecar4 | Original INT-Flash full Q8/K8 |
| `format_mxint8q_mxfp4k_q8k4_topp_p{095,090,085}` | Full block32/E8M0 MXINT8 sidecar8, same frozen recipe as initial MX Q8 | Native block32/E8M0 E2M1 sidecar4 | Original INT-Flash full Q8/K8 |

The second row is a mixed MX predictor, not an eight-bit E2M1 encoding and not original-Q8 × FP4K. It holds the initial MXINT8 Q8 predictor fixed while replacing K's TC high4 by native FP4. Report this distinction beside every result. Both start from the same original source DQ values and retain all shared-scale metadata in the bit budget. Native FP4 is not a contiguous high nibble of an INT8/FP8 word. No model weight training, final native-MX scores or other causal counterfactual is added.

The existing SPEC E2M1 values, block exponent rule e=floor(log2 A)−2 and nearest-even/saturation convention apply. Before implementation freeze the complete16-code lookup and midpoint ties from the primary specification: magnitude codes000..111 map to0,.5,1,1.5,2,3,4,6, sign in bit3; exact midpoints choose the endpoint whose magnitude encoding has low bit0. Canonical emitted numerical zero is positive; preserve any actually emitted code bytes in traces. Native FP4 alone remains4.25 logical bits/element with its E8M0 metadata; the mixed full-Q side remains8.25. No conditional run starts with an incomplete encoding/oracle definition.

## 3. Canonical MXINT8 FP32 arithmetic

Use eager operations with autocast disabled for this path. All floating operands below are explicit FP32 tensors. No BF16/FP16 matmul, TF32, fused format-adapter/selection kernel, compile rewrite, post-dot scale substitution or silent alternate backend is permitted. Freeze PyTorch/CUDA/cuBLAS/driver/device, matmul precision settings, operand layout/strides and query chunk64 in the execution manifest. Disable CUDA matmul TF32, set float32 matmul precision highest, and disable autocast around reconstruction, predictor matmul, normalization and selector. Record deterministic settings and the cuBLAS workspace policy before CUDA initialization; any required nondeterministic-kernel issue is investigated before freezing the executable implementation, not silently waived after results. A pinned CUDA GEMM reduction is not asserted to equal a serial CPU summation bitwise.

1. Form source `v = source_codes.to(float32) * source_scales.to(float32)` with a separate FP32 elementwise multiply, broadcasting the original token/head scale over D. Materialize the result before MX conversion. Do not use original pre-quantization floating Q/K.
2. For each contiguous valid block32 of a token/head, obtain finite A=max(abs(v)). For A>0 derive its exact binary exponent e with `frexp(A)` (e=returned_exponent−1) or verified IEEE fields, including subnormals. Avoid approximate log2. For A=0 use e=0 and codes0. Assert E8M0 byte=e+127 lies0..254; byte255 is invalid, no silent exponent clipping.
3. Convert with explicit FP32 power-of-two scaling `u = ldexp(v, 6−e)`, then `r = round_ties_even(u)`, then clamp r to[-127,127] in a wide representation, then cast to int8 c. This fixes the mathematical64*v/X expression as one binary scaling before rounding, avoiding intermediate overflow or an unrecorded alternate order. Record clamps, zero blocks and underflow. Preserve e/E8M0 and c exactly.
4. Extract `h = floor_divide(c.to(int16),16)`. Full sidecar8 operands are `ldexp(c.to(float32),e−6)`; retained high4 operands are `ldexp(h.to(float32),e−2)`, each materialized as its own FP32 tensor. Decode each block's own saved exponent. These are canonical binary-scaled forms of X*c/64 and X*h/4. Assert representable finite results for actual supported inputs; do not silently use FP64, clip overflow or flush values to make a run pass. Report genuine rounding/subnormal behavior.
5. For Q4K4 use prefix4 reconstructed Q and K. For Q8K4 use full sidecar8 Q and prefix4 K. Preserve layout[B,Hq,Q,D] and[B,Hq,N,D] after the inherited single GQA expansion. Materialize contiguous reconstructed operands. In the inherited query chunks, compute `dot = torch.matmul(qhat_chunk, khat.transpose(-1,-2))` with FP32 inputs/output and TF32 disabled. Accumulate over the full D in this call; do not replace it by block sums followed by a different reduction.
6. Create one FP32 scalar denominator by rounding binary64 `sqrt(D)` to FP32 (`sqrt_D_f32`). Compute `logits = torch.div(dot, sqrt_D_f32)` as a separate eager FP32 operation after GEMM, then apply the inherited legal-domain mask. Do not multiply reciprocal before the dot, omit a dyadic factor or apply original source scales again. Save the denominator value/bits and operation names. Apply the inherited stable FP32 softmax, descending-logit/ascending-position ties, FP32 cumsum and crossing-item selector exactly once.
7. Use S to gather original full source Q8/K8 and original scales; reuse the existing selected-full DQ-FP32 final backend, S-internal normalization, original V/PV/o_proj. Predictor sidecar values never become final operands.

For conditional FP4 replace only the respective operand reconstruction by FP32 `ldexp(decoded_E2M1_value,e_FP4)`, with e_FP4 from the frozen native-FP4 scale rule. Keep matmul/division/selection/final ordering identical. Mixed-Q8 uses the initial MXINT8 full-Q reconstruction above. Freeze those six configs before any conditional smoke/full runs.

## 4. Independent oracles and boundary evidence

The independent scalar reference must not import the production converter, reconstruction, Top-p or format helpers. Independently enumerate codes, E8M0 bytes, sign, clamp and RNE ties. Verify source255 integers across supported scales; zero/negative values; block boundaries; adjacent FP32 values around binary exponent transitions; midpoint neighbors; full8/prefix4 reconstructions; differing block/key scales; tails and masks. Compare discrete codes/prefixes/exponents and supported reconstructed FP32 operands exactly. Handle genuine subnormal platform differences explicitly rather than assuming a CPU/GPU bitwise guarantee; freeze supported behavior before accepting traces.

Keep three different checks explicit:

* CPU INT64/FP64 mathematical oracle verifies encoding, real-value/block-factor formulas and ideal selector semantics. Serial FP32 CPU arithmetic is an independent numerical cross-check with its recorded reduction order; it is not an exact CUDA GEMM emulator.
* Recompute predictor logits from independently decoded saved source/sidecar data with the pinned declared FP32 device primitives, without importing production helpers. Compare reconstruction bits exactly; require deterministic repeat execution on identical layouts and pinned backend, and apply the inherited frozen numerical tolerances to independently reduced logit/final-output comparisons. A loose logit tolerance cannot authorize a different selected set.
* Independently rerun stable FP32 softmax/cumsum/selection from saved **actual FP32 estimator logits** on the pinned device and declared key order. The actual-logit mask/indices/counts must match exactly. Also select using independently regenerated FP32 logits; if regeneration is not bitwise identical and membership differs near a tie or crossing, the cell remains unresolved until arithmetic/reduction is reproduced or a pre-result protocol revision is frozen. Do not replace the actual mask with an FP64 mask, add an epsilon, change p, or discard a hard row.

For every mismatch save actual and regenerated logits/probabilities, sorted key order, cumsum, crossing key, first divergent position, p, both selected sets and binary32 representations. Report logit-sort gaps and cumsum distance to p, including exact ties and equality crossing. FP64-vs-FP32 semantic differences are separately reported numerical evidence; exact FP32 execution replay remains mandatory. Preserve failures. Confirm trace-on/off model output/NLL identity before full runs.

## 5. Freeze and authorization boundary

Archive this note, final matrix/configs, exact arithmetic manifest, oracle code digest and parent adoption record before observing MX format full results. No format full result has been examined to choose this criterion. The parent must incorporate this decision and resolve any text saying MXFP4 needs a new discretionary decision: the latest user authorization already covers this specifically conditional MXFP4 follow-up, subject to correctness and release gates. Other-cause supplementary experiments still require their own concrete proposed SPEC and user approval; this note supplies no authorization for them.
