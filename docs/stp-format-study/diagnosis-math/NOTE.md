# Why TC MSB Q4K4 can lose quality at the same Top-p threshold

Date: 2026-10-28. Status: analytical derivation using existing evidence only. No model execution, tensor recomputation, numerical experiment, fitting, estimator intervention, or remedy evaluation was performed for this note. The running format study, its frozen source/specification/execution, and the completed diagnosis directories are unchanged.

Evidence: [main diagnosis](/output/project/experiments/stp-format-study/diagnosis/REPORT.md), [source distributions](/output/project/experiments/stp-format-study/diagnosis-distributions/REPORT.md), and [captured worst head](/output/project/experiments/stp-format-study/diagnosis/worst_head_example.json). Format definitions are in the existing [SPEC, §§4–5](/output/project/experiments/stp-format-study/spec/SPEC.md). This note does not assert results from the running SM/MX study.

## 1. Same threshold does not mean the same retained original probability

Fix one head/query and its legal key domain. Let real original full-Q8/K8 logits be

\[
z_j=\langle q,k_j\rangle/\sqrt D,\qquad
P_j=\operatorname{softmax}(z)_j.
\]

Here q and k include the ORIGINAL source dequantization scales; D=128 in the captured study. Let predictor logits be \(\hat z\), predictor probabilities \(Q=\operatorname{softmax}(\hat z)\), and S its nonempty Top-p set at threshold \(\tau\), including the crossing key. Thus \(Q(S)\ge\tau\). P is the full original-INT distribution, not an unquantized floating reference. All statements below concern exact real arithmetic on this fixed common domain; independently validated implementation replay supplies the separate finite-precision execution evidence.

**Probability discrepancy bound.** With \(\operatorname{TV}(P,Q)=\frac12\sum_j|P_j-Q_j|\),

\[
P(S)\ge Q(S)-\operatorname{TV}(P,Q)
       \ge\tau-\operatorname{TV}(P,Q).
\]

Proof: for every set, its probability difference has absolute value at most TV; apply this to S. The useful lower bound is its maximum with zero. This requires no independence assumptions and no special property of Top-p beyond \(Q(S)\ge\tau\). In particular, Q's threshold alone provides no positive universal lower bound on P(S): Q can place almost all mass on a key whose P mass approaches zero.

**Logit discrepancy bound, modulo an additive constant.** Suppose for some c,

\[
\max_j|\hat z_j-z_j-c|\le\epsilon.
\]

Write \(\delta_j=\hat z_j-z_j-c\). Then

\[
\frac{Q_j}{P_j}
 =\frac{e^{\delta_j}}{\sum_iP_i e^{\delta_i}}
 \in[e^{-2\epsilon},e^{2\epsilon}],
\qquad
P(S)\ge e^{-2\epsilon}Q(S)\ge\tau e^{-2\epsilon}.
\]

Both numerator and denominator lie between \(e^{-\epsilon}\) and \(e^\epsilon\), proving the bound. The best such \(\epsilon\) is half the range of the logit errors; c is their midrange. A constant bias gives \(\epsilon=0\) and identical softmax probabilities. If \(z_i-z_j>2\epsilon\), then \(\hat z_i>\hat z_j\): strict rank margins greater than twice this error survive. Preserving ranks alone does not preserve the Top-p crossing position, because cumulative probabilities can change.

These are conditional guarantees, not numerical certificates from the current summary tables. Mean centered RMSE .69606 is neither a per-head maximum error nor an \(\epsilon\) for every head. Even for one head with n keys, the elementary implication \(\max_j|e_j-\bar e|\le\sqrt n\,\operatorname{RMSE}(e-\bar e)\) can be loose. An average over heads cannot replace this head-specific premise.

## 2. Full-code recomputation cannot restore an excluded key

For fixed original full logits and values v, define the dense original-INT head output \(A=\sum_jP_jv_j\), retained mass \(m=P(S)>0\), and selected output

\[
A_S=\sum_{j\in S}\frac{P_j}{m}v_j.
\]

This is exactly what full original-Q8/K8 recomputation and normalization inside S mean mathematically. Predictor probabilities decide membership but are absent from the final weights. If \(m<1\), let \(A_{\bar S}\) be the similarly normalized outside output. Then

\[
A=mA_S+(1-m)A_{\bar S},\qquad
A-A_S=(1-m)(A_{\bar S}-A_S).
\]

For any fixed norm, let \(\Delta_V=\max_{i,j}\|v_i-v_j\|\), the diameter of the finite value set and its convex hull. Both conditional outputs lie in that hull, so

\[
\|A-A_S\|\le(1-m)\Delta_V.
\]

For \(m=1\), the difference is zero without defining the outside average. Thus the bounds in §1 also imply upper bounds on dense-head error by replacing \(1-m\) with \(\min(1,1-\tau+\operatorname{TV}(P,Q))\) or \(1-\tau e^{-2\epsilon}\). These are upper bounds, not lower bounds: identical values can make the output exact even when mass is missing.

**The reported PV reference is different.** The diagnosis reports error against original INT **Top-p** output \(A_T\), where T is selected using P at the same threshold, not against dense A. For arbitrary nonempty S and T, with \(m=P(S)\), \(t=P(T)\),

\[
\operatorname{TV}\big(P(\cdot\mid S),P(\cdot\mid T)\big)
=1-\frac{P(S\cap T)}{\max(m,t)},
\]

because the sum of the overlapping minimum probabilities is \(P(S\cap T)/\max(m,t)\). Hence

\[
\|A_S-A_T\|
\le\Delta_V\left[1-\frac{P(S\cap T)}{\max(m,t)}\right]
\le\Delta_V\big[(1-m)+(1-t)\big].
\]

The first inequality follows by decomposing the signed probability difference into equal positive/negative mass TV and taking the difference of two convex combinations of values. The second can equivalently be obtained by applying the dense-output bound twice and the triangle inequality. It remains valid when the displayed right side exceeds \(\Delta_V\). Equal retained mass does not imply equal selected output. Relative output error additionally divides by \(\|A_T\|\), which can be small through value cancellation; the reported relative errors are not missing-mass fractions.

## 3. TC prefixing changes relative logits through coupled residual terms

For a source integer x, TC high4 reconstruction is \(16\lfloor x/16\rfloor\), with discarded residual \(\ell=x-16\lfloor x/16\rfloor\in\{0,\ldots,15\}\). In real original units write

\[
\hat q=q-r_q,\quad \hat k_j=k_j-r_{k_j},\qquad
r_q=s_q\ell_q,\quad r_{k_j}=s_{k_j}\ell_{k_j}.
\]

Positive scales make every coordinate residual nonnegative, including for negative source codes. The exact real-logit identity is

\[
\hat z_j-z_j
=\frac{-\langle r_q,k_j\rangle
       -\langle q,r_{k_j}\rangle
       +\langle r_q,r_{k_j}\rangle}{\sqrt D}.
\]

Equivalently, with integer high nibbles H and residuals L,

\[
z_j-\hat z_j
=\frac{s_qs_{k_j}}{\sqrt D}
 [16\langle H_q,L_{k_j}\rangle
 +16\langle L_q,H_{k_j}\rangle
 +\langle L_q,L_{k_j}\rangle].
\]

These are real-arithmetic identities; the established diagnosis separately checked the decomposition against saved operands within its stated numerical tolerance. All terms retain the actual per-head/token scales and attention factor \(1/\sqrt D\).

Floor truncation shifts both signs downward: -1 becomes -16, +1 becomes 0. This is not symmetric rounding. Nonnegative operand residuals do **not** imply a fixed sign of each logit error: q and k are signed, and cross terms cancel. Nor may their RMS values be added as if independent. The existing p=.95 decomposition reports centered RMS .5040, .5848, .3224 for its three missing-score terms but .6961 for their sum, with observed correlations. Their means and correlations are evidence, not a license to assume iid/uniform residuals.

Writing \(\ell=7.5+\eta\) is an exact algebraic split, not a claim that \(E[\eta\mid H]=0\), that residuals are independent of q/k, or that 7.5 is an optimal correction. Query residuals can couple to key-dependent coordinate sums; key scales also vary. Consequently a common-looking offset term need not be constant across keys. Only an actually constant row offset cancels under softmax.

Q8K4 sets \(r_q=0\), giving \(\hat z_j-z_j=-\langle q,r_{k_j}\rangle/\sqrt D\). It removes query-prefix distortion while retaining key-prefix distortion. This algebra alone does not prove a smaller error in every head: cancellation in Q4K4 can occasionally help. The observed improvement is supplied by the controls.

**SM and MX scope.** SM4 is \(s\,\operatorname{sign}(x)16\lfloor|x|/16\rfloor\). It is an odd map that shrinks magnitudes toward zero, with signed residual \(s\,\operatorname{sign}(x)(|x|\bmod16)\), unlike TC's one-sided residual. Oddness does not prove zero mean on the actual distribution, unbiased dot products, superior ranks, or better PPL. MXINT8-TC4 still applies a TC floor after block requantization: it changes scales, rounded/clamped codes, and carries E8M0 shared metadata (4.25 or 8.25 logical bits per element with block32 scales). Its full-Q predictor is full sidecar8, not original Q8. Any measured MX change therefore combines representation, requantization, and metadata effects; it cannot isolate a four-bit encoding alone. All these predictors still use original full Q8/K8 for final selected execution in the main study.

## 4. What the existing observations establish

At p=.95 on Q4K4's own fixed inputs, the diagnosis reports mean retained original mass .96099 but **59/192 heads below .95**, four below .90, and two missing the original argmax. The mean selected count is 41.4688 versus 27.9583 for original INT Top-p. Extra keys and a favorable average mass therefore do not certify every head.

The captured layer0/query128/head2 selects only key128: predictor mass .957162, original mass **.380937**. Key127, the original argmax with mass .404557, is excluded, as is key126 with mass .143139. Recomputing original full logits on this singleton necessarily produces \(A_S=v_{128}\). Missing original mass is approximately .619063; this is neither the measured relative PV error (1.253888) nor a lower bound on it. The example concretely demonstrates the distinction between \(Q(S)\ge.95\) and \(P(S)\ge.95\), and the impossibility of restoring absent keys through selected-only recomputation.

The same-input p=.95 Q4K4 versus Q8K4 comparison gives centered logit RMSE .69606 versus .38485 and relative PV error .11015 versus .06222, using the main diagnosis's common Q4 trajectory. The distribution report's Q8 aggregate uses a different grouping/trajectory scope; it is not a replacement for this matched-input comparison. Q4K4's raw mean logit bias +2.3296 must not be confused with its centered relative-logit error. The .6961 centered error and observed rank changes supply the relevant local mechanism.

Existing full-window PPL gaps versus same-threshold INT increase from .351786 at p=.95 to .784881 at p=.90 and 1.306613 at p=.85. The smaller-threshold behavior is an observed trend, not a theorem of Top-p or this error algebra. The local theorems fix q/k/v; independent model trajectories alter downstream q/k/v, and corpus NLL depends on later nonlinear transformations and target logits. These bounds do not assign a PPL contribution to an individual term/head or prove that a particular format/remedy will recover it.

## 5. Scalar information loss is real, but is not a PPL lower bound

Within each reported head/operand/trajectory stratum, treat X as the empirical scalar source-code random variable and \(H=\lfloor X/16\rfloor\). Since H is deterministic,

\[
\mathcal H(X\mid H)=\mathcal H(X)-\mathcal H(H),\qquad
\inf_g E[(X-g(H))^2]=E[\operatorname{Var}(X\mid H)].
\]

The first is the entropy chain rule. The second follows from conditioning: squared error equals conditional variance plus \(E[(E[X\mid H]-g(H))^2]\), minimized by the conditional mean. Conditional variance is also that of the residual within a fixed prefix. The distribution report's equally weighted stratum means give approximately 3.22191 discarded conditional bits for Q and 3.92674 for K, by subtraction of its rounded entropy values; its measured conditional-variance means are 18.11980 and 20.21034 code-units squared. This demonstrates many-to-one scalar loss and the inability of a scalar-prefix-only decoder to recover every source value.

These statements condition only on the scalar prefix within those strata. They do not condition on the entire prefix vector, coordinate/key position, known scales, other predictor metadata, or learned correlations. They give neither a lower bound for every full predictor using such side information nor a lower bound on logit error, selected-output error, or model PPL. Discarded scalar information can be irrelevant or cancel in the downstream computation.
