# Proposed causal supplement v1 — awaiting explicit user approval

Date: 2026-10-28. State: **planned; no implementation or execution authorized**.
This proposal is separate from the running format study and its conditionally
preauthorized MXFP4 matrix. Neither reading this proposal nor finishing that
matrix authorizes these CPU counterfactuals or GPU runs. Wait for completion of
the main format work and explicit user approval. Approval of phase A alone does
not approve phase B. Do not alter the frozen format executable or old reports.

## Decision and evidence boundary

Recommend phase A first: fixed-input CPU diagnostics on existing saved tensors,
with no model instantiation, followed by a new decision on phase B. Diagnose
whether TC's reconstruction offset explains most local selector error, whether
residual variation remains decisive, and whether rare concentrated failures
matter more than average retained mass. Keep model confirmation to p=.85, where
the already measured TC gap is largest; this is an evidence-informed engineering
stress point, not an unbiased estimate across thresholds.

Sources: [TC diagnosis](/output/project/experiments/stp-format-study/diagnosis/REPORT.md),
[distribution evidence](/output/project/experiments/stp-format-study/diagnosis-distributions/REPORT.md),
[format SPEC](/output/project/experiments/stp-format-study/spec/SPEC.md), and
[MX execution rule](/output/project/experiments/stp-format-study/spec/MX_EXECUTION_DECISION.md).
Main format results remain pending in [MAIN_RESULTS.md](/output/project/experiments/stp-format-study/supplement-proposal/MAIN_RESULTS.md).

## Fixed common semantics

Reuse original INT8 source codes x in [-127,127], original positive FP32
per-token/head scales, causal legality, GQA mapping, D=128, original V and
full selected-source DQ-FP32 execution. Do not quantize the model again with a
new source recipe. All means/metrics operate on legal keys only. A source or
provenance mismatch blocks reuse. Decode/prefill query positions retain their
original absolute key-position identifiers.

Top-p: one FP32 stable softmax; descending logit, ascending key position on an
exact tie; FP32 cumulative sum; shortest prefix with cumulative probability >=p;
include the crossing key and at least one key. An exact equality crosses. Final
weights use original full Q8/K8 scores only inside the selected set; no second
Top-p. Store indices ascending plus a boolean mask. p=.95/.90/.85 in phase A;
p=.85 only in phase B. No post-result threshold, epsilon or tie changes.

Each phase freezes code/config/assets/tokenizer/tokens/environment hashes before
execution. Phase B inherits BF16 model, FP32 attention, C2048/S1024, original
scoring_ranges, reset positions/cache per full window, retained tail, FP32 CE
and CPU float64 NLL accumulation. Verify all T-1 targets exactly once; historical
expected count is 289075, not a number to force. Full runs are serial, independent
processes, trace off. Same-input comparisons and cross-run trajectories are
reported separately. No train/test fitting or hyperparameter search is included.

## Phase A: exact offset/residual diagnosis on frozen traces

Input inventory: the existing 54 integer-source traces = three p values × three
source trajectories (INT, TC Q4K4, TC Q8K4) × six captures (layers 0/16/31 ×
query127 prefill/query128 decode). Each contains 32 heads: 1728 source/head rows.
Hash the exact files named by the existing diagnosis manifests; report missing
captures rather than synthesizing them. The 192 INT-source rows per p are the
primary fixed-input cohort. Other trajectories are replication/sensitivity
cohorts, not independent samples or a calibration split.

For a scalar code define H(x)=floor(x/16), T(x)=16H(x), M(x)=16H(x)+7.5,
R(x)=x-M(x). H ranges [-8,7], T [-128,112], M [-120.5,119.5], R [-7.5,7.5].
This midpoint is fixed from the 16-code prefix interval, without empirical fits.
It deliberately maps x=0 to 7.5 and does not special-case zero or endpoint bins.
It is not sign-magnitude or a source recoding. Original x and scale remain
unchanged for the final execution.

Represent M exactly by doubled integer G=32H+15 in [-241,239], T by G=32H,
and full x by G=2x. Accumulate GQ·GK exactly in INT64 for the mathematical CPU
oracle. Candidate execution converts the exact dot to FP32, then applies
`(((dot * 0.25) * sQ) * sK) / sqrt_D_f32` with separate FP32 operations and
sqrt_D_f32=FP32(sqrt(128)). Disable autocast/TF32 and fusion. This fixes scale,
temperature and reduction semantics; ties are based on the resulting logits.
The doubled code is an arithmetic representation, not a claimed 9-bit stored
format. Max doubled midpoint dot is 128*241^2=7434368 < 2^24; the
TC doubled bound is 128*256^2=8388608 < 2^24. Residual-subset sums
remain INT64 before FP32 conversion; record any conversion rounding. Validate TC/full
controls against their inherited ordered arithmetic and save any rounding
boundary differences; mathematical equality alone cannot waive different masks.

Use the following **12 unique logit definitions** per source/head:

1. TC: TQ·TK, and established Q8K4: Q·TK (2 controls).
2. MQ·TK and TQ·MK (2 offset-side controls).
3. All eight subsets of terms A, B, C added to MQ·MK, where
   A=RQ·MK, B=MQ·RK, C=RQ·RK. The empty subset is midpoint Q4K4;
   the complete subset is the exact full INT dot (8 definitions).

Every dot includes the original sQ*sK/sqrt(D) factor for each key. Compute term
products exactly using doubled integers, with common factor 1/4; accumulate the
chosen integer sum **before** the declared FP32 scaling. Do not add separately
rounded FP32 term logits. Check the full-subset dot equals 4*(Q·K) exactly before
scaling. Oracle subsets require discarded residuals and are **nondeployable
causal diagnostics**. They are not new four-bit formats or latency claims.
Report covariance and all subset effects; single-term restoration can worsen
error by removing cancellation, so do not assign additive percentages of loss.

Add two selector diagnostics using existing TC logits: (i) TC Top-p with minimum
four legal keys, by extending its sorted prefix to min(N,max(4,n_TC)); (ii) the
same number of keys as diagnostic (i), n_floor4=min(N,max(4,n_TC)), chosen
by descending full INT logits with identical tie rules. The latter is a nondeployable count-matched oracle. It diagnoses whether
ranking can matter at exactly the same key count. This makes **14 definitions,
24192 head/definition metric rows**, before summaries; report term tensors
separately. Both selector diagnostics inherit TC logits: centered RMSE, entropy
and rank statistics repeat TC values; only selected-set/output metrics change.
These counts are an inventory, not independent trials.

Primary endpoint: mean retained-full-INT-mass shortfall max(0,p-mass), equally
weighted over the 192 INT-source head rows at each p. Co-primary diagnostic:
equal-weight mean of per-head centered logit RMSE over those same 192 rows,
with error row mean removed before each head RMS (not a pooled RMS). Secondary: retained
mass 1/5/50th percentiles and minimum, count below p and p-.05, argmax misses,
Spearman with average ranks for ties, selected count/fraction, PV absolute and
relative L2 versus same-input full-INT Top-p. If reference PV norm <=1e-12,
relative L2 is undefined; retain absolute L2 and count those cases. Report every
head and per-layer/query summaries, not only selected examples.

Predeclare concentration strata using **full INT reference only**: max probability
>=.5 versus <.5; also report reference effective support exp(entropy) <=4 versus
>4. Report empty strata as empty. Capture TC-to-reference entropy difference,
argmax membership, boundary logit gap and cumulative distance to p. Report exact
ties. Strata support diagnosis; they are oracle labels, not a proposed runtime
gate. No worst-head mining is allowed to choose phase B locations.

Phase A decision: midpoint supports offset reduction only if its centered RMSE
and mean shortfall both improve over TC on INT-source rows at all three p.
Failure of either condition rejects that combined claim; publish mixed outcomes.
Residual terms remain relevant if restoring at least one subset improves either
endpoint beyond midpoint. Report signed changes without a significance label or
"fraction of PPL explained." No numerical superiority gate discards a valid
negative result. Phase A does not automatically schedule phase B.

## Phase B: bounded model confirmation, separate explicit approval required

Proposal: five new cells at p=.85 only, each based on the original TC Q4K4
predictor. Existing compatible TC, Q8K4 and INT-full controls are reused after
hash/config/coverage checks. If reuse fails, request approval for the precisely
identified missing control runs; do not silently expand this five-cell budget.

| Cell/branch suffix | Exact intervention | Purpose |
|---|---|---|
| `midpoint-both-p085` | MQ·MK predictor everywhere, arithmetic above | Does fixed offset correction survive the full trajectory? |
| `tc-floor4-p085` | TC selector with min(4,N) selected keys everywhere | Does preventing tiny selected sets help? |
| `tc-budgetshift-p085` | Same extra-key budget as floor4, redistributed as below | Is a floor4 effect attributable only to added work? |
| `tc-intselector-l0-p085` | Original full-INT Top-p selector in layer0, TC elsewhere | Fixed early selector intervention and upstream propagation |
| `tc-intselector-l16-p085` | Same intervention in layer16, TC elsewhere | Equal one-layer structural control; distinct layer sensitivity |

Budgetshift is defined per batch item, layer, query on that run's current inputs.
Compute TC counts n_h for all 32 query heads and B=sum_h max(0,min(4,N)-n_h).
Order heads by cyclic sequence 16,17,...,31,0,...,15; repeatedly visit heads with
n_h>=min(4,N), adding the next TC-ranked unselected key until exactly B keys are
added. Skip a full head; preserve its original TC set. Before allocation, compute eligible capacity C=sum_h (N-n_h) over heads
with n_h>=min(4,N). If C<B, perform a deterministic whole-row no-op: retain
the original TC selected sets for all heads, add zero keys, and record B as
unspent. Do not partially allocate or change recipient eligibility. The selected
heads/keys must be legal. B=0 leaves TC unchanged and is recorded separately
from an infeasible C<B no-op. Report counts/fractions of both cases, B, C,
actual added keys and unspent budget per row and in summaries. This uses only
TC counts and logits, never oracle scores. On feasible rows it adds exactly
floor4's extra budget **on identical current inputs**; on infeasible rows it
preserves the baseline and does not supply a matched-budget intervention. Independent trajectories can have different B and
total counts; report that limitation and actual counts. Do not claim exact
cross-run cost matching. Interpret the floor4/budgetshift contrast conditional
on feasible same-input rows; whole-run differences also include the no-op
policy and cannot isolate targeting alone. Phase A can evaluate this rule locally as an optional
separately counted extension only if included in the approved manifest; it is
not included in the 24192-row CPU budget above.

Layer0/16 are fixed from the capture protocol, not selected by observed recovery.
These oracle interventions compute full logits for one layer and are not
hardware candidates. Both cover every head/query in that layer. They differ in
layer position and actual work, so "equal one layer" does not mean equal measured
latency. Local estimator errors on INT inputs establish a direct mechanism;
changes in later operands/outputs document propagation. Neither their NLL gains
nor their difference uniquely assigns total TC loss to that layer or separates
all nonlinear interactions.

Each approved new cell: one 4096-token/4095-target smoke process with six captures,
cache-off/on checks and independent replay, then one full process after checks.
**Maximum five smoke + five full processes, 30 trace records, five new full
corpus evaluations.** Cache checks are subchecks, not extra corpus sweeps; record
all invocations. No seed/p sweep, no repeat of a valid quality result. Failed
correctness checks stop that cell before full execution. No new candidate is
chosen from test PPL. Primary model outcome: signed CE difference from compatible
TC at p=.85 from unrounded NLL/targets; report PPL ratio and deltas against INT
and Q8K4 too. All five outcomes, including negative results, must be retained.

## Storage, arithmetic cost and performance interpretation

Midpoint retains the same four-bit prefix and original source scales; constant
7.5 needs no per-vector learned metadata. Expanding the midpoint dot yields
256 HQ·HK +120 sum(HQ)+120 sum(HK)+56.25D: two coordinate sums plus corrections,
or direct doubled-operand arithmetic with wider intermediates. At D128,Q/K
prefix storage is logically64 packed bytes each, plus inherited scales; original
full codes/scales remain necessary for final execution. Prototype doubled values
need int16/FP32 buffers; report actual bytes, conversion/sum work and caching.
No native 4-bit integer-MAC, speed, area or power claim follows from CPU/GPU
functional simulation. Oracle A/B/C access full residuals. Layer interventions
use full QK work. Floor4 adds selected final-score/PV work; budgetshift adds
head allocation work. Report prefill/decode separately, selected key counts,
logical bits, physical allocated bytes and full predictor/final MAC accounting.
Measured runtime requires its own pinned benchmark scope, outside this proposal.

## Isolation, verification, stopping and artifacts

Do not run from or edit `Twilight-STP_test_Format` while its study runs. Proposed
future isolated checkout: `/output/project/Twilight-STP_test_Supplement_v1`.
Proposed branches: `stp/supplement-v1-cpu-diagnostics` then
`stp/supplement-v1-model-confirmation`; suffixes in the table are config IDs,
not five divergent source branches. Create branches/checkouts only after the
corresponding approval and freeze their parent revision and patch. Do not merge
or rewrite the format branch. Initial approval record must name exact phases,
cell IDs, counts and any changed text; archive this v1 before revisions.

Future execution root:
`/output/project/experiments/stp-format-study/supplement-execution-v1/` with
`approval/`, `freeze/`, `phase-a/{manifest.json,head_metrics.csv,term_covariance.csv,
summary.csv,concentration.csv,verification.json,REPORT.md}`, and
`phase-b/<cell>/<run-id>/{config.json,command.txt,run_manifest.json,stdout.log,
coverage.json,window_nll.csv,metrics.json,traces/,replay/,failures/}`.
Top-level `artifact-index.json` hashes all evidence; `REPORT.md` links actual
outcomes and explains inherited/missing scope. These paths are proposed; do not
create execution evidence before approval. This proposal directory contains
preparation only and does not alter historical evidence or PADE history indexes.

Before model execution require independent scalar encode/reconstruction and
selector oracles, exhaustive source-code range tests, negative/zero/endpoint
cases, ties and threshold equality, integer bounds, same-input gauge checks,
original final-operand hashes, actual-logit mask replay, regenerated-logit
boundary diagnosis, cache equivalence and trace-on/off identity. Inherit the
frozen backend's exact tolerances and record their values in the freeze; never
choose tolerances after results. Any selected-set mismatch is unresolved even
when logits are close. The full-subset equality check is exact in integer space;
FP32 reference-path differences are saved and diagnosed separately.

Stop affected cells on arithmetic, provenance, selector, coverage, nonfinite or
replay failures; preserve partial/failed logs. Hardware/resource failures allow
one identical retry with both attempts retained; persistent failure requires a
revised proposal. Valid bad quality does not trigger retry. No opportunistic
extra p/layer/head/remedy cells. Main format results may motivate a written v2
before approval, but cannot silently change this frozen v1 after execution.

## Independent review clarification, 2026-10-28

Preparation-only corrections: count-matched oracle count, selector-control logit
metrics, per-head RMSE aggregation, and total budgetshift rule with explicit
infeasible-row no-op accounting. Initial proposal preserved in
[history/v1-before-independent-review/](/output/project/experiments/stp-format-study/supplement-proposal/history/v1-before-independent-review).
No execution or approval follows from these corrections.
