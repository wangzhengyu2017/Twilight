# Why these supplements, in this order

State: proposed preparation only. The existing traces were read through their
reports; no new tensor program, estimator replay, model run or GPU work was
executed. [SPEC.md](/output/project/experiments/stp-format-study/supplement-proposal/SPEC.md) defines all proposed calculations and gates.

The first useful question is whether offset correction changes the selector
error materially on fixed inputs. Existing evidence already rules out "too few
keys on average" as a sufficient explanation: at p=.85 TC selects .116829 of
keys versus .059787 for INT, yet PPL is 7.921542 versus 6.614929. Additional
selected work can coexist with bad membership. This does not prove a minimum
set floor will help, so floor4 is a deliberately small, fixed test rather than
a presumed remedy.

TC's negative-floor error has a mean near 7.5 source-code units in both operands.
The discarded conditional variance is also substantial: E[Var(residual|prefix)]
is 18.11980 for Q and 20.21034 for K. Therefore a constant-offset explanation
cannot simply replace a residual-information explanation. Midpoint reconstruction
changes only the fixed within-bin offset; Q-only/K-only midpoint controls locate
the asymmetric contribution without choosing data-dependent means. Eight
residual-term subsets are inexpensive once approved and necessary to expose
cancellation. They are an exact algebraic diagnostic, with discarded information
available to an oracle. A successful midpoint is implementable in principle;
a successful residual oracle is not evidence of a cheap deployable predictor.

A large raw mean logit bias is weak evidence because softmax discards constant
row shifts. Centered error and retained-mass shortfall are primary for that
reason. Main TC diagnosis reports .6961 centered RMSE despite +2.3296 raw mean
error at p=.95. Equally, average retained mass can exceed p while individual
rows lose most of the useful distribution. The captured minimum retained mass
falls to .1805 at p=.85. Fixed concentration strata and tail summaries address
that failure without selecting a favorable subset after looking at new results.
The count-matched full-INT ranking oracle is especially useful: it shows whether
better membership is possible at the exact same count on each saved input.

Upstream trajectory change is a second mechanism, not a reason to dismiss the
local one. Existing INT-source traces already exhibit TC selector error; actual
Q/K disagreement then grows by layers16/31. Same-input replay identifies local
error; fixed one-layer full-selector interventions test how a changed selector
propagates. Layer0 and layer16 are preselected structural controls. They do not
form a clean additive decomposition of corpus NLL, since nonlinear downstream
states, V directions, layer roles and interactions differ. Whole-model CE must
be measured before any PPL-recovery claim. The five proposed cells keep this
confirmation bounded; phase A approval does not authorize them.

The budgetshift control spends floor4's extra budget on other eligible heads
using a fixed cyclic order. On feasible rows, it compares a targeted tiny-set intervention
with a generic extra-key intervention locally. If eligible capacity cannot
spend the entire extra budget, the whole row preserves its TC baseline and
records all extra budget as unspent; no partial allocation or fabricated
intervention is permitted. Report this no-op frequency separately. Exact
budget equality holds only on feasible matched current inputs, because separate
model runs develop different counts. Whole-run contrasts also include the
no-op policy and cannot isolate targeting alone.
The SPEC requires actual work reporting and forbids claiming whole-run cost
matching. If this limitation is unacceptable for the intended claim, omit both
tail model cells in a written v2 and retain the exact-count fixed-input oracle;
do not substitute an elaborate corpus-tuned gating policy.

The main format study may itself settle the practical representation choice.
SM can change asymmetric truncation while still losing low information; MXINT8
adds block rescaling and requantization. Their interventions are broader than
midpoint, so results cannot alone assign recovery to offset versus variance.
Wait for [MAIN_RESULTS.md](/output/project/experiments/stp-format-study/supplement-proposal/MAIN_RESULTS.md) to be populated, then decide whether
phase B remains worth its cost. Its MXFP4 conditional rule remains independently
binding and must not be replaced by this recommendation.

No evidence here supports corpus contamination, fitting leakage or an evaluation
bug as the decisive current cause. The relevant risk is future analysis leakage:
choosing gates, heads, p values or candidate formulas on the same test PPL later
used to report recovery. This proposal avoids fitting and labels p=.85 as an
already informed stress test. If later work fits residual means or head gates,
it needs disjoint calibration data and a new held-out evaluation protocol;
that work is outside v1. Correlated captured heads are not independent corpus
samples, and no statistical-significance claim is planned.

Recommended approval sequence: complete and verify the main format matrix;
fill its evidence placeholder; approve only phase A if the mechanism remains
unclear; review every local result; then explicitly approve or decline the
five-cell phase B. No phase has been implemented or executed by this document.
