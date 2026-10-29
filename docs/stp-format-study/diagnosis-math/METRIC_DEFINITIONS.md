# Rank metric definitions: historical `spearman` and current stable ranks

Date: 2026-10-28. Status: implementation inspection only. The naming concern is valid: historical `spearman` is Pearson correlation of **stable ordinal ranks**, which generally differs from canonical Spearman correlation with average ranks for ties. No metrics were recomputed, and no existing report, source, index, replay result, or model execution was changed.

## What the implementations measure

The historical [analyze_tc.py, lines 14–18](/output/project/experiments/stp-format-study/diagnosis/analyze_tc.py) constructs ranks with:

```python
rank = torch.argsort(torch.argsort(logits, stable=True), stable=True)
```

Both sorts use the default ascending direction. For a logit vector x indexed in key-array order, the zero-based ordinal rank of index i is

\[
R_x(i)=\#\{j:x_j<x_i\}+\#\{j<i:x_j=x_i\}.
\]

Tied logits therefore receive distinct consecutive ranks according to their original array order. The script stores the centered Pearson correlation of these ranks under the field name `spearman`:

\[
\rho_{\mathrm{ordinal}}
=\frac{\sum_i(R_x(i)-\bar R_x)(R_y(i)-\bar R_y)}
 {\sqrt{\sum_i(R_x(i)-\bar R_x)^2\sum_i(R_y(i)-\bar R_y)^2}}.
\]

The current frozen [diagnostics.py, `logit_statistics`](/output/project/experiments/stp-format-study/execution-v1/freeze/source/benchmark/attention_format/diagnostics.py) uses the same stable ordinal construction, converts ranks to FP64, and labels the result `stable_rank_correlation`. It filters to legal keys before ranking. Its denominator is clamped below at 1e-30; the legacy function uses FP32 ranks and has no denominator clamp. The frozen [diagnose_tc.py](/output/project/experiments/stp-format-study/execution-v1/freeze/source/benchmark/attention_format/diagnose_tc.py) likewise uses stable ordinal ranks and the explicit `stable_rank_correlation` name. These naming and numerical details do not make the current diagnostic a newly defined average-tie Spearman metric.

Canonical Spearman uses the Pearson correlation of **midranks**: every member of a tied group gets that group's average rank. With no ties, midranks and ordinal ranks coincide, so these definitions agree apart from numerical arithmetic. With ties, they need not agree; neither the direction nor the size of the difference was measured here. The distinction matters for coarsely quantized predictors, whose exact logit ties are documented. Even an entirely constant vector receives distinct ordinal ranks from array order; an average-tie rank vector is constant instead. Stable ordinal correlation therefore measures agreement of deterministic total orders, including their tie-breaking convention, rather than solely agreement of the untied logit information.

## Array order, key-position order, and selection are distinct details

Stable sorting preserves **input array order** among equal values; the rank-statistics functions do not separately sort physical key positions. For the format study, frozen [replay.py, lines 67–68](/output/project/experiments/stp-format-study/execution-v1/freeze/source/benchmark/attention_format/replay.py) explicitly requires `key_positions == torch.arange(query_position + 1)`. Thus the accepted captures' rank ties are resolved by ascending key position. Legal-key filtering preserves that order. A future tensor with unsorted physical key positions would require a separate ordering contract; stability alone would not imply ascending position.

The frozen independent [oracle.py, `top_p`](/output/project/experiments/stp-format-study/execution-v1/freeze/source/benchmark/attention_format/oracle.py) explicitly sorts positions ascending, then stably sorts logits **descending**. Top-p ties therefore use ascending key position too. This selector ordering and crossing-key rule are separately validated by exact actual-mask replay. The ordinal correlation is a descriptive diagnostic; it does not choose the selected set or determine replay membership. The legacy Top-p helper also uses a stable descending-logit sort over its supplied key-array order.

## Scope and effect on conclusions

Historical `head_metrics.csv` and report tables labelled “Spearman” should be read as this stable ordinal-rank correlation. The [earlier mathematical note](NOTE.md)'s references to those reported rank statistics inherit the same definition. [SMOKE_SELECTION_BUDGET.md](SMOKE_SELECTION_BUDGET.md) uses the current saved `stable_rank_correlation` field. This clarification preserves the recorded numbers and avoids silently replacing a closed historical metric.

The legacy rank statistic compares each established estimator with a full-INT predictor on the **same saved source trajectory**. Current replay similarly compares the actual candidate predictor with its own original-full-INT logits on that candidate's saved operands. Neither definition makes independently evolved model trajectories identical. Comparing aggregate rank statistics across SM/MX cells remains descriptive; the separately verified layer0 common-input subset supplies the narrower actual common-input contrast. Cross-run set Jaccard remains a distinct metric and includes trajectory effects.

This correction affects metric interpretation, not measured PPL/NLL, selected-key counts, retained probability, PV error, or exact replay membership. The omitted-term algebra, captured key misses, and selected-only recomputation argument do not depend on using average-tie Spearman. It does limit treating the historical column as a canonical tie-aware Spearman coefficient or as proof that ranks themselves contain more predictive information. No average-tie replacement, revised correlation values, or new quality conclusion is claimed.

## Source identity

[metric_definition_sources.json](metric_definition_sources.json) records SHA256 hashes of all inspected source files and this note. The historical script digest is `da990aaf0ab77f877bdc5732409f8bf7288c8105c9540f25ef747fea3c7526c7`, matching the existing [diagnosis manifest](/output/project/experiments/stp-format-study/diagnosis/manifest.json). The frozen current diagnostic digest is `2426b6c4f2a80f099d9764c2067205c84e1d5d7d9274670e9369012e16ced703`; the inspected live-checkout `diagnostics.py` has the same digest. Frozen source snapshots, rather than a potentially changing checkout, anchor the current formula described here.
