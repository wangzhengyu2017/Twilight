# Format study final report

Llama 3.1-8B-Instruct, BF16 model, frozen WikiText-2 scoring: 289,075 targets per accepted full run. All format predictors select once with Top-p; final selected scores gather the original INT-Flash full Q8/K8 and scales. The predictor comparison does not measure a native INT kernel speedup.

| Method | p=.95 PPL | p=.90 PPL | p=.85 PPL | Evidence |
| --- | --- | --- | --- | --- |
| Dense FP | 6.484759140305276 | same dense reference | same dense reference | historical measurement; no parser-matched full gate |
| Dense INT | 6.486693431357931 | same dense reference | same dense reference | historical measurement; no parser-matched full gate |
| FP Top-p | 6.482529 | 6.530140 | 6.613165 | full corpus |
| INT Top-p | 6.485039 | 6.532782 | 6.614929 | full corpus |
| TC Q4K4 | 6.836825 | 7.317663 | 7.921542 | full corpus |
| TC Q8K4 | 6.500290 | 6.563772 | 6.667327 | full corpus |
| SM Q4K4 | 6.497612 | 6.557247 | 6.675108 | full corpus |
| SM Q8K4 | 6.498413 | 6.574305 | 6.697492 | full corpus |
| MXINT8-TC4 Q4K4 | 6.673347 | 6.921718 | 7.238384 | full corpus |
| MXINT8-TC4 Q8K4 | 6.503503 | 6.571011 | 6.675618 | full corpus |
| RS Q4K4 | planned_source_blocked; unverified | planned_source_blocked; unverified | planned_source_blocked; unverified | see cell status |
| RS Q8K4 | planned_source_blocked; unverified | planned_source_blocked; unverified | planned_source_blocked; unverified | see cell status |
| MXFP4 Q4K4 | 6.492194 | 6.544645 | 6.638278 | full corpus |
| MXINT8-Q8 / MXFP4-K4 | 6.484025 | 6.530161 | 6.603273 | full corpus |

Dense references are historical measured values; their absence of a parser-matched release gate must not be confused with verified format evidence. Missing, blocked and unverified format cells have no reported numeric quality result.

The frozen conditional decision is:

```json
{
  "activate_mxfp4": true,
  "primary": "q4k4",
  "status": "complete",
  "widths": {
    "q4k4": {
      "above_int_margin": true,
      "delta_ce_vs_int": 0.058841256259301256,
      "delta_ce_vs_tc": -0.056672276010248934,
      "mean_ce": {
        "int": 1.878553244150253,
        "mx": 1.9373945004095543,
        "tc": 1.9940667764198032
      },
      "no_tc_improvement": false,
      "ppl_ratio_vs_int": 1.0606068626752387,
      "ppl_ratio_vs_tc": 0.944903686235298
    },
    "q8k4": {
      "above_int_margin": false,
      "delta_ce_vs_int": 0.005936850394081761,
      "delta_ce_vs_tc": 0.0009463518835446383,
      "mean_ce": {
        "int": 1.878553244150253,
        "mx": 1.8844900945443348,
        "tc": 1.8835437426607902
      },
      "no_tc_improvement": true,
      "ppl_ratio_vs_int": 1.005954508417435,
      "ppl_ratio_vs_tc": 1.0009467998157777
    }
  }
}
```

The primary Q4K4 rule uses the equal mean of three unrounded NLL/target values: activate MXFP4 when MXINT8 has no improvement over matched TC, or exceeds matched INT Top-p by more than log(1.01) CE. Q8K4 is secondary. This writer reports the archived decision and does not revise the trigger. Conditional FP4 evidence is follow-up evidence. All six conditional cells were activated.

MX adds block32 E8M0 sidecar metadata: full MXINT8 is 8.25 logical bits/element and a retained four-bit predictor is 4.25. Native MXFP4 is 4.25; conditional Q8K4 uses MXINT8 full-Q8 (8.25) with MXFP4 K4 (4.25). These sidecars originate from original INT DQ values, introduce extra quantization/scale confounds, and do not replace the original final operands.

## Paired-set evidence

Each accepted pairing manifest covers 192 same-input head comparisons and 576 cross-run comparisons (192 each against INT, FP and TC), from six smoke traces. The table reports mean same-input agreement against original INT-full on candidate operands; complete cross-run summaries are in PAIRED_SETS.csv.

| Candidate | Mean Jaccard | Mean retained INT mass | Mean mass shortfall |
| --- | --- | --- | --- |
| format_mxfp4_q4k4_topp_p085 | 0.76044344 | 0.89188905 | 0.00385883 |
| format_mxfp4_q4k4_topp_p090 | 0.72177407 | 0.92432062 | 0.00396151 |
| format_mxfp4_q4k4_topp_p095 | 0.69843887 | 0.96008205 | 0.00344430 |
| format_mxint8q_mxfp4k_q8k4_topp_p085 | 0.83959305 | 0.88607268 | 0.00103211 |
| format_mxint8q_mxfp4k_q8k4_topp_p090 | 0.82250657 | 0.91907925 | 0.00197693 |
| format_mxint8q_mxfp4k_q8k4_topp_p095 | 0.76693938 | 0.95908309 | 0.00074830 |
| format_mxint8_tc4_q4k4_topp_p085 | 0.58562161 | 0.86926400 | 0.02407221 |
| format_mxint8_tc4_q4k4_topp_p090 | 0.56867739 | 0.91141091 | 0.01815354 |
| format_mxint8_tc4_q4k4_topp_p095 | 0.58027697 | 0.95482680 | 0.01035931 |
| format_mxint8_tc4_q8k4_topp_p085 | 0.79903480 | 0.87225964 | 0.00943953 |
| format_mxint8_tc4_q8k4_topp_p090 | 0.80210164 | 0.91090873 | 0.00771578 |
| format_mxint8_tc4_q8k4_topp_p095 | 0.78371606 | 0.95118666 | 0.00566222 |
| format_sm8_q4k4_topp_p085 | 0.39358116 | 0.94273215 | 0.00175607 |
| format_sm8_q4k4_topp_p090 | 0.36385229 | 0.96794617 | 0.00020554 |
| format_sm8_q4k4_topp_p095 | 0.35218819 | 0.98698846 | 0.00020501 |
| format_sm8_q8k4_topp_p085 | 0.60475311 | 0.91458494 | 0.00387831 |
| format_sm8_q8k4_topp_p090 | 0.56594689 | 0.94840847 | 0.00023090 |
| format_sm8_q8k4_topp_p095 | 0.51447756 | 0.97615762 | 0.00003647 |

## Causal update

The existing [REPORT.md](/output/project/experiments/stp-format-study/diagnosis/REPORT.md) and [REPORT.md](/output/project/experiments/stp-format-study/diagnosis-distributions/REPORT.md) establish one-sided TC truncation, omitted mixed terms, relative-logit distortion and missed high-mass keys in captured smoke heads. A removable common logit bias alone does not explain selection error; mixed terms can cancel. More selected keys do not guarantee better selection. These observations do not attribute fractions of corpus NLL to individual terms.

SM Q4K4: p=.95: ΔCE versus TC=-0.050888745, versus INT=+0.001936922; p=.90: ΔCE versus TC=-0.109720184, versus INT=+0.003737869; p=.85: ΔCE versus TC=-0.171200497, versus INT=+0.009056380.

SM Q8K4: p=.95: ΔCE versus TC=-0.000288823, versus INT=+0.002060136; p=.90: ΔCE versus TC=+0.001603487, versus INT=+0.006336009; p=.85: ΔCE versus TC=+0.004514094, versus INT=+0.012404109.

MXINT8-TC4 Q4K4: p=.95: ΔCE versus TC=-0.024201986, versus INT=+0.028623681; p=.90: ΔCE versus TC=-0.055626966, versus INT=+0.057831086; p=.85: ΔCE versus TC=-0.090187876, versus INT=+0.090069001.

MXINT8-TC4 Q8K4: p=.95: ΔCE versus TC=+0.000494143, versus INT=+0.002843102; p=.90: ΔCE versus TC=+0.001102213, versus INT=+0.005834735; p=.85: ΔCE versus TC=+0.001242699, versus INT=+0.009132714.

MXFP4 Q4K4: p=.95: ΔCE versus TC=-0.051722988, versus INT=+0.001102679; p=.90: ΔCE versus TC=-0.111643879, versus INT=+0.001814174; p=.85: ΔCE versus TC=-0.176733333, versus INT=+0.003523545.

MXINT8-Q8 / MXFP4-K4: p=.95: ΔCE versus TC=-0.002505322, versus INT=-0.000156363; p=.90: ΔCE versus TC=-0.005133764, versus INT=-0.000401243; p=.85: ΔCE versus TC=-0.009653633, versus INT=-0.001763619.

The per-threshold changes above preserve heterogeneous outcomes. Same-input pairing compares predictors on fixed candidate operands; cross-run comparisons also include upstream trajectory divergence. Neither smoke scope supplies independent corpus samples or a counterfactual full-model intervention. SM changes reconstruction behavior; MX additionally changes sidecar scales and requantization. A quality gain or loss alone cannot uniquely identify which mechanism caused it. Other causal interventions remain proposed in [SPEC.md](/output/project/experiments/stp-format-study/supplement-proposal/SPEC.md) and require user approval.

## Provenance and artifacts

Frozen execution Git HEAD: `124b78ae3e542bba3dc66bb35e58048373892ced`. Frozen source identity: `5910df4bbcb3ace61698c0945a3283527f0984fe66764375f7792767b326598a`. Live checkout HEAD at report generation: `124b78ae3e542bba3dc66bb35e58048373892ced`. Documentation commits do not redefine the recorded execution source hash.

Full NLL/targets and status: [FULL_TABLE.csv](FULL_TABLE.csv). Smoke paired sets: [PAIRED_SETS.csv](PAIRED_SETS.csv). Every rebound input and generated output is indexed with an absolute artifact path and SHA256 in [DELIVERY.manifest.json](DELIVERY.manifest.json).

| Artifact | Absolute path |
| --- | --- |
| BASELINE.csv | /output/project/experiments/stp-format-study/execution-v1/BASELINE.csv |
| BASELINE.manifest.json | /output/project/experiments/stp-format-study/execution-v1/BASELINE.manifest.json |
| BASELINE.decision.json | /output/project/experiments/stp-format-study/execution-v1/BASELINE.decision.json |
| freeze/freeze.json | /output/project/experiments/stp-format-study/execution-v1/freeze/freeze.json |
| freeze/code_manifest.json | /output/project/experiments/stp-format-study/execution-v1/freeze/code_manifest.json |
