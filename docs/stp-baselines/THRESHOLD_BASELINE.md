# Frozen WikiText-2 attention baselines

Model: Llama-3.1-8B-Instruct, BF16 weights and FP32 attention. All runs reuse the same frozen WikiText-2 raw test tokens, C2048/S1024, and score 289075 targets in 282 windows. Lower PPL is better.

| Method | Dense PPL | p=0.95 PPL | p=0.90 PPL | p=0.85 PPL |
| --- | ---: | ---: | ---: | ---: |
| FP | 6.484759140 | 6.482529433 | 6.530139845 | 6.613164945 |
| INT-Flash Q8/K8 | 6.486693431 | 6.485038852 | 6.532782195 | 6.614928624 |
| TC-MSB Q4×K4 | N/A | 6.836825225 | 7.317662664 | 7.921542008 |
| TC-MSB Q8×K4 | N/A | 6.500289851 | 6.563771998 | 6.667326949 |

MSB predicts one Top-p mask, then computes selected scores from the original full Q8/K8 codes and scales. There is no second Top-p. Quantization uses the inherited native FP16 input cast policy. V and the original floating cache remain unchanged.

| Variant | Total NLL | PPL | Targets |
| --- | ---: | ---: | ---: |
| fp_topp_p095 | 540313.198364257812 | 6.482529432622 | 289075 |
| int_topp_p095 | 540425.079101562500 | 6.485038852466 | 289075 |
| msb_q4k4_topp_p095 | 555695.658691406250 | 6.836825225318 | 289075 |
| msb_q8k4_topp_p095 | 541104.104553222656 | 6.500289850707 | 289075 |
| fp_common | 540412.610473632812 | 6.484759140305 | 289075 |
| intflash_qk8 | 540498.823669433594 | 6.486693431358 | 289075 |
| fp_topp_p085 | 546080.699096679688 | 6.613164945110 | 289075 |
| int_topp_p085 | 546157.782836914062 | 6.614928623952 | 289075 |
| msb_q4k4_topp_p085 | 598265.539672851562 | 7.921542008230 | 289075 |
| msb_q8k4_topp_p085 | 548438.588867187500 | 6.667326948693 | 289075 |
| fp_topp_p090 | 542428.527832031250 | 6.530139845058 | 289075 |
| int_topp_p090 | 542545.475219726562 | 6.532782194945 | 289075 |
| msb_q4k4_topp_p090 | 575343.361816406250 | 7.317662664364 | 289075 |
| msb_q8k4_topp_p090 | 543913.528808593750 | 6.563771997809 | 289075 |

## Verification and provenance

The p=0.90 and p=0.85 sweeps passed both complete smoke matrices, 48 real trace replays, 2304 paired head comparisons, exact full-corpus target/NLL accounting and final artifact-index verification. The p=0.95 study has its own accepted evidence. Numerical attention/scoring files are unchanged between these threshold studies; only the threshold and acceptance/reporting identities were generalized. Historical timing is not a controlled speed comparison.

Evaluated threshold source manifest SHA256: `b15fcb5f2489e590633af02cec925c0dbf9acd916394151ae747a14461ef4014`. This identifies the archived pre-commit source snapshot, not a Git commit.

HyperAI artifact locations (absolute filesystem paths):

- Threshold report: `/output/project/experiments/stp-slice-thresholds/REPORT.md`
- Full metrics and comparison data: `/output/project/experiments/stp-slice-thresholds/summary.json`
- Complete artifact index: `/output/project/experiments/stp-slice-thresholds/artifact-index.json`
- Frozen execution source: `/output/project/experiments/stp-slice-thresholds/preparation/execution-source/`
- p=0.95 report and source delivery: `/output/project/experiments/stp-slice/REPORT.md` and `/output/project/experiments/stp-slice/delivery-v2/`

Model weights, tokens and raw traces remain external artifacts and are not copied into Git.
