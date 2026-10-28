# Format-aware MSB predictor study

Implemented: sign-magnitude and MXINT8 predictor sidecars for Q4K4/Q8K4 at p=.95/.90/.85; conditional native/mixed MXFP4 paths are implemented but run only after the frozen MXINT8 result rule. Final scores always use original INT-Flash Q8/K8, with one Top-p selection. RS-INT8 remains blocked pending a verified BADA encoding definition.

- [SPEC](SPEC.md), [execution and conditional rule](MX_EXECUTION_DECISION.md).
- [MSB information analysis](MSB_INFORMATION_ANALYSIS.md), [additional source statistics](SOURCE_DISTRIBUTIONS.md).
- [Supplemental causal SPEC, separate approval required](SUPPLEMENT_SPEC.md).
- [Preceding measured baseline](../stp-baselines/THRESHOLD_BASELINE.md).

Implementation and synthetic CPU/CUDA validation are complete. Actual format model smoke, replay and full PPL are pending; no format PPL is claimed in this commit. Full machine evidence and historical links live at `/output/project/experiments/stp-format-study`; source checkout is `/output/project/Twilight-STP_test_Format`. Files copied here are review snapshots; execution freezes their authoritative external sources and hashes.
