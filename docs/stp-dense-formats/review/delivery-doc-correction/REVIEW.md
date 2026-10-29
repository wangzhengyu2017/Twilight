# Documentation correction review

PASS. The two previously identified documentation findings are resolved.

Dense REPORT and its generator contain only the filename replacement from `selection_sets.jsonl` to `diagnostic_sets.jsonl` relative to their prior reviewed hashes. Reversing that replacement reconstructs each prior SHA256. Both current index bindings match file hashes and byte sizes. The index still contains 881 entries; the other 879 hash bindings match the prior review. The four diagnostic files each contain 576 JSONL rows and retain their indexed, previously reviewed hashes.

The scientific MECHANISM header and preface now explicitly mark the preparation-stage status as historical and link current REPORT. Its previous hash was not recorded, so this review does not assert a byte-identical remainder. The scientific core and four published plot files retain their reviewed hashes.

This narrow review relies on the prior 881-file/574-gate-file integrity and scientific/visual review; it does not repeat full hashing, GPU/model work, tests or gates. The two prior JSON files are byte-identical copies of the combined review, not separate executions.

Prior review references:

- `/output/project/experiments/stp-format-study/review/scientific-synthesis-v1-independent-review.json` — SHA256 `1f9e2ba06b6bbe56c0ac1df5d346734e90329debe6aec1c1da6137b12a629e3b`

- `/output/project/experiments/stp-dense-formats/review/delivery-dense-v1-independent-review.json` — SHA256 `1f9e2ba06b6bbe56c0ac1df5d346734e90329debe6aec1c1da6137b12a629e3b`

Current artifact SHA256 values:

- `/output/project/experiments/stp-dense-formats/delivery-dense-v1/REPORT.md` — `5b727f1f1698d00955dc8aaa7f6db4536f1529e8ab2aed109b596ddba2594be5`

- `/output/project/experiments/stp-dense-formats/delivery-tools/write_dense_report.py` — `e88d22404ad05d3408fc0120b2393611bd7b1f7db9a4bad3cd75b9d3a2822935`

- `/output/project/experiments/stp-dense-formats/delivery-dense-v1/ARTIFACT_INDEX.json` — `e38cea98b753e25b63d73d8b79697812ac89998127401f1041a12c1eddd1f79a`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/MECHANISM.md` — `bec727786447162749ef8032e1f643cd869bc1de8975732c78b165d2fc4443e3`

- `/output/project/experiments/stp-format-study/analysis-final-readable/RENDER.manifest.json` — `820ab7f993e4e9dca2624c875f17a3e4c7d09a38e1a72f0385dcb17f8a404f98`

- `/output/project/experiments/stp-dense-formats/delivery-dense-v1/RESULTS.csv` — `5db5bd9ee9f93b38a32528b88e26f2706f7c1cddb5e4a87e1ee1a0af5d3303c7`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/MEASURED_RESULTS.csv` — `1107b11bde72d43179c6a606e63e00aac8eab5726734335d35a877b0ba0890bc`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/REPORT.md` — `c04543d4844b41ed27a9d49ffe89606355c852886687f67c2e57834aa4cd88d7`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/STATUS.md` — `f50f94cdaffdc385d1693a1ec677a51389e537118dfe3d0e9cf3efc44cf82fba`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/SYNTHESIS.manifest.json` — `47a6947e62b80837e9589c7f0ee3f883e4e6733b82629efea2093583da54b28b`

- `/output/project/experiments/stp-format-study/scientific-synthesis-v1/synthesize.py` — `5b5944ac5bc7b0f46bc63e7d4cfdac9773dd18f678767a691773cc173eef4523`

- `/output/project/experiments/stp-format-study/analysis-final-readable/ppl_vs_top_p.pdf` — `6aba5c5ad0376522e7495b7e585183cd17316a9170c1d18c32e546d6821a23cf`

- `/output/project/experiments/stp-format-study/analysis-final-readable/ppl_vs_selected_fraction.pdf` — `a5d313cb80902bfb6abd55814bd045e21866d98b2f081fdcb5a2285c7c127dda`

- `/output/project/experiments/stp-format-study/analysis-final-readable/ppl_vs_selected_fraction.png` — `04d81fab79df17470f852414b615ea94300764965f3b0933521f2245f96ad9c2`

- `/output/project/experiments/stp-format-study/analysis-final-readable/ppl_vs_top_p.png` — `0b1e9f788868ccc882ddae08e99790622cec655fba8efe1a177c8243476e4521`

Detailed checks and scope are recorded in [REVIEW.json](REVIEW.json).
