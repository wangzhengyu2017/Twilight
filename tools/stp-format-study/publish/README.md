# Post-delivery publisher

Prepared while the long experiment is running. Nothing here launches GPU/model work. No publication has been executed. The publisher validates and prints a concrete path/hash plan by default; it refuses incomplete execution, partial independent audit, missing index, failed writer, changed evidence, wrong branch/HEAD/origin or dirty checkout.

After terminal execution, run `python publish.py` from this directory for the real dry run. It reads the reviewed writer (SHA256 `20433803afb212e4c77f5d793fdcbfb2e7e57d8467c2c18e2eb99d4194ed209d`), rehashes the execution index, independent audit and full delivery source/output closure, rebuilds the report in disposable storage and compares its exact bytes. The independent audit must name exactly the twelve initial variant IDs and, when activated, all six conditional IDs in their correct phases. Each cell is rebound to its indexed current metrics digest, NLL/PPL/timing/target/window records and an accepted matching FULL_TABLE row. The complete validated index/source/output closure, writer identity, audit and writer-success state stay hash-bound through the prewrite and postcopy checks. It requires successful `preparation/delivery-autostart.json`; manual writer success without that state must be reconciled explicitly rather than guessed. It never accepts synthetic delivery output.

Root review must independently approve this exact publisher before execution. The external approval JSON has `approved: true`, `scope: "execute-format-study-publication"`, and `publisher_sha256` set to the actual SHA256 of publish.py. The existing user authorization covers the eventual push; this is a technical review gate, not a new permission request.

```
python publish.py --execute --root-review /absolute/external/ROOT_REVIEW.json --receipt-dir /absolute/external/new-publication-receipt
```

Execution requires clean `STP_test_Format` at execution source `124b78ae3e542bba3dc66bb35e58048373892ced`, with all fetch and push URLs each exactly the singleton `git@github.com:wangzhengyu2017/Twilight.git`. Before the first checkout write it repeats Git guards and source hashes, refuses destination collisions/symlinks and source directory symlinks/escapes, then writes only the reviewed whitelist, rehashes every copy, stages only explicit paths, commits `results: format study baselines and diagnostics`, pushes `origin STP_test_Format` without force and verifies `ls-remote` against local HEAD. Missing author name/email use the execution commit's original author via per-command Git options. Git configuration is not changed.

An external new receipt directory receives the concrete plan before mutation and the verified commit/remote/status/file hashes after push. A failure preserves completed work for inspection; there is no automatic reset, clean, force push, stash or retry. If a commit was made but push failed, inspect and reconcile it manually; the expected-HEAD guard deliberately blocks a blind rerun. Do not publish while any producer is still changing execution evidence. The final whole-study ARTIFACT_INDEX.json is closed by the parent after background publication; this script does not claim it exists or include a fabricated root hash.

The exact destination mapping is defined by `TEXT_SOURCES`, `CODE_SOURCES`, `PRODUCTS` and the small explicit generated paths in `plan()`. The whitelist consists of:

- `docs/stp-format-study/{README.md,MSB_INFORMATION_ANALYSIS.md,SOURCE_DISTRIBUTIONS.md,MATHEMATICAL_NOTE.md,SMOKE_SELECTION_BUDGET.md,METRIC_DEFINITIONS.md,SUPPLEMENT_PROPOSAL.md,SUPPLEMENT_REASONING.md,SUPPLEMENT_REVIEW.md}`.
- `docs/stp-format-study/history/{README.md,pre-delivery.md}` preserves the previous README text with relative links rebased to its new history location and distinguishes the preparation snapshot from the completed current status.
- `docs/stp-format-study/results/{REPORT.md,PPL_TABLE.csv,FULL_TABLE.csv,PAIRED_SETS.csv,DELIVERY.manifest.json,INDEPENDENT_AUDIT.json,WRITER_REVIEW.md,PUBLICATION.manifest.json}`.
- `tools/stp-format-study/orchestration/{README.md,orchestrate.py,freeze.py,activate_conditional.py,index_artifacts.py,progress.py,rebind_spec.py,controls.json}`.
- `tools/stp-format-study/delivery/{README.md,DOC_RECONCILIATION.md,write_report.py,test_write_report.py,autostart.py}`.
- `tools/stp-format-study/review/{independent_auditor_v1.py,independent_auditor_v1.provenance.json}`.
- `tools/stp-format-study/diagnostics/{analyze_tc.py,compare_saved_layer0.py,summarize_smoke_selection.py}`.
- `tools/stp-format-study/publish/{publish.py,README.md,test_publish.py}`.

Diagnostic/proposal Markdown copies transform relative links to their actual absolute external source locations. This fixes SOURCE_DISTRIBUTIONS' inherited links to main diagnosis, historical proposal and CSV/JSONL evidence. The exact transform is `external_links` and the publication manifest hashes both original source files and copied bytes. The delivered report/CSVs/manifest remain byte-for-byte unchanged, including valid copied relative report table links and absolute machine artifact paths. No raw traces, models, datasets, private .codex files or algorithm/specification changes are included. Archived tools preserve original machine paths and invocation semantics; they are reproducibility source snapshots, not silently relocated experiment state. The current result README links the concrete approval-required supplemental proposal; the existing frozen SPEC and historical proposal remain untouched.

The publisher updates Git's current/history docs. Root separately reconciles external study README/history/status after actual terminal delivery and before closing its final whole-study index; the publisher does not mutate those source documents or execution-v1.

Run `python test_publish.py` for 21 lightweight isolated guard checks, including stale audit metrics, omitted/unverified delivery cells, additional remote URLs, preserved history link destinations, source directory symlinks and underlying evidence changing after copy. All fixture data are TEST-labelled disposable temporary files. No actual repository writes, experiment work, commit or push occurs in tests.
