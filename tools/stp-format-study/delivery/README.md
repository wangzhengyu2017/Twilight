# External delivery writer

This tool reads closed study artifacts and writes a new external report directory. It never starts a model, GPU task, diagnostic tensor analysis, or experiment, and never edits frozen code/specification/execution evidence.

After all initial and any activated conditional full runs have finished:

```bash
python /output/project/experiments/stp-format-study/delivery-tools/write_report.py \
  --execution-root /output/project/experiments/stp-format-study/execution-v1 \
  --study-root /output/project/experiments/stp-format-study \
  --output-dir /output/project/experiments/stp-format-study/delivery-v1
```

The output directory must be new and outside execution-v1. `--check-only` validates without writing. While orchestration is running or evidence is missing/corrupt, the writer exits 2 without a report. A complete terminal execution with missing/unverified rows preserves statuses and suppresses their numeric metrics; it does not invent quality evidence.

Validation binds BASELINE.csv and its decision to BASELINE.manifest.json, rehashes all recorded source/artifact/gate evidence transitively, verifies the archived matrix/specification, source files and patch, checks source manifest identity, and independently checks full/windows target coverage, window NLL totals, asset/code/environment/config identities and matching passing gate identities. It compares frozen corpus identities across measured methods; accepted format/control environments must match. Historical dense environment differences remain explicit historical provenance. The terminal execution artifact index is mandatory and all its recorded hashes are rebound. When conditional execution occurs, INITIAL_BASELINE is the authority: both its output/decision and nested sources, exact activation paths/hashes, the original matrix identity, deterministic six-cell activation transformation and all six accepted conditional full rows are required. Each accepted format also requires a unique complete bound pairing manifest with all four reference roles, 768 rows, finite bounded set/mass metrics and underlying source/artifact hashes; FULL_TABLE links its gate and pairing paths. Final and initial decisions must agree with terminal activation status. This consumes the frozen producer's `verified` decision; it does not rerun GPU gates or replace the established evidence protocol. Historical dense rows explicitly retain their parser-matched gate limitation. BASELINE manifest is an integrity binding, not a cryptographic signature or independent trusted timestamp.

Outputs: concise English REPORT.md; unified PPL_TABLE.csv; full NLL/target/status FULL_TABLE.csv; scope-separated PAIRED_SETS.csv; absolute-path SHA256 DELIVERY.manifest.json. FULL_TABLE.csv preserves original CSV strings for NLL, targets and PPL without rounding. Dense values are emitted only from validated complete historical rows; constants in code are expected-value checks, never fallback measurements. The writer independently recomputes the frozen decision from all 15 verified matched MX/TC/INT inputs for both widths and three thresholds. Conditional execution uses INITIAL_BASELINE inputs, matched exactly to validated final run identities and raw metrics. The complete archived decision shape, CE means/deltas/ratios and Booleans must agree (finite binary64 values compare at rel/abs 2e-15; trigger Booleans compare exactly). The writer never rewrites the predeclared conditional rule. It reports the archived complete decision including both widths and every threshold's CE change against matched TC/INT. Native FP4 and mixed MXINT8-Q8/FP4-K4 remain distinct.

Run synthetic-only checks:

```bash
python /output/project/experiments/stp-format-study/delivery-tools/test_write_report.py
```

Fifteen tests cover conditional activation and provenance corruption, missing/failed dense evidence, raw precision, coverage totals, gate/corpus identities and unactivated FP4 rejection. Fixtures exist only inside temporary TEST-labelled directories and generated test reports say TEST DATA ONLY. They are removed automatically. There are no saved synthetic measurements in the study. The real running evidence was also checked with `--check-only`; it was safely refused as nonterminal, without writing outputs.

Documentation reconciliation instructions are in [DOC_RECONCILIATION.md](DOC_RECONCILIATION.md). No Git documentation changes or commits are made by this writer.
