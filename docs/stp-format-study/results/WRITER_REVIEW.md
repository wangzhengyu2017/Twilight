# External delivery writer review

Final verdict: the corrected writer is approved for postprocessing actual terminal evidence. No blocking code findings remain within this review scope. Actual final study results remain unapproved until execution is terminal and the complete real bundle validates. The initial writer required the corrections recorded below. This audit is read-only for the frozen checkout/specification/execution evidence. Only temporary synthetic fixtures were executed; no model, GPU task, or actual-result report was launched. The execution status inspected during review was `running` at stage `freeze`.

Reviewed: `delivery-tools/write_report.py`, `test_write_report.py`, `README.md`, `DOC_RECONCILIATION.md`; compared against archived `execution-v1/freeze/tools/{orchestrate,activate_conditional,index_artifacts}.py`, frozen summary/pairing producers, freeze identity, and matrix.

## Initial findings

1. **Blocking: conditional trigger identity is wrong and incompletely validated.** `load_bundle` returns `BASELINE.decision.json` even when the producer archived the authoritative activation decision as `INITIAL_BASELINE.decision.json`. It does not bind the adopted conditional matrix's `conditional_activation` paths/digests to those initial files or verify orchestration activation against that initial decision. A temporary synthetic probe with status activated=true, initial decision=true, final decision=false, and invalid activation hashes was accepted and returned false. The final decision must remain separately identified as the later summary decision, never replace the initial trigger.
2. **Blocking: evidence closure stops too soon.** Recursive JSON map traversal runs before initial manifest, conditional metadata, and diagnosis roots are added. A hash-valid initial source JSON pointing to a deliberately wrong nested leaf hash was accepted; the leaf never entered the evidence index. Add all evidence roots before traversing to closure, including the initial decision manifest and adopted/original matrices.
3. **Blocking: numeric rows need actual per-method source evidence.** The production build path accepted the one-row synthetic fixture with `verified=true` but no run path or matching run metrics, gates, or coverage. A CSV hash binding proves byte integrity, not that each accepted metric has its required source evidence. Check accepted rows against bound full-run metrics/status/coverage/provenance and matching archived release-gate evidence; preserve explicit per-method artifact paths. Require 289075-target identical coverage/asset identities for accepted comparable runs. Do not accept numeric RS rows against the source-blocked frozen matrix.
4. **Blocking: root artifact index ignored.** The producer generates `execution-v1/artifact-index.json` only at closure, after terminal status. The writer ignores it and accepts a production fixture without it. Require its terminal existence, schema/root identity, and every indexed file hash; index it in the delivery manifest. This also prevents publishing in the small terminal-status-before-index window.
5. **High: historical dense values are inserted without actual bound evidence.** Hardcoded constants are printed even when neither historical dense row nor dense metrics exists. The test currently asserts that behavior. Preserve the known parser-gate limitation, but require the actual bound historical measurement before displaying a numeric reference. Unavailable historical evidence should remain unavailable.
6. **Medium: full precision is lost.** `FULL_TABLE.csv` formats NLL and PPL to twelve decimal places. Preserve original numeric CSV strings or round-trip precision for full precision evidence, keeping shorter formatting only in the presentation table.
7. **Medium: pairing structure is underchecked.** Counts alone are checked; unknown summary roles are silently called cross-run trajectories, required roles need not exist, and summaries can be nonfinite. Require the four producer roles (`int`, `fp`, `tc`, `own_source_int_full`), 768 total rows, expected 576/192 scopes, and finite metrics; retain the same-input vs trajectory distinction already used in prose.

## Verification and scope

The original three synthetic unit tests passed. Two additional temporary negative probes demonstrated findings 1–5; their temporary report directories were automatically removed. One first probe deliberately corrupted a direct initial source hash and correctly failed, confirming direct initial sources were checked; the missing validation is recursive initial-only leaves and activation bindings.

The original frozen matrix contains all six conditional FP4 IDs even before activation, with `conditional_not_activated` status. Merely finding those IDs in the final CSV must never imply the six runs were activated or completed. The unified table correctly separates Q4K4 and Q8K4 and includes historical FP/INT Top-p and TC controls across all three thresholds; missing and blocked cells must retain statuses. Historical dense parser-gate limitations, original full Q8/K8 final operands, single Top-p selection, sidecar bit accounting, and causal scope limits are clearly explained.

The documentation reconciliation note correctly identifies copied source-relative link problems and the missing PADE documentation convention file. It defers actual result/doc reconciliation until terminal evidence; it does not authorize new experiments.

## Remediation status

Findings were sent directly to the writer agent and parent supervisor. Writer edits and expanded synthetic tests are in progress. No review finding is marked resolved until the patched writer is independently re-read and its targeted negative tests pass. No actual final study report has been approved by this review.

## First remediation re-review

The writer was independently re-read after its first remediation patch, and its eight expanded synthetic tests passed. The patch now binds the archived initial trigger and activation paths/hashes, rejects a changed later decision, validates initial matrix identity, requires and rehashes the execution artifact index, traverses evidence to closure before output, validates full run coverage/NLL/provenance plus passing release gate, preserves raw NLL/PPL strings, adds per-row metrics/status/gate paths, and checks pairing role/count/finite-metric structure. These changes address the original main findings.

Two specific remaining cases were sent for correction before approval:

- A failed historical dense row with its expected PPL still prints a numeric dense reference, bypassing measured-row validation. A temporary synthetic probe reproduced this with `status=failed` and an empty run path.
- The adopted conditional matrix is not yet checked against the producer's deterministic transformation of the original six conditional rows. The first positive fixture actually retains `conditional_not_activated` statuses in the adopted matrix, yet the writer accepts it and reports activation. Require the conditional phase, exactly the six original conditional rows marked required/activated, preserved definitions, and reject accepted conditional numeric rows when activation is false.

Cross-method token/asset/scoring-range identity checks were also requested to make the same-coverage requirement explicit. The current patch validates coverage independently within each run. The actual active execution `--check-only` invocation still safely returned exit 2 without producing output. These are synthetic/structural audit results, not experiment measurements.


## Final remediation verdict

All reported defects were corrected and independently re-reviewed. The final writer is suitable for postprocessing the actual closed execution bundle; it does not constitute approval of experiment results while execution is still running.

Final verification:

- All **15 synthetic unit tests passed** independently. Their temporary files are TEST-labelled and generated fixture reports explicitly say TEST DATA ONLY; no synthetic measurements were retained as study evidence.
- An independent synthetic CLI refusal probe asserted exit 2, empty stdout, a nonterminal explanation on stderr, and no created output directory.
- The final actual `--check-only` invocation exited 2 with `Execution is not terminal complete; no final report written`. No actual final report was generated by this review.
- Both existing historical dense runs passed the writer's read-only full-coverage/provenance structural validation with gate enforcement disabled, preserving their documented parser-matched-gate limitation. A read-only scan of 11 existing saved pairing manifests found no metric-range conflict with the new finite [0,1] checks.

The final code reconstructs both widths' frozen CE decision from all 15 matched verified MX/TC/INT inputs, including the complete means/deltas/ratios/Boolean schema. During conditional execution it uses the archived initial CSV/decision, requires exact initial-to-validated-final run/metric/provenance matches, and checks the deterministic six-row activation transformation and all six accepted conditional full runs. The numeric consistency tolerance applies only to finite serialized decision values; trigger Booleans remain exact. A later summary decision cannot silently replace the initial trigger.

Accepted candidate rows now require actual bound full-run artifacts, matching release gates, frozen candidate source identity, common corpus identities, and a unique complete bound paired-set manifest. FULL_TABLE preserves original NLL/PPL strings and links per-method metrics/status/gate/pairing paths. Dense absent or failed rows and all unverified format rows suppress numeric results. RS source-blocked rows cannot become accepted numeric results. The execution artifact index and all recorded hashes are mandatory, and transitive evidence closure completes before output creation.

Reviewed file identities:

- `write_report.py`: SHA256 `20433803afb212e4c77f5d793fdcbfb2e7e57d8467c2c18e2eb99d4194ed209d`.
- `test_write_report.py`: SHA256 `eec191baa25c7f4114a7492ae7bbb401d250aaf7b7a6217c17d693b87a7145f4`.

The integrity manifests remain producer-bound checks, not independent signatures. This review does not rerun GPU release gates or trace replay. Git code/specification, frozen execution-root files, and the running model study were not modified by the reviewer. Documentation reconciliation remains deferred until the actual terminal report validates.
