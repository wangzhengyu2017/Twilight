#!/usr/bin/env python3
"""Read closed, hash-bound evidence and create an external delivery bundle. No model imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

TARGETS = 289075
DENSE = {'fp_common': 6.484759140305276, 'intflash_qk8': 6.486693431357931}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class Evidence:
    def __init__(self):
        self.files = {}
        self.source_archives = ()

    def bind(self, path, digest=None):
        path = Path(path).resolve()
        actual = sha(path)
        if digest is not None and actual != digest:
            raise ValueError('Changed evidence: ' + str(path))
        if str(path) in self.files and self.files[str(path)] != actual:
            raise ValueError('Evidence changed during validation: ' + str(path))
        self.files[str(path)] = actual
        return path

    def configure_sources(self, root):
        """Only archived snapshots are eligible; never consult a live checkout/CWD."""
        root = Path(root).resolve()
        historical = root.parent.parent / 'stp-slice-thresholds/preparation/execution-source'
        self.source_archives = (
            (root / 'freeze/code_manifest.json', root / 'freeze/source'),
            (historical / 'code_manifest.json', historical / 'files'),
        )

    @staticmethod
    def source_path(root, name):
        # Reject traversal and aliases even when normalization would stay inside.
        relative = Path(name)
        if relative.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
            raise ValueError('Unsafe archived source path: ' + name)
        root = Path(root).resolve()
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError('Archived source path escapes snapshot: ' + name)
        return path

    def bind_source(self, document, name, digest, owner=None):
        identity = document.get('code_sha256')
        if not identity:
            raise ValueError('Relative source lacks code_sha256: ' + str(owner))
        matches = []
        for manifest_path, source_root in self.source_archives:
            if not manifest_path.is_file():
                continue
            manifest = read(manifest_path)
            if manifest.get('code_sha256') == identity:
                matches.append((manifest_path, source_root, manifest))
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous archived source for code_sha256: ' + str(identity))
        manifest_path, source_root, manifest = matches[0]
        if canonical({'head': manifest['head'], 'files': manifest['files']}) != identity:
            raise ValueError('Invalid archived source manifest identity: ' + str(manifest_path))
        self.bind(manifest_path)
        if manifest['files'].get(name) != digest:
            raise ValueError('Source digest differs from archived manifest: ' + name)
        return self.bind(self.source_path(source_root, name), digest)

    def maps(self, document, owner=None):
        if not isinstance(document, dict):
            return
        for key in ('source_files', 'artifacts', 'evidence_files'):
            for name, digest in document.get(key, {}).items():
                if Path(name).is_absolute():
                    self.bind(name, digest)
                elif key == 'source_files':
                    self.bind_source(document, name, digest, owner)
                else:
                    raise ValueError('Artifact/evidence path must be absolute: ' + name)
        for key in ('inputs', 'outputs'):
            for entry in document.get(key, []):
                if isinstance(entry, dict) and 'path' in entry and 'sha256' in entry:
                    if not Path(entry['path']).is_absolute():
                        raise ValueError('Input/output path must be absolute: ' + entry['path'])
                    self.bind(entry['path'], entry['sha256'])

    def closure(self):
        checked = set()
        while set(self.files) - checked:
            for name in sorted(set(self.files) - checked):
                checked.add(name)
                path = Path(name)
                if path.suffix == '.json':
                    self.maps(read(path), owner=path)


def raw_metric(row, key):
    if not row.get(key):
        return 'unavailable'
    numeric(row[key])
    return row[key]


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def numeric(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite metric')
    return result


def accepted(row):
    return row.get('status') == 'complete' and str(row.get('verified')).lower() == 'true'


def metric(row, key, digits=6):
    return f'{numeric(row[key]):.{digits}f}' if row.get(key) else 'unavailable'


def link(path):
    path = Path(path).resolve()
    return f'[{path.name}]({path})'


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check_full_run(run, metrics, evidence, require_gate):
    def bound(name):
        path = (run/name).resolve()
        if str(path) not in evidence.files:
            raise ValueError('Full run lacks bound evidence: ' + str(path))
        return read(path)
    if metrics.get('profile') != 'full' or metrics.get('mode') != 'windows':
        raise ValueError('Measured row is not full windows: ' + str(run))
    assets, coverage = bound('asset_manifest.json'), bound('coverage.json')
    if type(metrics['num_targets']) is not int or type(metrics['num_input_tokens']) is not int or metrics['num_input_tokens'] != metrics['num_targets']+1:
        raise ValueError('Invalid full integer target/input counts')
    if numeric(metrics['sum_nll']) < 0 or numeric(metrics['ppl']) < 1:
        raise ValueError('Invalid full NLL/PPL domain')
    targets = metrics['num_targets']
    if targets != TARGETS or targets != assets['num_tokens']-1 or targets != assets['windows']['expected_targets']:
        raise ValueError('Full asset/target coverage mismatch')
    if canonical(assets) != metrics['asset_manifest_sha256'] or assets['tokens_sha256'] != metrics['tokens_sha256']:
        raise ValueError('Full asset identity mismatch')
    ranges = coverage['scoring_ranges']
    if canonical(ranges) != metrics['scoring_ranges_sha256'] or coverage['num_targets'] != targets or coverage['all_layers_covered'] is not True:
        raise ValueError('Full coverage identity mismatch')
    next_target = 1
    for row in ranges:
        if row['lo'] != next_target or row['target_end'] <= row['lo']:
            raise ValueError('Full coverage gap or overlap')
        next_target = row['target_end']
    if next_target != assets['num_tokens']:
        raise ValueError('Full coverage tail missing')
    windows_path = run/'window_metrics.jsonl'
    if str(windows_path.resolve()) not in evidence.files:
        raise ValueError('Full run lacks bound window evidence')
    windows = [json.loads(line) for line in windows_path.read_text().splitlines() if line.strip()]
    if len(windows) != len(ranges):
        raise ValueError('Full window count mismatch')
    for row, expected in zip(windows,ranges):
        if any(row.get(k) != v for k,v in expected.items()) or row['num_targets'] != expected['target_end']-expected['lo'] or not row.get('logits_sha256'):
            raise ValueError('Full window coverage mismatch')
        if type(row['num_targets']) is not int or numeric(row['sum_nll']) < 0:
            raise ValueError('Invalid window NLL/target domain')
    if sum(r['num_targets'] for r in windows) != targets or sum(r['sum_nll'] for r in windows) != metrics['sum_nll']:
        raise ValueError('Full window NLL/target totals mismatch')
    code, env = bound('code_manifest.json'), bound('environment.json')
    config = bound('resolved_config.json')['algorithm']
    if canonical({'head':code['head'],'files':code['files']}) != code['code_sha256'] or canonical({k:v for k,v in env.items() if k != 'environment_sha256'}) != env['environment_sha256']:
        raise ValueError('Archived run source/environment identity mismatch')
    if str((run/'code.patch').resolve()) not in evidence.files or sha(run/'code.patch') != code['patch_sha256']:
        raise ValueError('Archived run patch mismatch')
    if code['code_sha256'] != metrics['code_sha256'] or env['environment_sha256'] != metrics['environment_sha256'] or canonical(config) != metrics['semantic_config_sha256']:
        raise ValueError('Full run provenance mismatch')
    if require_gate:
        gate_hash = metrics.get('gate_report_sha256')
        gates = [Path(name) for name,digest in evidence.files.items() if gate_hash and digest == gate_hash and Path(name).suffix == '.json']
        for path in gates:
            gate = read(path)
            if gate.get('passed') is not True or gate.get('status') != 'complete':
                continue
            if any(gate.get(k) != metrics[k] for k in ('code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256')):
                continue
            hashes = gate.get('semantic_config_sha256')
            if isinstance(hashes,dict) and hashes.get(metrics.get('variant_id',metrics.get('method'))) != metrics['semantic_config_sha256']:
                continue
            evidence.maps(gate)
            return str(path)
        raise ValueError('Accepted row lacks bound passing matching release gate: ' + str(run))
    return ''


def check_decision(rows, final_rows, decision):
    lookup = {}
    for row in rows:
        if accepted(row):
            if row['variant_id'] in lookup:
                raise ValueError('Duplicate decision input')
            lookup[row['variant_id']] = row
    final = {r['variant_id']:r for r in final_rows if accepted(r)}
    expected = dict(status='complete',primary='q4k4',widths={})
    for q in (4,8):
        means = {}
        for role,prefix in [('mx',f'format_mxint8_tc4_q{q}k4_topp_p'),('tc',f'msb_q{q}k4_topp_p'),('int','int_topp_p')]:
            group=[]
            for p in ('095','090','085'):
                vid=prefix+p
                if vid not in lookup or vid not in final:
                    raise ValueError('Missing verified decision input: '+vid)
                row=lookup[vid]
                for key in ('run','sum_nll','num_targets','ppl','code_sha256','environment_sha256','asset_manifest_sha256','tokens_sha256','scoring_ranges_sha256'):
                    if row.get(key) != final[vid].get(key):
                        raise ValueError('Initial decision input differs from validated final evidence: '+vid+' '+key)
                group.append(numeric(row['sum_nll'])/int(row['num_targets']))
            means[role]=sum(group)/3
        delta_tc,delta_int=means['mx']-means['tc'],means['mx']-means['int']
        expected['widths'][f'q{q}k4']=dict(mean_ce=means,delta_ce_vs_tc=delta_tc,delta_ce_vs_int=delta_int,ppl_ratio_vs_tc=math.exp(delta_tc),ppl_ratio_vs_int=math.exp(delta_int),no_tc_improvement=means['mx']>=means['tc'],above_int_margin=delta_int>math.log(1.01))
    primary=expected['widths']['q4k4']
    expected['activate_mxfp4']=primary['no_tc_improvement'] or primary['above_int_margin']
    def same(actual,wanted):
        if isinstance(wanted,dict):
            return isinstance(actual,dict) and set(actual)==set(wanted) and all(same(actual[k],v) for k,v in wanted.items())
        if isinstance(wanted,bool):
            return type(actual) is bool and actual is wanted
        if isinstance(wanted,float):
            # Fixed tiny binary64 comparison for serialized numeric records;
            # the frozen trigger comparisons and their Booleans remain exact.
            return isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual) and math.isclose(actual,wanted,rel_tol=2e-15,abs_tol=2e-15)
        return actual == wanted
    if not same(decision,expected):
        raise ValueError('Archived decision does not match frozen binary64 CE rule and verified inputs')


def check_pairing(document):
    if document.get('status') != 'complete' or document.get('passed') is not True or document.get('counts') != {'cross_run_trajectory':576,'same_input_estimator':192} or document.get('num_rows') != 768:
        raise ValueError('Pairing scope or completion mismatch')
    if set(document['summary']) != {'int','fp','tc','own_source_int_full'}:
        raise ValueError('Pairing reference scope mismatch')
    if not document.get('source_files') or len(document.get('artifacts',{})) < 2:
        raise ValueError('Pairing underlying source/artifact evidence missing')
    for summary in document['summary'].values():
        for metric_name in ('jaccard','precision','recall','retained_mass','shortfall'):
            if 'mean' not in summary[metric_name] or any(not 0 <= numeric(v) <= 1 for v in summary[metric_name].values()):
                raise ValueError('Invalid pairing set/probability metric')


def load_bundle(root, evidence, allow_incomplete=False):
    root = Path(root).resolve()
    evidence.configure_sources(root)
    status = read(evidence.bind(root / 'orchestration_status.json'))
    if status.get('status') != 'complete' and not allow_incomplete:
        raise ValueError('Execution is not terminal complete; no final report written')
    baseline = evidence.bind(root / 'BASELINE.csv')
    manifest = read(evidence.bind(root / 'BASELINE.manifest.json'))
    evidence.bind(baseline, manifest['output_sha256'])
    decision_path = evidence.bind(root / 'BASELINE.decision.json', manifest['decision_sha256'])
    evidence.maps(manifest)
    final_decision = read(decision_path)
    activated = status.get('conditional_activated')
    if not isinstance(activated, bool):
        raise ValueError('Missing terminal conditional activation Boolean')
    initial = root / 'INITIAL_BASELINE.manifest.json'
    if activated:
        initial_manifest = read(evidence.bind(initial))
        evidence.bind(root / 'INITIAL_BASELINE.csv', initial_manifest['output_sha256'])
        initial_decision_path = evidence.bind(root / 'INITIAL_BASELINE.decision.json', initial_manifest['decision_sha256'])
        evidence.maps(initial_manifest)
        decision = read(initial_decision_path)
        if decision.get('status') != 'complete' or decision.get('activate_mxfp4') is not True:
            raise ValueError('Activation lacks affirmative initial decision')
        if final_decision != decision:
            raise ValueError('Final decision differs from immutable initial decision')
    else:
        if initial.exists() or (root/'conditional-handoff/matrix.json').exists():
            raise ValueError('Nonconditional status conflicts with conditional artifacts')
        decision = final_decision
        if decision.get('activate_mxfp4') is not False:
            raise ValueError('Terminal status conflicts with activation decision')
    index = read(evidence.bind(root/'artifact-index.json'))
    if index.get('schema_version') != 'format-artifact-index-v1' or Path(index.get('root','')).resolve() != root:
        raise ValueError('Invalid execution artifact index identity')
    indexed = index.get('files', {})
    for name, digest in indexed.items():
        if not Path(name).is_absolute() or root not in Path(name).resolve().parents:
            raise ValueError('Execution index contains non-execution artifact')
        evidence.bind(name, digest)
    for path in (root/'orchestration_status.json', root/'BASELINE.csv', root/'BASELINE.manifest.json', root/'BASELINE.decision.json'):
        if str(path.resolve()) not in indexed:
            raise ValueError('Terminal artifact absent from execution index: ' + str(path))
    freeze = read(evidence.bind(root / 'freeze/freeze.json'))
    code = read(evidence.bind(root / 'freeze/code_manifest.json'))
    # Match the producer's canonical hashing, which uses default JSON ASCII encoding.
    canonical = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical({'head': code['head'], 'files': code['files']}) != code['code_sha256']:
        raise ValueError('Invalid frozen source manifest identity')
    if code['head'] != freeze['head'] or code['code_sha256'] != freeze['code_sha256']:
        raise ValueError('Freeze/source identity mismatch')
    matrix = root / ('conditional-handoff/matrix.json' if status.get('conditional_activated') else 'freeze/handoff/matrix.json')
    adopted = read(evidence.bind(matrix, manifest['matrix_sha256']))
    original_matrix = evidence.bind(root / 'freeze/handoff/matrix.json', freeze['matrix_sha256'])
    original = read(original_matrix)
    conditional_ids = {v['variant_id'] for v in original['variants'] if v.get('status') == 'conditional_not_activated'}
    if len(conditional_ids) != 6:
        raise ValueError('Original matrix lacks six conditional definitions')
    if activated:
        evidence.bind(original_matrix, initial_manifest['matrix_sha256'])
        transformed = []
        for variant in original['variants']:
            variant = dict(variant, required=variant.get('status') == 'conditional_not_activated')
            if variant['required']:
                variant['status'] = 'conditional_activated_pending_smoke'
            transformed.append(variant)
        if adopted.get('execution_phase') != 'conditional' or adopted.get('variants') != transformed:
            raise ValueError('Conditional matrix does not match frozen activation transformation')
        activation = adopted.get('conditional_activation', {})
        if activation.get('trigger_verified') is not True:
            raise ValueError('Conditional matrix lacks verified activation')
        for key, expected in [('decision', root/'INITIAL_BASELINE.decision.json'), ('decision_manifest', initial)]:
            if Path(activation.get(key+'_path','')).resolve() != expected.resolve():
                raise ValueError('Conditional activation path mismatch: ' + key)
            evidence.bind(expected, activation[key+'_sha256'])
    elif adopted.get('conditional_activation') or adopted.get('execution_phase') == 'conditional':
        raise ValueError('Unactivated matrix contains conditional activation')
    if adopted.get('diagnosis_manifest'):
        diagnosis = adopted['diagnosis_manifest']
        evidence.maps(read(evidence.bind(diagnosis['path'], diagnosis['sha256'])))
    for name, digest in adopted.get('specification_files', {}).items():
        evidence.bind(root / 'freeze/spec' / Path(name).name, digest)
    evidence.bind(root / 'freeze/asset_manifest.json', freeze['asset_manifest_sha256'])
    evidence.bind(root / 'freeze/code.patch', code['patch_sha256'])
    for name, digest in code['files'].items():
        evidence.bind(evidence.source_path(root / 'freeze/source', name), digest)
    with baseline.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    evidence.closure()
    original_variants = {v['variant_id']:v for v in read(original_matrix)['variants']}
    expected = {f'format_{kind}_q{q}k4_topp_p{p}' for kind in ('sm8','mxint8_tc4') for q in (4,8) for p in ('095','090','085')}
    if not expected.issubset(original_variants):
        raise ValueError('Frozen matrix lacks initial format definitions')
    measured_metrics = []
    for row in rows:
        vid = row['variant_id']
        if accepted(row) and vid in conditional_ids and not activated:
            raise ValueError('Unactivated conditional variant cannot have accepted result: ' + vid)
        if accepted(row) or (vid in DENSE and row.get('status') == 'complete' and row.get('ppl')):
            run = Path(row.get('run',''))
            if not run.is_absolute():
                raise ValueError('Measured row lacks absolute run evidence: ' + vid)
            required = ['metrics.json','status.json','asset_manifest.json','coverage.json','window_metrics.jsonl','code_manifest.json','environment.json','resolved_config.json','code.patch']
            for name in required:
                if str((run/name).resolve()) not in evidence.files:
                    raise ValueError('Measured row lacks bound evidence: ' + str(run/name))
            metrics = read(run/'metrics.json')
            if read(run/'status.json').get('status') != 'complete':
                raise ValueError('Measured run is incomplete: ' + vid)
            if metrics.get('variant_id',metrics.get('method')) != vid:
                raise ValueError('Run variant identity mismatch: ' + vid)
            for key in ('sum_nll','num_targets','ppl'):
                if numeric(row[key]) != numeric(metrics[key]):
                    raise ValueError('CSV/run metric mismatch: ' + vid + ' ' + key)
            if int(row['num_targets']) != TARGETS:
                raise ValueError('Unexpected frozen WT2 target count: ' + vid)
            nll, ppl = numeric(row['sum_nll']), numeric(row['ppl'])
            if not math.isclose(math.exp(nll / TARGETS), ppl, rel_tol=1e-10):
                raise ValueError('PPL/NLL mismatch: ' + vid)
            row['_gate_path'] = check_full_run(run,metrics,evidence,accepted(row))
            if accepted(row) and vid.startswith('format_'):
                if metrics['code_sha256'] != freeze['code_sha256']:
                    raise ValueError('Format run differs from frozen execution source')
                pairs = [Path(name) for name in evidence.files if Path(name).name == 'pairing_manifest.json' and read(name).get('candidate_variant_id') == vid]
                if len(pairs) != 1:
                    raise ValueError('Accepted format lacks unique bound pairing manifest: '+vid)
                check_pairing(read(pairs[0]))
                row['_pairing_path'] = str(pairs[0])
            row['_measured_valid'] = True
            measured_metrics.append((metrics,accepted(row),vid))
            if accepted(row):
                if original_variants.get(vid,{}).get('status') == 'planned_source_blocked':
                    raise ValueError('Source-blocked variant cannot have accepted numeric result: ' + vid)
    if measured_metrics:
        reference = measured_metrics[0][0]
        for metrics,verified,vid in measured_metrics:
            for key in ('protocol_id','tokens_sha256','subset_sha256','scoring_ranges_sha256','asset_manifest_sha256','num_targets','num_input_tokens','model_dtype','backend'):
                if key not in reference or key not in metrics or metrics[key] != reference[key]:
                    raise ValueError('Incompatible measured metric identity: ' + vid + ' ' + key)
        environments = {m['environment_sha256'] for m,verified,vid in measured_metrics if verified and vid not in DENSE}
        if len(environments) > 1:
            raise ValueError('Incompatible accepted format/control environments')
    if activated and not conditional_ids.issubset({r['variant_id'] for r in rows if accepted(r)}):
        raise ValueError('Activated conditional full results are incomplete')
    evidence.closure()
    if activated:
        with (root/'INITIAL_BASELINE.csv').open(newline='') as stream:
            decision_rows=list(csv.DictReader(stream))
    else:
        decision_rows=rows
    check_decision(decision_rows,rows,decision)
    if decision.get('status') != 'complete':
        raise ValueError('Conditional decision unavailable; no final report written')
    return rows, decision, freeze, status


def build(root, study, output, test_data=False):
    evidence = Evidence()
    rows, decision, freeze, status = load_bundle(root, evidence)
    lookup = {}
    for row in rows:
        key = row['variant_id']
        if key in lookup and accepted(row) and accepted(lookup[key]):
            raise ValueError('Duplicate accepted full variant: ' + key)
        if key not in lookup or accepted(row):
            lookup[key] = row
    def display(vid):
        row = lookup.get(vid)
        if row and accepted(row):
            return metric(row, 'ppl')
        return (row.get('status', 'absent') + '; unverified') if row else 'absent'
    groups = [('FP Top-p', 'fp_topp_p'), ('INT Top-p', 'int_topp_p'),
              ('TC Q4K4', 'msb_q4k4_topp_p'), ('TC Q8K4', 'msb_q8k4_topp_p'),
              ('SM Q4K4', 'format_sm8_q4k4_topp_p'), ('SM Q8K4', 'format_sm8_q8k4_topp_p'),
              ('MXINT8-TC4 Q4K4', 'format_mxint8_tc4_q4k4_topp_p'),
              ('MXINT8-TC4 Q8K4', 'format_mxint8_tc4_q8k4_topp_p'),
              ('RS Q4K4', 'format_rsint8_q4k4_topp_p'), ('RS Q8K4', 'format_rsint8_q8k4_topp_p')]
    fp4 = [('MXFP4 Q4K4', 'format_mxfp4_q4k4_topp_p'),
           ('MXINT8-Q8 / MXFP4-K4', 'format_mxint8q_mxfp4k_q8k4_topp_p')]
    if decision.get('activate_mxfp4'):
        groups += fp4
    ppl_rows = []
    for vid, value in DENSE.items():
        row = lookup.get(vid)
        # Historical dense measurements remain usable with their actual limitation.
        measured = row is not None and row.get('_measured_valid') is True
        if measured and numeric(row['ppl']) != value:
            raise ValueError('Historical dense PPL changed: ' + vid)
        label = 'Dense FP' if vid == 'fp_common' else 'Dense INT'
        ppl_rows.append([label, raw_metric(row,'ppl') if measured else 'historical evidence absent', 'same dense reference' if measured else 'absent', 'same dense reference' if measured else 'absent',
                         ('historical measurement; ' + ('parser gate verified' if accepted(row) else 'no parser-matched full gate')) if measured else 'no validated historical measurement'])
    for label, prefix in groups:
        states = [display(prefix + p) for p in ('095', '090', '085')]
        if prefix.startswith('format_rs'):
            states = [value if value != 'absent' else 'source blocked' for value in states]
        ppl_rows.append([label, *states, 'full corpus' if all(accepted(lookup.get(prefix+p, {})) for p in ('095','090','085')) else 'see cell status'])
    full_rows = []
    for vid, row in sorted(lookup.items()):
        ok = accepted(row)
        historical = vid in DENSE and row.get('_measured_valid') is True
        show = ok or historical
        full_rows.append([vid, row.get('status', 'absent'), 'yes' if ok else 'no',
                          raw_metric(row, 'sum_nll') if show else 'unavailable',
                          row.get('num_targets', 'unavailable') if show else 'unavailable',
                          raw_metric(row, 'ppl') if show else 'unavailable',
                          row.get('selection_selected_fraction') if ok else 'unavailable',
                          row.get('verification_error') or row.get('reason', ''), row.get('run', ''),
                          str(Path(row['run'])/'metrics.json') if row.get('run') else '',
                          str(Path(row['run'])/'status.json') if row.get('run') else '',row.get('_gate_path',''),row.get('_pairing_path','')])
    pairing = []
    # The closure also contains historical control pairings with other schemas.
    # Render only the manifests selected and strictly validated for accepted
    # format rows by load_bundle; all historical evidence remains hash-bound.
    pair_paths = {Path(row['_pairing_path']) for row in rows
                  if accepted(row) and row['variant_id'].startswith('format_')}
    for path in sorted(pair_paths):
        document = read(path)
        check_pairing(document)
        evidence.maps(document)
        for role, summary in document['summary'].items():
            pairing.append([document['candidate_variant_id'],
                            'same-input estimator' if role == 'own_source_int_full' else 'cross-run trajectory', role,
                            *[f"{summary[k]['mean']:.8f}" for k in ('jaccard','precision','recall','retained_mass','shortfall')],
                            str(path)])
    diagnostics = [study/'diagnosis/REPORT.md', study/'diagnosis-distributions/REPORT.md']
    for path in diagnostics:
        evidence.bind(path)
    evidence.bind(study/'supplement-proposal/SPEC.md')
    if not test_data:
        index = study/'diagnosis/artifact_index.json'
        for name, entry in read(evidence.bind(index))['files'].items():
            evidence.bind(index.parent/name, entry['sha256'])
        evidence.maps(read(evidence.bind(study/'diagnosis-distributions/manifest.json')))
    prefix = '# TEST DATA ONLY — NOT PUBLISHED RESULTS\n\n' if test_data else '# Format study final report\n\n'
    text = prefix + ('Llama 3.1-8B-Instruct, BF16 model, frozen WikiText-2 scoring: 289,075 targets per accepted full run. '
        'All format predictors select once with Top-p; final selected scores gather the original INT-Flash full Q8/K8 and scales. '
        'The predictor comparison does not measure a native INT kernel speedup.\n\n')
    text += table(['Method','p=.95 PPL','p=.90 PPL','p=.85 PPL','Evidence'],ppl_rows) + '\n\n'
    text += ('Dense references are historical measured values; their absence of a parser-matched release gate must not be confused with verified format evidence. '
             'Missing, blocked and unverified format cells have no reported numeric quality result.\n\n')
    text += 'The frozen conditional decision is:\n\n```json\n' + json.dumps(decision, indent=2, sort_keys=True) + '\n```\n\n'
    text += ('The primary Q4K4 rule uses the equal mean of three unrounded NLL/target values: activate MXFP4 when MXINT8 has no improvement over matched TC, '
             'or exceeds matched INT Top-p by more than log(1.01) CE. Q8K4 is secondary. This writer reports the archived decision and does not revise the trigger. '
             'Conditional FP4 evidence is follow-up evidence. ' + ('All six conditional cells were activated.' if decision.get('activate_mxfp4') else 'FP4 was not activated.') + '\n\n')
    text += ('MX adds block32 E8M0 sidecar metadata: full MXINT8 is 8.25 logical bits/element and a retained four-bit predictor is 4.25. '
             'Native MXFP4 is 4.25; conditional Q8K4 uses MXINT8 full-Q8 (8.25) with MXFP4 K4 (4.25). '
             'These sidecars originate from original INT DQ values, introduce extra quantization/scale confounds, and do not replace the original final operands.\n\n')
    text += '## Paired-set evidence\n\n'
    text += ('Each accepted pairing manifest covers 192 same-input head comparisons and 576 cross-run comparisons (192 each against INT, FP and TC), from six smoke traces. '
             'The table reports mean same-input agreement against original INT-full on candidate operands; complete cross-run summaries are in PAIRED_SETS.csv.\n\n')
    text += table(['Candidate','Mean Jaccard','Mean retained INT mass','Mean mass shortfall'],
                  [[r[0],r[3],r[6],r[7]] for r in pairing if r[1]=='same-input estimator']) + '\n\n'
    text += '## Causal update\n\n'
    text += ('The existing ' + link(diagnostics[0]) + ' and ' + link(diagnostics[1]) +
             ' establish one-sided TC truncation, omitted mixed terms, relative-logit distortion and missed high-mass keys in captured smoke heads. '
             'A removable common logit bias alone does not explain selection error; mixed terms can cancel. '
             'More selected keys do not guarantee better selection. These observations do not attribute fractions of corpus NLL to individual terms.\n\n')
    for label, prefix in groups:
        if not prefix.startswith('format_') or prefix.startswith('format_rs'):
            continue
        changes = []
        width = 8 if 'q8k4' in prefix else 4
        for p in ('095','090','085'):
            a,b,c = lookup.get(prefix+p,{}),lookup.get(f'msb_q{width}k4_topp_p'+p,{}),lookup.get('int_topp_p'+p,{})
            if all(accepted(r) for r in (a,b,c)):
                changes.append(f"p=.{p[1:]}: ΔCE versus TC={numeric(a['sum_nll'])/TARGETS-numeric(b['sum_nll'])/TARGETS:+.9f}, versus INT={numeric(a['sum_nll'])/TARGETS-numeric(c['sum_nll'])/TARGETS:+.9f}")
        text += label + ': ' + ('; '.join(changes) if changes else 'no accepted full comparison available') + '.\n\n'
    text += ('The per-threshold changes above preserve heterogeneous outcomes. Same-input pairing compares predictors on fixed candidate operands; '
             'cross-run comparisons also include upstream trajectory divergence. Neither smoke scope supplies independent corpus samples or a counterfactual full-model intervention. '
             'SM changes reconstruction behavior; MX additionally changes sidecar scales and requantization. A quality gain or loss alone cannot uniquely identify which mechanism caused it. '
             'Other causal interventions remain proposed in ' + link(study/'supplement-proposal/SPEC.md') + ' and require user approval.\n\n')
    text += '## Provenance and artifacts\n\n'
    text += f"Frozen execution Git HEAD: `{freeze['head']}`. Frozen source identity: `{freeze['code_sha256']}`. "
    try:
        head = subprocess.check_output(['git','-C',freeze['source_root'],'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = 'unavailable'
    text += f'Live checkout HEAD at report generation: `{head}`. Documentation commits do not redefine the recorded execution source hash.\n\n'
    text += ('Full NLL/targets and status: [FULL_TABLE.csv](FULL_TABLE.csv). Smoke paired sets: [PAIRED_SETS.csv](PAIRED_SETS.csv). '
             'Every rebound input and generated output is indexed with an absolute artifact path and SHA256 in [DELIVERY.manifest.json](DELIVERY.manifest.json).\n\n')
    text += table(['Artifact','Absolute path'], [[name,str((root/name).resolve())] for name in ('BASELINE.csv','BASELINE.manifest.json','BASELINE.decision.json','freeze/freeze.json','freeze/code_manifest.json')]) + '\n'
    evidence.closure()
    # Create outputs only after evidence validation and report construction succeed.
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Delivery output must be outside frozen execution root')
    output.mkdir(parents=True,exist_ok=False)
    (output/'REPORT.md').write_text(text)
    for filename, headers, values in [
        ('PPL_TABLE.csv',['method','p095_ppl_or_status','p090_ppl_or_status','p085_ppl_or_status','evidence'],ppl_rows),
        ('FULL_TABLE.csv',['variant_id','status','verified','sum_nll','num_targets','ppl','selected_fraction','verification_note','run','metrics_path','status_path','gate_path','pairing_path'],full_rows),
        ('PAIRED_SETS.csv',['variant_id','comparison_scope','reference','mean_jaccard','mean_precision','mean_recall','mean_retained_mass','mean_shortfall','manifest'],pairing)]:
        with (output/filename).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(headers);writer.writerows(values)
    generated = {str(p.resolve()):sha(p) for p in output.iterdir() if p.is_file()}
    (output/'DELIVERY.manifest.json').write_text(json.dumps({'schema':'stp-format-delivery-v1','test_data_only':test_data,'source_files':evidence.files,'artifacts':generated,'writer_sha256':sha(__file__)},indent=2,sort_keys=True)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-root',type=Path,required=True)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true',help='Validate closed summary evidence without writing a report')
    args = parser.parse_args()
    try:
        if args.check_only:
            evidence=Evidence();load_bundle(args.execution_root.resolve(),evidence)
            print(f'Validated {len(evidence.files)} evidence files; no outputs written')
        else:
            print(build(args.execution_root.resolve(),args.study_root.resolve(),args.output_dir))
    except (OSError,ValueError,KeyError,TypeError) as error:
        print('Report not generated: '+str(error),file=sys.stderr);return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
