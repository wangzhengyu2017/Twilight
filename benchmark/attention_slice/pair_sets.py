"""Pair actual model trajectories and separate same-input estimator evidence."""
import argparse
import csv
import json
from pathlib import Path

from benchmark.attention_baseline.common import canonical_hash, load_json, sha256_file, write_json
from .replay import check, load_trace_records, replay_run
from .gate import read_matrix


def trace_runs(root, variant):
    matches = []
    for index in Path(root).rglob('trace/index.json'):
        run = index.parent.parent
        config = load_json(run / 'resolved_config.json')
        config = config.get('algorithm', config)
        if config.get('attention_slice', {}).get('variant_id') == variant:
            matches.append(run)
    check(len(matches) == 1, f'{variant}: expected one capture run, found {len(matches)}')
    return matches[0]


def identity(meta, actual, head):
    selection = meta.get('selection', {})
    required = ('tokens_sha256', 'model_file_hashes', 'scope', 'sample_id', 'window_id',
                'layer_id', 'stage', 'query_position', 'head_to_kv')
    check(all(key in meta for key in required), 'incomplete pairing identity')
    asset_hash = meta.get('asset_manifest_sha256')
    check(asset_hash is not None, 'missing asset manifest identity')
    return {
        **{key: meta[key] for key in required}, 'asset_manifest_sha256': asset_hash,
        'p': meta.get('p', selection.get('p')), 'tie_break': meta.get('tie_break', selection.get('tie_break')),
        'key_positions': actual['key_positions'].tolist(), 'head_id': head,
    }


def set_metrics(predicted, reference, reference_probs, p):
    intersection = int((predicted & reference).sum())
    union = int((predicted | reference).sum())
    npred, nref = int(predicted.sum()), int(reference.sum())
    check(npred > 0 and nref > 0, 'empty comparison set')
    mass = float(reference_probs[predicted].double().sum())
    reference_mass = float(reference_probs[reference].double().sum())
    return {'num_predicted': npred, 'num_reference': nref, 'num_intersection': intersection,
            'num_union': union, 'recall': intersection / nref, 'precision': intersection / npred,
            'jaccard': intersection / union, 'retained_mass': mass,
            'shortfall': max(0.0, p - mass), 'reference_retained_mass': reference_mass}


def _selection_rows(run, records):
    """Bind the separately consumable set listing to the actual tensor masks."""
    path = run / 'selection_sets.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    check(len(rows) == 192, f'{run}: expected 192 set rows')
    lookup = {(entry['record_id'], head): (entry, meta, actual)
              for entry, meta, actual, _ in records for head in range(32)}
    seen = set()
    for row in rows:
        head = row.get('head_id', row.get('head'))
        key = (row['record_id'], head)
        check(key in lookup and key not in seen, 'unknown/duplicate selection row')
        seen.add(key)
        entry, meta, a = lookup[key]
        check(row['key_positions'] == a['key_positions'].tolist(), 'set listing key domain mismatch')
        selected = a['key_positions'][a['selected_mask'][0, head, 0]].tolist()
        check(row['selected_key_positions'] == selected and row['num_selected'] == len(selected), 'set listing/mask mismatch')
        for name in ('actual_sha256', 'derived_sha256', 'meta_sha256'):
            check(row[name] == entry[name], 'set listing trace hash mismatch')
        for name, value in identity(meta, a, head).items():
            check(row.get(name) == value, 'set listing identity mismatch: ' + name)
        ref = row.get('selected_mask_ref', {})
        check(ref.get('field') == 'selected_mask' and ref.get('sha256') == entry['actual_sha256']
              and ref.get('index') == [0, head, 0], 'invalid selected mask reference')
        check(Path(ref['file']).resolve() == (run / 'trace' / entry.get('path', entry['record_id']) / 'actual.pt').resolve(), 'mask reference points to wrong trace')
    return path


def pair_sets(matrix, runs_root, output_dir):
    matrix_path = Path(matrix).resolve()
    spec, configs = read_matrix(matrix_path)
    variants = list(configs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, groups, reports = [], {}, {}, {}
    for variant in variants:
        run = trace_runs(runs_root, variant).resolve()
        resolved = load_json(run / 'resolved_config.json')
        check(resolved.get('algorithm', resolved) == configs[variant], 'capture/matrix config mismatch')
        reports[variant] = replay_run(run)
        records = load_trace_records(run)
        sets = _selection_rows(run, records)
        index = load_json(run / 'trace/index.json')
        check(index['selection_sets']['sha256'] == sha256_file(sets), 'set listing index hash mismatch')
        for path in [run / name for name in ('status.json', 'metrics.json', 'resolved_config.json',
                    'asset_manifest.json', 'code_manifest.json', 'code.patch', 'environment.json')]:
            if path.exists():
                sources[str(path)] = sha256_file(path)
        sources[str(sets)] = sha256_file(sets)
        sources[str(run / 'trace/index.json')] = sha256_file(run / 'trace/index.json')
        group = {}
        for entry, meta, a, derived in records:
            for filename in ('actual.pt', 'derived.pt', 'meta.json'):
                path = run / 'trace' / entry.get('path', entry['record_id']) / filename
                sources[str(path)] = sha256_file(path)
            for head in range(32):
                ident = identity(meta, a, head)
                key = canonical_hash(ident)
                check(key not in group, 'duplicate pairing identity')
                group[key] = (ident, entry, meta, a, derived, head)
        groups[variant] = group
    for variant in variants[1:]:
        check(set(groups[variant]) == set(groups[variants[0]]), 'cross-run identity/domain mismatch: ' + variant)
    for predicted_id in variants[2:]:
        for key, (ident, pe, pm, pa, pd, head) in groups[predicted_id].items():
            pred = pa['selected_mask'][0, head, 0]
            base = {'predicted_variant_id': predicted_id, 'pair_identity_sha256': key,
                    **ident, 'predicted_record_id': pe['record_id'],
                    'predicted_actual_sha256': pe['actual_sha256'], 'predicted_meta_sha256': pe['meta_sha256'],
                    'predicted_derived_sha256': pe['derived_sha256'],
                    'predicted_selected_key_positions': pa['key_positions'][pred].tolist()}
            for reference_id in variants[:2]:
                _, re, rm, ra, rd, _ = groups[reference_id][key]
                ref = ra['selected_mask'][0, head, 0]
                metrics = set_metrics(pred, ref, ra['estimator_probs'][0, head, 0], ident['p'])
                rows.append({**base, 'comparison_kind': 'cross_run_trajectory',
                             'reference_variant_id': reference_id, 'reference_record_id': re['record_id'],
                             'reference_actual_sha256': re['actual_sha256'], 'reference_meta_sha256': re['meta_sha256'],
                             'reference_derived_sha256': re['derived_sha256'],
                             'reference_selected_key_positions': ra['key_positions'][ref].tolist(), **metrics})
            ref = pd['reference_int_top_p_mask_same_input'][0, head, 0]
            metrics = set_metrics(pred, ref, pd['reference_int_probs_same_input'][0, head, 0], ident['p'])
            rows.append({**base, 'comparison_kind': 'same_input_estimator',
                         'reference_variant_id': 'int_top_p_same_input_diagnostic',
                         'reference_record_id': pe['record_id'], 'reference_actual_sha256': pe['actual_sha256'],
                         'reference_meta_sha256': pe['meta_sha256'], 'reference_derived_sha256': pe['derived_sha256'],
                         'reference_selected_key_positions': pa['key_positions'][ref].tolist(), **metrics})
    counts = {kind: sum(row['comparison_kind'] == kind for row in rows)
              for kind in ('cross_run_trajectory', 'same_input_estimator')}
    check(counts == {'cross_run_trajectory': 768, 'same_input_estimator': 384}, 'pair count mismatch')
    json_path, csv_path = output / 'paired_sets.jsonl', output / 'set_comparison.csv'
    json_path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))
    columns = ['comparison_kind', 'predicted_variant_id', 'reference_variant_id', 'pair_identity_sha256',
               'sample_id', 'window_id', 'layer_id', 'stage', 'query_position', 'head_id', 'p',
               'num_predicted', 'num_reference', 'num_intersection', 'num_union', 'recall', 'precision',
               'jaccard', 'retained_mass', 'shortfall', 'reference_retained_mass',
               'predicted_actual_sha256', 'predicted_derived_sha256', 'predicted_meta_sha256',
               'reference_actual_sha256', 'reference_derived_sha256', 'reference_meta_sha256']
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'status': 'complete', 'passed': True, 'schema_version': 'slice-pairing-v2',
                'matrix': str(matrix_path), 'matrix_sha256': sha256_file(matrix_path),
                'runs_root': str(Path(runs_root).resolve()), 'num_rows': len(rows), 'counts': counts,
                'num_head_identities': 192, 'source_files': sources, 'replay_reports': reports,
                'quality_is_measured_not_gated': True,
                'mass_scope': 'six target traces per variant; reference full distribution; not full-corpus estimates',
                'artifacts': {str(path.resolve()): sha256_file(path) for path in (json_path, csv_path)}}
    write_json(output / 'pairing_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        pair_sets(args.matrix, args.runs_root, args.output_dir)
    except Exception as exc:
        write_json(Path(args.output_dir) / 'pairing_failure.json', {'status': 'failed', 'passed': False, 'reason': str(exc)})
        raise


if __name__ == '__main__':
    main()
