"""Report all four official variants, retaining missing and failed evidence."""
import argparse
import csv
import math
from pathlib import Path

from benchmark.attention_baseline.common import canonical_hash, load_json, sha256_file, write_json
from .replay import check
from .gate import read_matrix


def comparable(left, right):
    for key in ('protocol_id', 'tokens_sha256', 'subset_sha256', 'scoring_ranges_sha256',
                'asset_manifest_sha256', 'num_targets', 'num_input_tokens', 'model_dtype', 'backend'):
        check(key in left and key in right and left[key] == right[key], 'incompatible metric identity: ' + key)


def valid_metrics(metrics, mode):
    nll, targets, ppl = metrics['sum_nll'], metrics['num_targets'], metrics['ppl']
    check(isinstance(targets, int) and targets > 0 and math.isfinite(nll) and math.isfinite(ppl)
          and nll >= 0 and ppl >= 1, 'invalid completed metrics')
    check(math.isclose(ppl, math.exp(nll / targets), rel_tol=1e-8, abs_tol=1e-8), 'PPL/NLL inconsistent')
    if mode == 'windows':
        check(targets == metrics['num_input_tokens'] - 1, 'incomplete target coverage')


def summarize(matrix, runs_root, output):
    spec, configs = read_matrix(matrix)
    variant_ids = list(configs)
    p = configs[variant_ids[0]]['attention_slice']['selection']['p']
    variants = [v for v in spec['variants'] if v['required']]
    groups, captures, controls, source_files = {}, {}, {}, {}
    for status_path in sorted(Path(runs_root).rglob('status.json')):
        run = status_path.parent
        cp = run / 'resolved_config.json'
        if not cp.exists():
            continue
        config = load_json(cp)
        algorithm = config.get('algorithm', config)
        if 'attention_quantization' not in algorithm:
            continue
        status = load_json(status_path)
        mp = run / 'metrics.json'
        metrics = load_json(mp) if mp.exists() else {}
        mode = metrics.get('mode', 'cached_decode' if 'decode' in run.name else 'windows')
        profile = metrics.get('profile', 'full' if 'full' in run.name else 'smoke')
        sl = algorithm.get('attention_slice')
        variant = sl['variant_id'] if sl else algorithm['attention_quantization']['method']
        if sl:
            check(variant in configs and algorithm == configs[variant], 'run/matrix config mismatch')
        key = (profile, mode, variant)
        capture = sl is not None and ((run / 'trace').exists() or bool(config.get('trace_targets')) or 'decode-on' in run.name)
        container = captures if capture else groups if sl else controls
        check(key not in container, 'duplicate summary run: ' + str(key))
        container[key] = (run, status, metrics, config)
        for path in (status_path, cp, mp, run / 'coverage.json', run / 'selection_stats.json', run / 'costs.json'):
            if path.exists():
                source_files[str(path.resolve())] = sha256_file(path)
    protocols = sorted({key[:2] for key in groups} | {key[:2] for key in captures})
    if not protocols:
        protocols = [('full', 'windows'), ('smoke', 'windows'), ('smoke', 'cached_decode')]
    rows = []
    for profile, mode in protocols:
        for variant in variants:
            vid = variant['variant_id']
            bq, bk = (None, None) if vid == variant_ids[0] else (8, 8) if vid == variant_ids[1] else (4, 4) if vid == variant_ids[2] else (8, 4)
            row = {'profile': profile, 'mode': mode, 'variant_id': vid, 'baseline_name': variant['baseline_name'],
                   'q_prefix_bits': bq, 'k_prefix_bits': bk, 'p': p, 'status': 'not_run'}
            data = groups.get((profile, mode, vid))
            capture = captures.get((profile, mode, vid))
            if capture:
                row.update(capture_run=str(capture[0].resolve()), capture_status=capture[1].get('status', 'failed'),
                           capture_reason=capture[1].get('reason', ''))
            if data:
                run, status, metrics, config = data
                row.update(run=str(run.resolve()), status=status.get('status', 'failed'), reason=status.get('reason', ''))
                if row['status'] == 'complete':
                    valid_metrics(metrics, mode)
                    check(metrics['variant_id'] == vid and metrics['baseline_name'] == variant['baseline_name'], 'metric variant/name mismatch')
                    for name in ('sum_nll', 'num_targets', 'ppl', 'num_input_tokens', 'protocol_id',
                                 'tokens_sha256', 'asset_manifest_sha256', 'scoring_ranges_sha256', 'code_sha256', 'environment_sha256'):
                        if name in metrics:
                            row[name] = metrics[name]
                    # Comparing each MSB to INT preserves the intended interpretation.
                    ref_id = variant_ids[1] if vid in variant_ids[2:] else variant_ids[0]
                    ref = groups.get((profile, mode, ref_id))
                    row['reference_variant_id'] = ref_id
                    if ref and ref[1].get('status') == 'complete':
                        comparable(metrics, ref[2])
                        row['delta_nll_vs_reference'] = metrics['sum_nll'] - ref[2]['sum_nll']
                        row['delta_ppl_vs_reference'] = metrics['ppl'] - ref[2]['ppl']
                    dense_method = 'fp_common' if vid == variant_ids[0] else 'intflash_qk8'
                    dense = controls.get((profile, mode, dense_method))
                    if dense:
                        row.update(dense_baseline_run=str(dense[0].resolve()), dense_baseline_status=dense[1].get('status', 'failed'))
                        if dense[1].get('status') == 'complete':
                            valid_metrics(dense[2], mode)
                            comparable(metrics, dense[2])
                            row.update(dense_baseline_sum_nll=dense[2]['sum_nll'], dense_baseline_num_targets=dense[2]['num_targets'],
                                       dense_baseline_ppl=dense[2]['ppl'], delta_nll_vs_dense=metrics['sum_nll'] - dense[2]['sum_nll'],
                                       delta_ppl_vs_dense=metrics['ppl'] - dense[2]['ppl'])
                    stats_path = run / 'selection_stats.json'
                    if stats_path.exists():
                        stats = load_json(stats_path)
                        for name in ('num_rows', 'selected_count', 'allowed_count', 'mean_num_selected', 'selected_fraction'):
                            if name in stats:
                                row['selection_' + name] = stats[name]
            rows.append(row)
    pair_manifests = list(Path(runs_root).rglob('pairing_manifest.json'))
    set_rows = []
    for manifest_path in pair_manifests:
        manifest = load_json(manifest_path)
        if manifest.get('status') != 'complete' or not manifest.get('passed'):
            continue
        sources = manifest.get('artifacts', {})
        pairs_path = manifest_path.parent / 'paired_sets.jsonl'
        check(sources.get(str(pairs_path.resolve())) == sha256_file(pairs_path), 'paired rows hash mismatch')
        source_files[str(manifest_path.resolve())] = sha256_file(manifest_path)
        source_files[str(pairs_path.resolve())] = sha256_file(pairs_path)
        set_rows.extend(__import__('json').loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
    for row in rows:
        # Trace-set measurements are explicitly separate from corpus aggregates.
        if row['profile'] != 'smoke':
            continue
        for kind, refid, label in [('cross_run_trajectory', variant_ids[0], 'cross_run_fp'),
                                   ('cross_run_trajectory', variant_ids[1], 'cross_run_int'),
                                   ('same_input_estimator', 'int_top_p_same_input_diagnostic', 'same_input_int')]:
            matches = [p for p in set_rows if p['predicted_variant_id'] == row['variant_id'] and p['comparison_kind'] == kind and p['reference_variant_id'] == refid]
            if matches:
                check(len(matches) == 192, 'duplicate/incomplete set summary')
                row[label + '_trace_heads'] = len(matches)
                for metric in ('recall', 'precision', 'jaccard', 'num_predicted', 'num_reference', 'retained_mass', 'shortfall'):
                    row[label + '_mean_' + metric] = sum(p[metric] for p in matches) / len(matches)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = ['profile', 'mode', 'variant_id', 'baseline_name', 'q_prefix_bits', 'k_prefix_bits', 'p', 'status', 'run', 'reason', 'sum_nll', 'num_targets', 'ppl']
    names = first + sorted(set().union(*(set(row) for row in rows)) - set(first))
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out.with_suffix(out.suffix + '.manifest.json'), {
        'schema_version': 'slice-summary-v2', 'matrix_sha256': sha256_file(matrix),
        'source_files': source_files, 'output_sha256': sha256_file(out),
        'methodology': {'quality_metrics_are_not_correctness_gates': True,
                        'decode_capture': 'off measurement reported separately from on capture status',
                        'set_statistics': '192 target/head measurements per predictor and reference; not corpus estimates',
                        'dense_baselines': 'only actual compatible recorded runs; missing historical reports remain absent',
                        'selection': f'single p={p:g} top-p; full precision within fixed S; four variants remain separate'},
        'num_rows': len(rows)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--runs-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        summarize(args.matrix, args.runs_root, args.output)
    except Exception as exc:
        write_json(str(args.output) + '.failure.json', {'status': 'failed', 'reason': str(exc)})
        raise


if __name__ == '__main__':
    main()
