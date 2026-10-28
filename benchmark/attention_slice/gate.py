"""SLICE-v2 evidence gate: bind all four real-GPU smoke runs to full execution."""
from pathlib import Path
import json
import math
from benchmark.attention_baseline.common import (load_json, write_json, canonical_hash, sha256_file,
    load_config, code_provenance, configure_runtime, environment, validate_assets, load_tokens)
from benchmark.attention_baseline.compare import compare_runs
from benchmark.attention_baseline.gate import validate_quantization_evidence

from .formal import matrix_variants, validate_formal_config, variants_for_p

REQUIRED = variants_for_p(.95)


def read_matrix(path):
    path = Path(path).resolve()
    matrix = load_json(path)
    rows = matrix['variants']
    p, variants = matrix_variants(row['variant_id'] for row in rows)
    if 'top_p' in matrix and (type(matrix['top_p']) not in (float, int) or matrix['top_p'] != p):
        raise ValueError('Matrix TopP threshold/variant IDs mismatch')
    if not all(row.get('required') is True for row in rows):
        raise ValueError('SLICE-v2 requires all four variants in the fixed order')
    configs = {}
    for row in rows:
        cfgpath = (path.parent / row['config']).resolve()
        if not cfgpath.is_relative_to(path.parent): raise ValueError('Config escapes handoff')
        cfg = load_config(cfgpath)
        s = cfg['attention_slice']
        if (s['variant_id'],s['baseline_name']) != (row['variant_id'],row['baseline_name']):
            raise ValueError('Matrix/config identity mismatch')
        validate_formal_config(cfg)
        configs[row['variant_id']] = cfg
    expected = [('fp_common','fp_full',None,None),('intflash_qk8','int_full',8,8),
                ('intflash_qk8','tc_msb',4,4),('intflash_qk8','tc_msb',8,4)]
    for cfg, values in zip(configs.values(), expected):
        s=cfg['attention_slice']
        if (cfg['attention_quantization']['method'],s['estimator'],s['q_prefix_bits'],s['k_prefix_bits']) != values:
            raise ValueError('Formal variant arithmetic differs from SLICE-v2 matrix')
    if (matrix.get('expected_trace_records_per_variant'), matrix.get('expected_trace_records_total'), matrix.get('expected_heads_per_record')) != (6,24,32):
        raise ValueError('Trace count contract differs')
    if not matrix.get('require_pair_sets') or not matrix.get('require_same_input_reference'):
        raise ValueError('Both pairing modes are required')
    return matrix, configs


def check_run(path, config, profile, mode, identities, targets):
    path=Path(path)
    if load_json(path/'status.json').get('status')!='complete': raise ValueError(f'Incomplete run: {path}')
    m=load_json(path/'metrics.json'); cov=load_json(path/'coverage.json')
    if (m['profile'],m['mode'],m['num_targets']) != (profile,mode,targets): raise ValueError(f'Run protocol/count mismatch: {path}')
    if m['semantic_config_sha256'] != canonical_hash(config): raise ValueError('Run config hash mismatch')
    resolved=load_json(path/'resolved_config.json')
    if resolved['algorithm']!=config or any(m.get(k)!=v or resolved.get(k)!=v for k,v in {'query_chunk':64,'context_length':2048,'stride':1024,'prefill_tokens':128,'decode_targets':16}.items()):
        raise ValueError('Runtime scoring/attention parameters differ')
    if resolved['dtype']!='bfloat16' or m['model_dtype']!='bfloat16': raise ValueError('Model dtype changed')
    for key,value in identities.items():
        if m.get(key)!=value: raise ValueError(f'{path}: identity mismatch: {key}')
    if not math.isfinite(m['sum_nll']) or not math.isfinite(m['ppl']) or m['ppl']<1: raise ValueError('Invalid NLL/PPL')
    if m['ppl']!=math.exp(m['sum_nll']/targets): raise ValueError('PPL formula mismatch')
    rows=[json.loads(s) for s in (path/('window_metrics.jsonl' if mode=='windows' else 'target_metrics.jsonl')).read_text().splitlines() if s]
    if sum(r['num_targets'] for r in rows)!=targets or sum(r['sum_nll'] for r in rows)!=m['sum_nll']:
        raise ValueError('Per-row NLL/target accounting mismatch')
    if not all('logits_sha256' in r for r in rows): raise ValueError('Missing original logits hashes')
    from benchmark.attention_baseline.common import scoring_ranges, canonical_hash as h
    expected_ranges=scoring_ranges(m['num_input_tokens']) if mode=='windows' else [{'input_query_position':127+i,'predicted_target':128+i} for i in range(16)]
    if cov['scoring_ranges']!=expected_ranges or m['scoring_ranges_sha256']!=h(expected_ranges): raise ValueError('Label coverage differs')
    if mode=='windows':
        if len(rows)!=len(expected_ranges) or any(any(row[k]!=r[k] for k in r) for row,r in zip(rows,expected_ranges)): raise ValueError('Window positions differ')
    elif len(rows)!=16 or any(any(row[k]!=r[k] for k in r) for row,r in zip(rows,expected_ranges)):
        raise ValueError('Decode positions differ')
    expected_calls={'prefill':len(rows)} if mode=='windows' else {'prefill':1,'decode':15}
    method=config['attention_quantization']['method']
    if set(cov['layers'])!={str(i) for i in range(32)} or not cov['all_layers_covered']:
        raise ValueError('Missing attention layers')
    if any(v!={method:expected_calls} for v in cov['layers'].values()): raise ValueError('Layer method/stage coverage differs')
    validate_quantization_evidence(path,method,cov['layers'],cov)
    if config.get('attention_slice'):
        s=config['attention_slice']
        if m['variant_id']!=s['variant_id'] or m['selection_passes']!=1: raise ValueError('Slice execution identity differs')
        stats=load_json(path/'selection_stats.json'); costs=load_json(path/'costs.json')
        if stats.get('selection_passes')!=1 or stats.get('selected_count_min',0)<1 or not 0<stats['selected_count']<=stats['allowed_count']:
            raise ValueError('Invalid selection statistics')
        if stats['actual_pv']!='dense_zero_masked' or costs['actual_pv']!='dense_zero_masked': raise ValueError('PV cost disclosure missing')
        if s['estimator']=='tc_msb' and not 0<costs['selected_full_qk_macs']<=costs['actual_selected_full_qk_macs']:
            raise ValueError('Selected full-QK physical/valid accounting differs')
    return m


def build_gate(smoke_root,matrix_path,model_path,assets):
    root=Path(smoke_root).resolve(); matrix_path=Path(matrix_path).resolve()
    matrix,configs=read_matrix(matrix_path)
    configure_runtime();code=code_provenance();env=environment()
    manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    identities={'code_sha256':code['code_sha256'],'environment_sha256':env['environment_sha256'],
        'asset_manifest_sha256':canonical_hash(manifest),'tokens_sha256':manifest['tokens_sha256']}
    v=load_json(root/'validation/validation.json')
    if v['status']!='passed' or any(v[d]['status']!='passed' for d in ('cpu','gpu')): raise ValueError('CPU/GPU numerical validation failed')
    p, variants = matrix_variants(configs)
    if (v.get('top_p') != p or v.get('variant_ids') != list(variants)
            or v.get('semantic_config_sha256') != {vid:canonical_hash(cfg) for vid,cfg in configs.items()}):
        raise ValueError('Numerical validation threshold/config matrix differs')
    for key in ('code_sha256','environment_sha256'):
        if v.get(key)!=identities[key]: raise ValueError('Numerical validation identity changed: '+key)
    paths=[root/'validation/validation.json']; traces=0
    for method in ('fp_common','intflash_qk8'):
        cfg=load_config(matrix_path.parent/'configs'/('fp_dense_control.json' if method=='fp_common' else 'int_dense_control.json'))
        p=root/f'{method}-windows-control';check_run(p,cfg,'smoke','windows',identities,4095);paths.extend(p.rglob('*'))
    from benchmark.attention_slice.replay import replay_run
    for vid,cfg in configs.items():
        for suffix,mode,num in [('windows','windows',4095),('decode-off','cached_decode',16),('decode-on','cached_decode',16)]:
            p=root/f'{vid}-{suffix}';check_run(p,cfg,'smoke',mode,identities,num);paths.extend(p.rglob('*'))
        off,on=root/f'{vid}-decode-off',root/f'{vid}-decode-on'
        compared=compare_runs(off,on)
        if not compared['passed']: raise ValueError('Trace off/on changed outputs')
        replay=replay_run(on)
        if not replay.get('passed') or replay.get('status')!='complete': raise ValueError('Trace replay failed')
        idx=load_json(on/'trace/index.json')
        if len(idx['records'])!=6: raise ValueError('Missing trace records')
        traces+=len(idx['records'])
    pair=load_json(root/'pairing/pairing_manifest.json')
    if pair.get('status')!='complete' or not pair.get('passed'): raise ValueError('Set pairing did not pass')
    paired=[json.loads(s) for s in (root/'pairing/paired_sets.jsonl').read_text().splitlines() if s]
    kinds={k:sum(r['comparison_kind']==k for r in paired) for k in ('cross_run_trajectory','same_input_estimator')}
    if kinds!={'cross_run_trajectory':768,'same_input_estimator':384} or traces!=24:
        raise ValueError('Incomplete set pairing/trace matrix')
    paths.extend((root/'pairing').rglob('*'))
    paths.extend([matrix_path,matrix_path.parent/'SPEC.md',matrix_path.parent/'PLAN.md',matrix_path.parent/'README.md',matrix_path.parent/'trace_targets.json'])
    paths.extend((matrix_path.parent/'configs').glob('*.json'))
    evidence={str(p.resolve()):sha256_file(p) for p in sorted(set(paths)) if p.is_file()}
    return {'schema_version':'attention-slice-gate-v2','status':'complete','passed':True,'smoke_root':str(root),
        **identities,'matrix_path':str(matrix_path),'matrix_sha256':sha256_file(matrix_path),
        'semantic_config_sha256':{vid:canonical_hash(c) for vid,c in configs.items()},'query_chunk':64,
        'context_length':2048,'stride':1024,'trace_records':traces,'pairing_counts':kinds,'evidence_files':evidence}


def validate_gate_report(report,model_path,assets,config,query_chunk=64):
    report=load_json(report) if not isinstance(report,dict) else report
    if report.get('schema_version')!='attention-slice-gate-v2' or report.get('status')!='complete' or report.get('passed') is not True:
        raise ValueError('SLICE-v2 smoke gate did not pass')
    if query_chunk!=report['query_chunk'] or query_chunk!=64: raise ValueError('Query chunk changed')
    if code_provenance()['code_sha256']!=report['code_sha256']: raise ValueError('Code changed; rerun slice smoke')
    if environment()['environment_sha256']!=report['environment_sha256']: raise ValueError('Environment changed; rerun slice smoke')
    m=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    if canonical_hash(m)!=report['asset_manifest_sha256']: raise ValueError('Model/token assets changed')
    vid=config['attention_slice']['variant_id']
    if canonical_hash(config)!=report['semantic_config_sha256'].get(vid): raise ValueError('Variant config changed')
    matrix,configs=read_matrix(report['matrix_path'])
    if sha256_file(report['matrix_path'])!=report['matrix_sha256'] or config!=configs[vid]: raise ValueError('Matrix changed')
    for name,expected in report['evidence_files'].items():
        if not Path(name).is_file() or sha256_file(name)!=expected: raise ValueError('Smoke evidence changed: '+name)
    return report
