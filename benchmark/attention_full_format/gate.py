"""Fail-closed full-format smoke gate, binding reviewed source and all evidence."""
import argparse
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,canonical_hash,sha256_file,code_provenance,configure_runtime,environment,validate_assets
from benchmark.attention_baseline.compare import compare_runs
from benchmark.attention_slice.gate import check_run
from .formal import read_matrix,validate_formal_config,TARGET_HASH
from .oracle import check,ATOL,RTOL

SCHEMA='attention-full-format-gate-v1'


def check_full_run(path,cfg,profile,mode,identities,targets):
    m=check_run(path,cfg,profile,mode,identities,targets)
    s=cfg['attention_full_format'];check((m['variant_id'],m['selection_mode'],m['selection_passes'],m['final_operand_origin'],m['final_operand_profile'])==(s['variant_id'],'top_p',1,'stored_format_bytes','decoded_full_format'),'actual full-format execution metadata')
    manifest=load_json(Path(path)/'full_format_manifest.json')
    check(manifest['attention_full_format']==s and manifest['selection_passes']==1 and manifest['final_operand_profile']=='decoded_full_format' and manifest['final_operand_origin']=='stored_format_bytes','dense method manifest')
    stats=load_json(Path(path)/'selection_stats.json');costs=load_json(Path(path)/'costs.json')
    check(stats['selection_passes']==costs['selection_passes']==1 and stats['selection_mode']==costs['selection_mode']=='top_p' and stats['final_operand_profile']==costs['final_operand_profile']=='decoded_full_format','nonzero dense selector count')
    check(0<stats['selected_count']<=stats['allowed_count'] and 0<stats['selected_fraction']<=1. and stats['selected_count_min']>=1,'dense execution omitted legal keys')
    check(costs['full_qk_macs']>0 and costs['actual_pv_macs']>0,'dense operation evidence missing')
    if profile=='full':check(len(load_json(Path(path)/'coverage.json')['scoring_ranges'])==282 and targets==289075,'full frozen coverage')
    return m


def build_gate(smoke_root,matrix_path,model_path,assets,review_report):
    root=Path(smoke_root).resolve();matrix_path=Path(matrix_path).resolve();matrix,configs=read_matrix(matrix_path)
    configure_runtime();code=code_provenance();env=environment();manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    identities=dict(code_sha256=code['code_sha256'],environment_sha256=env['environment_sha256'],asset_manifest_sha256=canonical_hash(manifest),tokens_sha256=manifest['tokens_sha256'])
    review_path=Path(review_report).resolve();review=load_json(review_path)
    check(review.get('status')=='complete' and review.get('passed') is True and review.get('code_sha256')==code['code_sha256'],'independent implementation acceptance missing/stale')
    check(review.get('independent_reviewer') is True,'independent implementation reviewer required')
    numerical=load_json(root/'validation/validation.json');hashes={v:canonical_hash(c) for v,c in configs.items()}
    check(numerical.get('status')=='passed' and numerical['cpu']['status']==numerical['gpu']['status']=='passed','CPU/CUDA codec/attention validation incomplete')
    check(numerical['variant_ids']==list(configs) and numerical['semantic_config_sha256']==hashes,'numerical matrix/config identity')
    for name in ('code_sha256','environment_sha256'):check(numerical[name]==identities[name],'numerical provenance changed '+name)
    check((numerical['atol'],numerical['rtol'])==(ATOL,RTOL),'numerical tolerances changed')
    evidence=[matrix_path,review_path,root/'validation/validation.json']
    regression_path=root/'regression/report.json';regression=load_json(regression_path)
    check(regression.get('passed') is True and regression.get('status')=='complete','preserved original dispatch regression incomplete')
    check(regression['candidate_code_sha256']==code['code_sha256'] and regression['candidate_environment_sha256']==env['environment_sha256'],'regression identity changed')
    check(len(regression['controls'])==6 and all(r['exact_logits_and_nll'] is True and r['num_rows']>0 for r in regression['controls']),'six small preserved controls required')
    for name,digest in regression['source_files'].items():check(sha256_file(name)==digest,'regression input changed');evidence.append(Path(name))
    evidence.append(regression_path);reports={};traces=0
    for vid,cfg in configs.items():
        for suffix,mode,targets in [('windows','windows',4095),('decode-off','cached_decode',16),('decode-on','cached_decode',16)]:
            run=root/(vid+'-'+suffix);check_full_run(run,cfg,'smoke',mode,identities,targets);evidence.extend(p for p in run.rglob('*') if p.is_file())
        off,on=root/(vid+'-decode-off'),root/(vid+'-decode-on')
        compare=compare_runs(off,on);check(compare['passed'] and compare['exact_logits_and_nll'],'trace visibility changed exact logits/NLL')
        replay=load_json(on/'replay.json');check(replay['schema_version']=='attention-full-format-replay-v1' and replay['passed'] is True and replay['status']=='complete','independent replay incomplete')
        check(replay['num_records']==6 and replay['model_weights_replayed'] is True and replay['device']=='cuda','real trace/model replay incomplete')
        check(replay['index_sha256']==sha256_file(on/'trace/index.json'),'replay input changed')
        check((replay['atol'],replay['rtol'])==(ATOL,RTOL),'replay tolerances changed')
        check(all(r['passed'] and r['byte_decode_exact'] and r['actual_support_exact'] and r['regenerated_support_exact'] and r['retained_scores_exact'] and r['selection_passes']==1 and r['final_operand_profile']=='decoded_full_format' and r['model_row_replay']['status']=='passed' for r in replay['records']),'per-record replay acceptance incomplete')
        # Re-read hashes/target coverage/set listing, without rerunning model work.
        from .replay import load_trace_records
        records=load_trace_records(on);traces+=len(records)
        check(load_json(on/'metrics.json')['capture_config_sha256']==TARGET_HASH,'trace target JSON binding')
        reports[vid]=dict(compare=compare,replay_sha256=sha256_file(on/'replay.json'),index_sha256=replay['index_sha256'])
    check(traces==36,'36 actual full-format traces required')
    pairing_path=root/'pairing/pairing_manifest.json';pair=load_json(pairing_path)
    check(pair.get('status')=='complete' and pair.get('passed') is True and pair.get('estimands_separate') is True,'diagnostic pairing incomplete')
    expected_cross=sum(pair['cross_run_pairs'].values())
    check(expected_cross>=576 and all(n==192 for n in pair['cross_run_pairs'].values()),'cross-run pair coverage differs')
    check(pair['counts']==pair['expected_counts']==dict(cross_run_trajectory=expected_cross,same_input=2304),'same-input/cross-run pair coverage differs')
    for name,digest in {**pair['source_files'],**pair['artifacts']}.items():check(sha256_file(name)==digest,'pair evidence changed '+name);evidence.append(Path(name))
    evidence.append(pairing_path)
    for row in matrix['variants']:evidence.append(matrix_path.parent/row['config'])
    evidence.append((matrix_path.parent/matrix['trace_targets']).resolve());evidence.extend(Path(name) for name in matrix['specification_files'])
    files={str(path.resolve()):sha256_file(path) for path in sorted(set(evidence)) if path.is_file()}
    return dict(schema_version=SCHEMA,status='complete',passed=True,smoke_root=str(root),**identities,head=code['head'],source_files=code['files'],matrix_path=str(matrix_path),matrix_sha256=sha256_file(matrix_path),semantic_config_sha256=hashes,query_chunk=64,context_length=2048,stride=1024,trace_records=traces,selection_mode='top_p',selection_passes=1,final_operand_profile='decoded_full_format',actual_masks_used_in_forward=True,pairing_counts=pair['counts'],replay_reports=reports,evidence_files=files,independent_review=dict(path=str(review_path),sha256=sha256_file(review_path)))


def validate_gate_report(report,model_path,assets,config,query_chunk=64):
    r=load_json(report) if not isinstance(report,dict) else report
    check(r.get('schema_version')==SCHEMA and r.get('status')=='complete' and r.get('passed') is True,'full-format release gate not passed')
    check(r['selection_mode']=='top_p' and r['selection_passes']==1 and r['trace_records']==36 and r['final_operand_profile']=='decoded_full_format','dense release semantics/coverage')
    configure_runtime();code=code_provenance();env=environment()
    check(code['head']==r['head'] and code['code_sha256']==r['code_sha256'] and code['files']==r['source_files'],'source freeze changed')
    check(query_chunk==r['query_chunk']==64 and env['environment_sha256']==r['environment_sha256'],'runtime environment/chunk changed')
    manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    check(canonical_hash(manifest)==r['asset_manifest_sha256'] and manifest['tokens_sha256']==r['tokens_sha256'],'model/assets changed')
    validate_formal_config(config);vid=config['attention_full_format']['variant_id'];check(canonical_hash(config)==r['semantic_config_sha256'].get(vid),'dense config changed')
    _,configs=read_matrix(r['matrix_path']);check(config==configs[vid] and sha256_file(r['matrix_path'])==r['matrix_sha256'],'dense matrix changed')
    for name,digest in r['evidence_files'].items():check(Path(name).is_file() and sha256_file(name)==digest,'smoke evidence changed '+name)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('smoke-root','matrix','model','assets','review-report','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();check(not Path(a.output).exists(),'gate outputs are immutable')
    try:report=build_gate(a.smoke_root,a.matrix,a.model,a.assets,a.review_report)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
