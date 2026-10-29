"""Fail-closed RS release gate: ten jobs, sixty actual traces, both estimands."""
import argparse
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,canonical_hash,sha256_file,code_provenance,configure_runtime,environment,validate_assets
from benchmark.attention_baseline.compare import compare_runs
from benchmark.attention_slice.gate import check_run
from .formal import read_matrix,validate_formal_config,TARGET_HASH
from .oracle import check,ATOL,RTOL
SCHEMA='attention-rs-gate-v1'

def check_rs_run(path,cfg,profile,mode,identities,targets):
    path=Path(path);m=check_run(path,cfg,profile,mode,identities,targets);s=cfg['attention_rs'];manifest=load_json(path/'rs_manifest.json');stats=load_json(path/'selection_stats.json');costs=load_json(path/'costs.json')
    for key in ('variant_id','mode','selection_passes','final_operand_origin','final_scores','final_execution'):
        # The scorer metrics mode is protocol mode; attention mode lives inside attention_rs.
        if key=='mode':check(m['attention_rs']['mode']==s[key],'RS attention mode')
        else:check(m[key]==manifest[key]==s[key],'RS actual mode/origin '+key)
    check(manifest['attention_rs']==s and m['attention_rs']==s,'RS method manifest binding')
    check(stats['selection_passes']==costs['selection_passes']==s['selection_passes'] and stats['selected_count_min']>=1,'RS selector count')
    check(0<stats['selected_count']<=stats['allowed_count'] and stats['actual_pv']==costs['actual_pv']=='dense_zero_masked','RS selected/PV accounting')
    if s['mode']=='dense_full':check(stats['selected_count']==stats['allowed_count'] and stats['selected_fraction']==1.,'RS dense all-legal execution')
    if s['mode']=='msb_topp':check(0<costs['selected_full_qk_macs']<=costs['actual_selected_full_qk_macs'],'MSB valid/padded source-QK work')
    else:check(costs['selected_full_qk_macs']==costs['actual_selected_full_qk_macs']==0,'native final must not recompute original-source scores')
    check(costs['estimator_qk_macs']>0 and costs['actual_pv_macs']>0,'RS actual work missing')
    check(all(stats.get(key,0)>0 for key in ('rs_source_clamp_elements','rs_encode_elements','rs_full_decode_elements','rs_prefix_extract_elements','rs_prefix_decode_elements','rs_saturation_check_elements')),'RS conversion work missing')
    saturation=load_json(path/'rs_statistics.json');by=saturation['by_layer_operand'];check(set(by)=={str(layer)+':'+operand for layer in range(32) for operand in ('q','k')},'all-layer/operand saturation coverage')
    import math
    for group in by.values():
        check(len(group['denominator'])==32 and all(value>0 for value in group['denominator']),'per-head saturation denominator')
        check(all(0<=sat<=den for sat,den in zip(group['saturation_count'],group['denominator'])),'saturation fraction')
        hist=[group['source_code_'+str(value)+'_count'] for value in range(120,128)];deltas=[group['conversion_delta_'+str(value)+'_count'] for value in range(1,9)]
        check([sum(x) for x in zip(*hist)]==group['saturation_count'] and hist==deltas,'saturation histogram/delta accounting')
        for field,values in group.items():
            if '_histogram' in field:continue
            if isinstance(values,list):check(all(math.isfinite(value) for value in values),'nonfinite RS reduction '+field)
    if profile=='full':check(len(load_json(path/'coverage.json')['scoring_ranges'])==282 and targets==289075,'RS full scoring protocol')
    return m

def build_gate(smoke_root,matrix_path,model_path,assets,review_report):
    root=Path(smoke_root).resolve();matrix_path=Path(matrix_path).resolve();matrix,configs=read_matrix(matrix_path);configure_runtime();code=code_provenance();env=environment();manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    identities=dict(code_sha256=code['code_sha256'],environment_sha256=env['environment_sha256'],asset_manifest_sha256=canonical_hash(manifest),tokens_sha256=manifest['tokens_sha256']);review_path=Path(review_report).resolve();review=load_json(review_path)
    check(review.get('status')=='complete' and review.get('passed') is True and review.get('independent_reviewer') is True and review.get('code_sha256')==code['code_sha256'],'RS independent implementation acceptance missing/stale')
    check(review.get('matrix_sha256')==sha256_file(matrix_path),'RS review matrix binding missing/stale')
    numerical=load_json(root/'validation/validation.json');hashes={vid:canonical_hash(cfg) for vid,cfg in configs.items()}
    check(numerical.get('status')=='passed' and numerical['cpu']['status']==numerical['gpu']['status']=='passed','RS CPU/CUDA numerical acceptance incomplete')
    check(numerical['variant_ids']==list(configs) and numerical['semantic_config_sha256']==hashes and numerical['code_sha256']==code['code_sha256'] and numerical['environment_sha256']==env['environment_sha256'],'RS numerical freeze differs');check((numerical['atol'],numerical['rtol'])==(ATOL,RTOL),'RS tolerance changed')
    regression=load_json(root/'regression/report.json');check(regression.get('status')=='complete' and regression.get('passed') is True and regression['candidate_code_sha256']==code['code_sha256'] and regression['candidate_environment_sha256']==env['environment_sha256'],'RS preserved-path regression incomplete')
    check(len(regression['controls'])==6 and all(row['exact_logits_and_nll'] is True for row in regression['controls']),'six preserved controls required')
    evidence=[matrix_path,review_path,root/'validation/validation.json',root/'regression/report.json'];traces=0;reports={}
    for name,digest in regression['source_files'].items():check(sha256_file(name)==digest,'regression source changed');evidence.append(Path(name))
    from .replay import load_trace_records
    for vid,cfg in configs.items():
        for suffix,mode,targets in [('windows','windows',4095),('decode-off','cached_decode',16),('decode-on','cached_decode',16)]:
            path=root/(vid+'-'+suffix);check_rs_run(path,cfg,'smoke',mode,identities,targets);evidence.extend(file for file in path.rglob('*') if file.is_file())
        off,on=root/(vid+'-decode-off'),root/(vid+'-decode-on');comparison=compare_runs(off,on);check(comparison['passed'] and comparison['exact_logits_and_nll'],'RS tracing changed exact output/NLL')
        check(load_json(off/'rs_statistics.json')==load_json(on/'rs_statistics.json'),'RS trace changed saturation reductions')
        replay=load_json(on/'replay.json');check(replay.get('schema_version')=='attention-rs-replay-v1' and replay.get('status')=='complete' and replay.get('passed') is True,'RS replay incomplete')
        check(replay['num_records']==6 and replay['model_weights_replayed'] is True and replay['byte_decode_exact'] is True and replay['device']=='cuda' and (replay['atol'],replay['rtol'])==(ATOL,RTOL),'RS real CUDA/model replay incomplete')
        check(replay['index_sha256']==sha256_file(on/'trace/index.json'),'RS replay source changed')
        check(all(row['passed'] and row['source_and_bytes_exact'] and row['final_origin_verified'] and row['selection_passes']==cfg['attention_rs']['selection_passes'] and row['model_row_replay']['status']=='passed' for row in replay['records']),'RS per-record acceptance incomplete')
        records=load_trace_records(on);traces+=len(records);check(load_json(on/'metrics.json')['capture_config_sha256']==TARGET_HASH,'RS target binding');reports[vid]=dict(replay_sha256=sha256_file(on/'replay.json'),compare=comparison)
    check(traces==60 and sum(cfg['attention_rs']['selection_passes'] for cfg in configs.values())==9,'RS sixty traces/nine selectors')
    check(sum(cfg['attention_rs']['final_operand_origin']=='rs_full_decoded' for cfg in configs.values())==4,'RS four native/six original final jobs')
    pairing_path=root/'pairing/pairing_manifest.json';pair=load_json(pairing_path);check(pair.get('status')=='complete' and pair.get('passed') is True and pair['estimands_separate'] is True and pair['actual_native_full_pairs']==6,'both RS pairing modes required')
    expected_controls={vid+':'+role for vid,cfg in configs.items() if cfg['attention_rs']['mode']=='msb_topp' for role in ('tc','sm')}
    available=set(pair['historical_controls']);unavailable=set(pair.get('unavailable_dense_controls',{}))
    check(available-{'fp','source_int'}==expected_controls and (available&{'fp','source_int'})|unavailable=={'fp','source_int'},'all compatible TC/SM controls and explicit dense availability required')
    check(pair['counts']==pair['expected_counts']==dict(cross_run_trajectory=(9+len(available))*192,same_input_estimator=7680),'RS pair row coverage')
    for name,digest in {**pair['source_files'],**pair['artifacts']}.items():check(sha256_file(name)==digest,'RS pair evidence changed');evidence.append(Path(name))
    evidence.append(pairing_path);evidence.extend(matrix_path.parent/row['config'] for row in matrix['variants']);evidence.append(matrix_path.parent/matrix['trace_targets']);evidence.extend(Path(name) for name in matrix['specification_files'])
    return dict(schema_version=SCHEMA,status='complete',passed=True,**identities,head=code['head'],source_files=code['files'],matrix_path=str(matrix_path),matrix_sha256=sha256_file(matrix_path),semantic_config_sha256=hashes,query_chunk=64,context_length=2048,stride=1024,trace_records=60,execution_rows=1920,diagnostic_rows=5760,native_final_jobs=4,original_final_jobs=6,zero_selector_jobs=1,one_selector_jobs=9,replay_reports=reports,pairing_counts=pair['counts'],evidence_files={str(path.resolve()):sha256_file(path) for path in sorted(set(evidence)) if path.is_file()})

def validate_gate_report(report,model_path,assets,config,query_chunk=64):
    r=load_json(report) if not isinstance(report,dict) else report;check(r.get('schema_version')==SCHEMA and r.get('status')=='complete' and r.get('passed') is True,'RS release gate not passed')
    check((r['trace_records'],r['execution_rows'],r['native_final_jobs'],r['original_final_jobs'],r['zero_selector_jobs'],r['one_selector_jobs'])==(60,1920,4,6,1,9),'RS release coverage')
    configure_runtime();code=code_provenance();env=environment();check(code['head']==r['head'] and code['code_sha256']==r['code_sha256'] and code['files']==r['source_files'],'RS code freeze changed')
    check(query_chunk==r['query_chunk']==64 and env['environment_sha256']==r['environment_sha256'],'RS runtime changed');manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets);check(canonical_hash(manifest)==r['asset_manifest_sha256'] and manifest['tokens_sha256']==r['tokens_sha256'],'RS model/assets changed')
    validate_formal_config(config);vid=config['attention_rs']['variant_id'];check(canonical_hash(config)==r['semantic_config_sha256'].get(vid),'RS config changed');_,configs=read_matrix(r['matrix_path']);check(config==configs[vid] and sha256_file(r['matrix_path'])==r['matrix_sha256'],'RS matrix changed')
    for name,digest in r['evidence_files'].items():check(Path(name).is_file() and sha256_file(name)==digest,'RS smoke evidence changed '+name)
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('smoke-root','matrix','model','assets','review-report','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args();check(not Path(a.output).exists(),'RS gate output immutable')
    try:report=build_gate(a.smoke_root,a.matrix,a.model,a.assets,a.review_report)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
