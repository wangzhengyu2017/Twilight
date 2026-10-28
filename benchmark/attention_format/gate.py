"""Format-study release gate; every evidence file is rebound before full execution."""
import argparse
import json
from pathlib import Path
from benchmark.attention_baseline.common import (load_json,write_json,canonical_hash,sha256_file,load_config,code_provenance,configure_runtime,environment,validate_assets)
from benchmark.attention_baseline.compare import compare_runs
from benchmark.attention_slice.gate import check_run
from .formal import validate_formal_config,variant_contract
from .oracle import check


def read_matrix(path):
    path=Path(path).resolve();matrix=load_json(path);configs={}
    check(matrix['study']=='A_common_original_int8_final' and matrix.get('native_mx_full_reference_authorized') is False,'wrong study/final execution')
    check(matrix.get('expected_trace_records_per_variant')==6 and matrix.get('expected_heads_per_record')==32 and matrix.get('require_pair_sets') is True and matrix.get('require_same_input_reference') is True,'trace/pairing coverage contract')
    for row in matrix['variants']:
        if row['predictor_format']=='rsint8':
            check(row['status']=='planned_source_blocked' and 'config' not in row,'RS must remain source blocked');continue
        if not row.get('required'):continue
        cfgpath=(path.parent/row['config']).resolve();check(cfgpath.is_relative_to(path.parent),'unsafe config path')
        config=load_config(cfgpath);fmt,bq,bk,p=validate_formal_config(config)
        check((config['attention_slice']['variant_id'],fmt,bq,bk,p)==(row['variant_id'],row['predictor_format'],row['q_total_prefix_bits'],row['k_total_prefix_bits'],row['p']),'matrix/config mismatch')
        check(row['variant_id'] not in configs,'duplicate variant');configs[row['variant_id']]=config
    expected={f'format_{fmt}_q{bq}k4_topp_p{p}' for fmt in ('sm8','mxint8_tc4') for bq in (4,8) for p in ('095','090','085')}
    if matrix.get('execution_phase','initial')=='conditional':
        activation=matrix.get('conditional_activation',{})
        check(activation.get('trigger_verified') is True,'conditional FP4 requires verified trigger')
        decision=Path(activation['decision_path']).resolve()
        check(sha256_file(decision)==activation['decision_sha256'],'conditional decision changed')
        report=load_json(decision);check(report.get('activate_mxfp4') is True and report.get('status')=='complete','conditional trigger not established')
        manifest_path=Path(activation['decision_manifest_path']).resolve();check(sha256_file(manifest_path)==activation['decision_manifest_sha256'],'conditional source manifest changed')
        decision_manifest=load_json(manifest_path);check(decision_manifest.get('decision_sha256')==activation['decision_sha256'],'conditional decision/source binding')
        check(bool(decision_manifest['source_files']),'conditional source evidence absent')
        for name,digest in decision_manifest['source_files'].items():check(sha256_file(name)==digest,'conditional source changed: '+name)
        conditional_expected={f'format_{fmt}_q{bits}k4_topp_p{p}' for fmt,bits in (('mxfp4',4),('mxint8q_mxfp4k',8)) for p in ('095','090','085')}
        configs={vid:cfg for vid,cfg in configs.items() if vid in conditional_expected}
        check(set(configs)==conditional_expected,'all six conditional cells required')
    else:
        check(matrix.get('execution_phase','initial')=='initial','unknown execution phase')
        check(set(configs)==expected,'exactly twelve initial variants required')
    return matrix,configs


def build_gate(smoke_root,matrix_path,model_path,assets):
    from .replay import replay_run
    root=Path(smoke_root).resolve();matrix_path=Path(matrix_path).resolve();matrix,configs=read_matrix(matrix_path)
    configure_runtime();code=code_provenance();env=environment();manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    identities=dict(code_sha256=code['code_sha256'],environment_sha256=env['environment_sha256'],asset_manifest_sha256=canonical_hash(manifest),tokens_sha256=manifest['tokens_sha256'])
    numerical=load_json(root/'validation/validation.json')
    check(numerical['status']=='passed' and all(numerical[k]['status']=='passed' for k in ('cpu','gpu')),'CPU/GPU numerical checks incomplete')
    check(numerical['variant_ids']==list(configs),'numerical matrix identity')
    hashes={vid:canonical_hash(config) for vid,config in configs.items()}
    check(numerical['semantic_config_sha256']==hashes,'numerical config identity')
    for key in ('code_sha256','environment_sha256'):check(numerical[key]==identities[key],'numerical '+key)
    evidence=[root/'validation/validation.json',matrix_path];traces=0;pair_counts={};replays={}
    regression_path=root/'regression/report.json';regression=load_json(regression_path)
    check(regression.get('passed') is True and regression.get('status')=='complete','original dense/full-TopP regression incomplete')
    check(regression['candidate_code_sha256']==identities['code_sha256'] and regression['candidate_environment_sha256']==identities['environment_sha256'],'regression provenance changed')
    check(len(regression['controls'])==6 and all(row.get('exact_logits_and_nll') is True and row.get('num_rows',0)>0 for row in regression['controls']),'six exact original regressions required')
    check(bool(regression['source_files']),'regression sources absent')
    for name,digest in regression['source_files'].items():check(sha256_file(name)==digest,'regression evidence changed: '+name)
    evidence.append(regression_path);evidence.extend(Path(name) for name in regression['source_files'])
    if matrix.get('execution_phase')=='conditional':
        activation=matrix['conditional_activation'];evidence.extend(Path(activation[key]) for key in ('decision_path','decision_manifest_path'))
        evidence.extend(Path(name) for name in load_json(activation['decision_manifest_path'])['source_files'])
    specifications=matrix.get('specification_files',{});check(len(specifications)>=2,'frozen SPEC and MX arithmetic decision required')
    for name,digest in specifications.items():
        spec=Path(name).resolve();check(spec.is_file() and sha256_file(spec)==digest,'specification absent or changed');evidence.append(spec)
    diagnosis_binding=matrix['diagnosis_manifest'];diagnosis_path=Path(diagnosis_binding['path']).resolve()
    check(sha256_file(diagnosis_path)==diagnosis_binding['sha256'],'TC diagnosis manifest changed');diagnosis=load_json(diagnosis_path)
    check(diagnosis.get('status')=='verified' and (diagnosis['traces'],diagnosis['source_head_operand_rows'],diagnosis['sign_rows'],diagnosis['logit_rows'])==(54,3456,10368,1728),'actual TC distribution diagnosis incomplete')
    evidence.append(diagnosis_path)
    for item in diagnosis['inputs']+diagnosis['outputs']:
        path=Path(item['path']).resolve();check(sha256_file(path)==item['sha256'],'TC diagnosis source/output changed');evidence.append(path)
    for vid,cfg in configs.items():
        for suffix,mode,targets in [('windows','windows',4095),('decode-off','cached_decode',16),('decode-on','cached_decode',16)]:
            run=root/f'{vid}-{suffix}';m=check_run(run,cfg,'smoke',mode,identities,targets)
            check(m['selection_passes']==1,'selection pass count')
            costs=load_json(run/'costs.json');check(0<costs['selected_full_qk_macs']<=costs['actual_selected_full_qk_macs'],'selected original full QK work')
            evidence.extend(run.rglob('*'))
        off,on=root/f'{vid}-decode-off',root/f'{vid}-decode-on'
        check(compare_runs(off,on)['passed'],'trace off/on logits or NLL changed')
        replays[vid]=replay_run(on);check(replays[vid]['passed'],'replay incomplete');traces+=replays[vid]['num_records']
        pairing=root/'pairing'/vid/'pairing_manifest.json';pair=load_json(pairing)
        check(pair.get('status')=='complete' and pair.get('passed') is True,'pairing incomplete')
        check(pair['candidate_variant_id']==vid,'pair candidate mismatch')
        check(pair['counts']=={'cross_run_trajectory':576,'same_input_estimator':192},'pair coverage incomplete')
        for name,digest in {**pair['source_files'],**pair['artifacts']}.items():check(sha256_file(name)==digest,'pair source changed')
        pair_counts[vid]=pair['counts'];evidence.extend(pairing.parent.rglob('*'));evidence.extend(Path(n) for n in pair['source_files'])
    check(traces==6*len(configs),'trace coverage')
    evidence.extend((matrix_path.parent/'configs').glob('*.json'));evidence.append(matrix_path.parent/matrix['trace_targets'])
    files={str(p.resolve()):sha256_file(p) for p in sorted(set(evidence)) if p.is_file()}
    return dict(schema_version='attention-format-gate-v1',status='complete',passed=True,smoke_root=str(root),**identities,head=code['head'],source_files=code['files'],matrix_path=str(matrix_path),matrix_sha256=sha256_file(matrix_path),semantic_config_sha256=hashes,query_chunk=64,context_length=2048,stride=1024,trace_records=traces,pairing_counts=pair_counts,replay_reports=replays,evidence_files=files)


def validate_gate_report(report,model_path,assets,config,query_chunk=64):
    report=load_json(report) if not isinstance(report,dict) else report
    check(report.get('schema_version')=='attention-format-gate-v1' and report.get('passed') is True and report.get('status')=='complete','format release gate incomplete')
    check(query_chunk==report['query_chunk']==64,'query chunk changed');configure_runtime();code=code_provenance();env=environment()
    check(code['code_sha256']==report['code_sha256'] and code['head']==report['head'] and code['files']==report['source_files'],'source changed')
    check(env['environment_sha256']==report['environment_sha256'],'environment changed');manifest=validate_assets(model_path,Path(assets).parent/'tokens.pt',assets)
    check(canonical_hash(manifest)==report['asset_manifest_sha256'] and manifest['tokens_sha256']==report['tokens_sha256'],'assets changed')
    validate_formal_config(config);vid=config['attention_slice']['variant_id'];check(canonical_hash(config)==report['semantic_config_sha256'].get(vid),'config changed')
    _,configs=read_matrix(report['matrix_path']);check(config==configs[vid] and sha256_file(report['matrix_path'])==report['matrix_sha256'],'matrix changed')
    for name,digest in report['evidence_files'].items():check(Path(name).is_file() and sha256_file(name)==digest,'evidence changed: '+name)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('smoke-root','matrix','model','assets','output'):p.add_argument('--'+flag,required=True)
    a=p.parse_args()
    try:report=build_gate(a.smoke_root,a.matrix,a.model,a.assets)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
