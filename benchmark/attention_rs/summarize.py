"""Ten RS measurement rows; unexecuted measurements remain null."""
import argparse
import csv
import json
import math
from pathlib import Path
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from .formal import read_matrix
from .oracle import check
from .pair_sets import NOTE

def summarize(matrix,runs_root,output_dir,profile='full',baselines_manifest=None):
    _,configs=read_matrix(matrix);root=Path(runs_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False);rows=[];sources={};anchors={};baseline=load_json(baselines_manifest) if baselines_manifest else {}
    if baseline:
        required={'source_dense','fp_dense'}|{vid+':'+role for vid,cfg in configs.items() if cfg['attention_rs']['mode']=='msb_topp' for role in ('tc','sm')}
        check(set(baseline)==required,'all frozen dense/TC/SM PPL anchors required')
    for role,path in baseline.items():
        path=Path(path).resolve();check(load_json(path/'status.json')['status']=='complete','baseline incomplete');m=load_json(path/'metrics.json');cfg=load_json(path/'resolved_config.json')['algorithm']
        check(m['profile']=='full' and m['mode']=='windows' and m['num_targets']==289075,'baseline protocol changed');check(m['ppl']==math.exp(m['sum_nll']/m['num_targets']),'baseline formula')
        if role=='source_dense':check(cfg['attention_quantization']['method']=='intflash_qk8' and not any(key in cfg for key in ('attention_slice','attention_rs','attention_dense_format','attention_full_format')),'original-source dense role')
        if role=='fp_dense':check(cfg['attention_quantization']['method']=='fp_common' and not any(key in cfg for key in ('attention_slice','attention_rs','attention_dense_format','attention_full_format')),'FP dense role')
        if ':' in role:
            candidate,kind=role.split(':');expected=configs[candidate]['attention_rs'];actual=cfg['attention_slice']
            check(actual['q_prefix_bits']==expected['q_bits'] and actual['k_prefix_bits']==expected['k_bits'] and actual['selection']['p']==expected['selection']['p'],'historical PPL same-p/shape')
            check(actual['estimator']==('tc_msb' if kind=='tc' else 'format_msb') and actual['final_scores']=='full_int8_qk_dq','historical PPL final origin')
            if kind=='sm':check(actual['predictor_format']=='sm8','historical PPL SM role')
        anchors[role]=(m,path)
        for name in ('status.json','metrics.json','resolved_config.json','asset_manifest.json','code_manifest.json','coverage.json'):sources[str(path/name)]=sha256_file(path/name)
    suffix='windows-full' if profile=='full' else 'windows';expected=289075 if profile=='full' else 4095
    for vid,cfg in configs.items():
        s=cfg['attention_rs'];run=root/(vid+'-'+suffix);status=load_json(run/'status.json') if (run/'status.json').is_file() else dict(status='not_executed')
        row=dict(variant_id=vid,mode=s['mode'],p=s['selection']['p'],q_bits=s['q_bits'],k_bits=s['k_bits'],selection_passes=s['selection_passes'],final_scores=s['final_scores'],final_operand_origin=s['final_operand_origin'],final_execution=s['final_execution'],profile=profile,planned=True,implemented=True,executed=status['status']!='not_executed',verified=False,status=status['status'],code_sha256=None,sum_nll=None,num_targets=None,ppl=None,delta_ppl_vs_original_source_dense=None,delta_ppl_vs_same_p_native_full=None,delta_ppl_vs_compatible_controls=None,historical_baseline_environment_differences=None,historical_dense_anchor_verification='measured_historical; matching_full_gate_unavailable; verified=no' if baseline else None,selection_mean=None,selected_fraction=None,retained_original_reference_mass=None,same_input_pairing_scope=None,saturation_q_count=None,saturation_k_count=None,saturation_q_denominator=None,saturation_k_denominator=None,comparison_note=NOTE,run_dir=str(run))
        if status['status']=='complete':
            m=load_json(run/'metrics.json');check(m['num_targets']==expected and m['attention_rs']==s and m['semantic_config_sha256']==canonical_hash(cfg),'RS summary protocol')
            check(math.isfinite(m['sum_nll']) and m['ppl']==math.exp(m['sum_nll']/expected),'RS PPL formula');row.update(sum_nll=m['sum_nll'],num_targets=m['num_targets'],ppl=m['ppl'],code_sha256=m['code_sha256'])
            stats=load_json(run/'selection_stats.json');sat=load_json(run/'rs_statistics.json')['by_layer_operand'];row.update(selection_mean=stats['mean_selected_keys'],selected_fraction=stats['selected_fraction'])
            for operand in ('q','k'):
                row['saturation_'+operand+'_count']=sum(sum(group['saturation_count']) for key,group in sat.items() if key.endswith(':'+operand));row['saturation_'+operand+'_denominator']=sum(sum(group['denominator']) for key,group in sat.items() if key.endswith(':'+operand))
            for name in ('status.json','metrics.json','resolved_config.json','method_manifest.json','rs_manifest.json','coverage.json','selection_stats.json','costs.json','rs_statistics.json','code_manifest.json'):sources[str(run/name)]=sha256_file(run/name)
            gate=root/'gate_reference.json';smoke_gate=root/'gate.json'
            if gate.is_file():
                reference=load_json(gate);check(sha256_file(reference['path'])==reference['sha256'],'RS gate reference changed');accepted=load_json(reference['path']);row['verified']=accepted.get('passed') is True and accepted['code_sha256']==m['code_sha256'] and accepted['semantic_config_sha256'][vid]==m['semantic_config_sha256']
            elif smoke_gate.is_file():accepted=load_json(smoke_gate);row['verified']=accepted.get('passed') is True and accepted.get('code_sha256')==m['code_sha256']
            for role,(anchor,path) in anchors.items():
                for key in ('asset_manifest_sha256','tokens_sha256','context_length','stride','query_chunk','model_dtype'):check(m[key]==anchor[key],'baseline compatibility '+key)
                if profile=='full' and role=='source_dense':row['delta_ppl_vs_original_source_dense']=m['ppl']-anchor['ppl']
            env_differences={role:dict(candidate_environment_sha256=m['environment_sha256'],historical_environment_sha256=anchor['environment_sha256'],historical_driver=load_json(path/'environment.json').get('driver')) for role,(anchor,path) in anchors.items() if role in ('source_dense','fp_dense') or role.startswith(vid+':') if m['environment_sha256']!=anchor['environment_sha256']}
            row['historical_baseline_environment_differences']=json.dumps(env_differences,sort_keys=True) if env_differences else None
            controls={role:m['ppl']-anchor['ppl'] for role,(anchor,path) in anchors.items() if profile=='full' and role.startswith(vid+':')}
            row['delta_ppl_vs_compatible_controls']=json.dumps(controls,sort_keys=True) if controls else None
        rows.append(row)
    pairing=root/'pairing/paired_sets.jsonl'
    if pairing.is_file():
        paired=[json.loads(line) for line in pairing.read_text().splitlines() if line]
        sources[str(pairing)]=sha256_file(pairing)
        for row in rows:
            selected=[value for value in paired if value['comparison_kind']=='same_input_estimator' and value['candidate_variant_id']==row['variant_id'] and value['reference_role']=='predictor_vs_source_combined']
            if selected:
                row['retained_original_reference_mass']=sum(value['retained_mass'] for value in selected)/len(selected)
                row['same_input_pairing_scope']='companion_capture_rows_only'
    lookup={r['variant_id']:r for r in rows}
    for row in rows:
        if row['p'] is not None:
            reference=lookup['rs_full_q8k8_topp_p'+row['variant_id'][-3:]]
            if row['ppl'] is not None and reference['ppl'] is not None:row['delta_ppl_vs_same_p_native_full']=row['ppl']-reference['ppl']
    csvpath=out/'RS_RESULTS.csv'
    with csvpath.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md=['# Native RS measurements','',NOTE,'','PPL values are actual separate model executions. CPU codec identity is not a dense PPL measurement. No native low-bit throughput is claimed.','', '| Variant | Final operands | Status | Verified | Targets | PPL |','|---|---|---|---|---:|---:|']
    for row in rows:md.append('| '+' | '.join('null' if row[key] is None else str(row[key]) for key in ('variant_id','final_operand_origin','status','verified','num_targets','ppl'))+' |')
    (out/'RS_RESULTS.md').write_text('\n'.join(md)+'\n');report=dict(schema_version='attention-rs-summary-v1',status='complete',profile=profile,rows=rows,historical_anchors={role:dict(path=str(path),ppl=m['ppl'],num_targets=m['num_targets'],measured=True,verified=False,matching_full_gate_available=False if role in ('fp_dense','source_dense') else None,verification_note='read-only measured anchor; RS gate does not confer historical verification') for role,(m,path) in anchors.items()},source_files=sources,artifacts={str(p):sha256_file(p) for p in (csvpath,out/'RS_RESULTS.md')},comparison_note=NOTE)
    # Runner-compatible alias points at the same ten actual measurement rows.
    (out/'BASELINE.csv').write_bytes(csvpath.read_bytes());report['artifacts'][str(out/'BASELINE.csv')]=sha256_file(out/'BASELINE.csv');write_json(out/'summary_manifest.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('matrix','runs-root','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--profile',choices=['smoke','full'],default='full');p.add_argument('--baselines-manifest');a=p.parse_args();summarize(a.matrix,a.runs_root,a.output_dir,a.profile,a.baselines_manifest)
if __name__=='__main__':main()
