"""Same-input INT-TopP/format-dense controls and separate real-run trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_dense_format.pair_sets import identity,quality,bind_run
from benchmark.attention_dense_format.replay import load_trace_records as load_dense
from benchmark.attention_slice.replay import load_trace_records as load_source_int
from benchmark.attention_dense_format.oracle import check,top_p
from .formal import read_matrix,THRESHOLDS
from .replay import load_trace_records

HISTORICAL={'095':'/output/project/experiments/stp-slice/20261028-v2-smoke/int_topp_p095-decode-on',**{p:f'/output/project/experiments/stp-slice-thresholds/p{p}-smoke/int_topp_p{p}-decode-on' for p in ('090','085')}}


def source_group(run,loader,sources):
    run=Path(run).resolve();records=loader(run);bind_run(run,records,sources)
    listing=run/'selection_sets.jsonl'
    if listing.is_file():sources[str(listing)]=sha256_file(listing)
    result={}
    for entry,meta,a,d in records:
        key=canonical_hash(identity(meta,a));check(key not in result,'duplicate pair identity');result[key]=(entry,meta,a,d)
    return result


def pair_sets(matrix,runs_root,output_dir,dense_smoke_root=None,controls_manifest=None):
    _,configs=read_matrix(matrix);root=Path(runs_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    sources={};groups={};rows=[];artifacts=[];cross_counts={};unavailable={};controls=load_json(controls_manifest) if controls_manifest else HISTORICAL
    for vid,cfg in configs.items():
        run=root/(vid+'-decode-on');check(load_json(run/'resolved_config.json')['algorithm']==cfg,'pair config differs');groups[vid]=source_group(run,load_trace_records,sources)
    first=next(iter(groups))
    for group in groups.values():check(set(group)==set(groups[first]),'full-format trajectory identity mismatch')
    # Every candidate has two explicitly same-input controls, no invented real run.
    for vid,group in groups.items():
        s=configs[vid]['attention_full_format'];p=s['selection']['p'];suffix=vid[-3:]
        for key,(entry,meta,a,d) in group.items():
            source=d['reference_source_int_logits_same_input'];mask,source_probs,_=top_p(source,a['allowed_mask'],a['key_positions'],p)
            selected=source.masked_fill(~mask,-torch.inf);source_pv=selected.softmax(-1)@a['v_visible'].float()
            saved=dict(source_int_top_p_mask=mask,source_int_selected_probs=selected.softmax(-1),source_int_pv_head=source_pv)
            path=out/(vid+'-'+entry['record_id']+'-controls.pt');torch.save(saved,path);artifacts.append(path);digest=sha256_file(path)
            for role,refmask,refprobs,reflogits,refpv in [('source_int_top_p',mask,source_probs,source,source_pv),('own_format_dense',a['allowed_mask'].expand_as(a['selected_mask']),a['full_format_probs'],a['full_format_logits'],d['full_format_dense_pv_same_input'])]:
                for head in range(32):
                    pred=a['selected_mask'][0,head,0];ref=refmask[0,head,0]
                    rows.append(dict(comparison_kind='same_input',reference_role=role,candidate_variant_id=vid,reference_variant_id=role,p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,candidate_record_id=entry['record_id'],candidate_actual_sha256=entry['actual_sha256'],candidate_derived_sha256=entry['derived_sha256'],reference_artifact=str(path),reference_artifact_sha256=digest,actual_selected_key_positions=a['key_positions'][pred].tolist(),reference_key_positions=a['key_positions'][ref].tolist(),**quality(pred,ref,refprobs[0,head,0],a['full_format_logits'][0,head,0],reflogits[0,head,0],a['pv_head'][0,head,0],refpv[0,head,0],p,a['key_positions'].numel())))
    references={};dense_root=Path(dense_smoke_root).resolve() if dense_smoke_root else Path('/output/project/experiments/stp-dense-formats/execution-v1/smoke')
    for vid,cfg in configs.items():
        suffix=vid[-3:];fmt=cfg['attention_full_format']['format'];p=cfg['attention_full_format']['selection']['p']
        candidate_refs={}
        historical=Path(controls[suffix]).resolve() if controls.get(suffix) else None
        if historical and (historical/'trace/index.json').is_file():
            controlcfg=load_json(historical/'resolved_config.json')['algorithm'];sc=controlcfg.get('attention_slice',{})
            check(sc.get('estimator')=='int_full' and sc['selection']['p']==p,'historical control must be full source-INT TopP at same p')
            key='historical_int_p'+suffix
            if key not in references:references[key]=source_group(historical,load_source_int,sources)
            candidate_refs[key]=references[key]
        else:unavailable[vid+':historical_int']='No authentic matching cross-run source; same-input source INT control retained'
        dense_run=dense_root/('dense_'+fmt+'_q8k8-decode-on')
        if (dense_run/'trace/index.json').is_file():
            key='dense_'+fmt
            if key not in references:references[key]=source_group(dense_run,load_dense,sources)
            candidate_refs[key]=references[key]
        else:unavailable[vid+':format_dense']='No authentic matching dense capture supplied; exact same-input format-dense distribution retained'
        # One directed SM->MX comparison at each threshold avoids duplicate pairs.
        if fmt=='sm8':candidate_refs['mxint8_q8k8_topp_p'+suffix]=groups['mxint8_q8k8_topp_p'+suffix]
        for role,reference in candidate_refs.items():
            check(set(reference)==set(groups[vid]),'real cross-run identity/domain mismatch '+role);count=0
            for key,(entry,meta,a,d) in groups[vid].items():
                re,rm,ra,rd=reference[key]
                if role.startswith('dense_'):
                    refmask,refprobs,_=top_p(ra['logits_e'],ra['allowed_mask'],ra['key_positions'],p);reflogits=ra['logits_e'];refpv=ra['pv_head']
                elif role.startswith('historical_'):
                    refmask=ra['selected_mask'];refprobs=ra['estimator_probs'];reflogits=ra['estimator_logits'];refpv=ra['pv_head']
                else:refmask=ra['selected_mask'];refprobs=ra['full_format_probs'];reflogits=ra['full_format_logits'];refpv=ra['pv_head']
                for head in range(32):
                    pred=a['selected_mask'][0,head,0];ref=refmask[0,head,0]
                    rows.append(dict(comparison_kind='cross_run_trajectory',reference_role=role,candidate_variant_id=vid,reference_variant_id=rm.get('variant_id',role),p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,candidate_record_id=entry['record_id'],reference_record_id=re['record_id'],candidate_actual_sha256=entry['actual_sha256'],reference_actual_sha256=re['actual_sha256'],candidate_code_sha256=meta['code_sha256'],reference_code_sha256=rm['code_sha256'],actual_selected_key_positions=a['key_positions'][pred].tolist(),reference_key_positions=ra['key_positions'][ref].tolist(),**quality(pred,ref,refprobs[0,head,0],a['full_format_logits'][0,head,0],reflogits[0,head,0],a['pv_head'][0,head,0],refpv[0,head,0],p,a['key_positions'].numel())));count+=1
            cross_counts[vid+':'+role]=count;check(count==192,'cross-run coverage incomplete')
    expected=dict(same_input=2304,cross_run_trajectory=sum(cross_counts.values()));counts={kind:sum(row['comparison_kind']==kind for row in rows) for kind in expected};check(counts==expected and counts['cross_run_trajectory']>=576,'pair coverage differs')
    jsonpath=out/'paired_sets.jsonl';jsonpath.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows));artifacts.append(jsonpath);csvpath=out/'set_comparison.csv'
    with csvpath.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(row.keys() for row in rows))));writer.writeheader();writer.writerows(rows)
    artifacts.append(csvpath)
    if controls_manifest:sources[str(Path(controls_manifest).resolve())]=sha256_file(controls_manifest)
    report=dict(schema_version='full-format-pairing-v1',status='complete',passed=True,counts=counts,expected_counts=expected,cross_run_pairs=cross_counts,unavailable_cross_run_controls=unavailable,estimands_separate=True,source_files=sources,artifacts={str(path):sha256_file(path) for path in artifacts},quality_is_measured_not_threshold_gated=True)
    write_json(out/'pairing_manifest.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('matrix','runs-root','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--dense-smoke-root');p.add_argument('--controls-manifest');a=p.parse_args()
    try:pair_sets(a.matrix,a.runs_root,a.output_dir,a.dense_smoke_root,a.controls_manifest)
    except Exception as exc:write_json(Path(a.output_dir)/'failure.json',dict(status='failed',passed=False,reason=str(exc)));raise
if __name__=='__main__':main()
