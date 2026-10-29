"""RS paired estimands: separate saturation, prefix selection and final format."""
import argparse
import csv
import json
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_dense_format.pair_sets import identity,bind_run,historical_records
from benchmark.attention_slice.pair_sets import set_metrics
from .formal import read_matrix
from .oracle import check,compare,selection
from .replay import load_trace_records
from .historical import validate_source_final

NOTE='Native full RS final scores differ from original-source MSB final scores; their cross-run PPL gap is not truncation alone. Source-scale reuse is a study adaptation of paper LSQ/QAT and clips120..127 to119.'

def errors(a,b):
    x=a.double();y=b.double();return dict(logit_max_abs=float((x-y).abs().max()),logit_rmse=float((x-y).square().mean().sqrt()),prob_max_abs=float((a.float().softmax(-1)-b.float().softmax(-1)).abs().max()))

def pair_sets(matrix,runs_root,output_dir,controls_manifest=None):
    _,configs=read_matrix(matrix);root=Path(runs_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False);groups={};sources={};rows=[];expected_cross=0
    for vid,cfg in configs.items():
        run=root/(vid+'-decode-on');records=load_trace_records(run);check(load_json(run/'resolved_config.json')['algorithm']==cfg,'RS pair config differs');bind_run(run,records,sources);groups[vid]={}
        for name in ('selection_sets.jsonl','rs_statistics.json','rs_manifest.json'):sources[str(run/name)]=sha256_file(run/name)
        for entry,meta,a,d in records:
            key=canonical_hash(identity(meta,a));check(key not in groups[vid],'duplicate RS pair identity');groups[vid][key]=(entry,meta,a,d)
    first=next(iter(groups))
    for vid in groups:check(set(groups[vid])==set(groups[first]),'RS cross-run protocol identity mismatch')
    # Six MSB jobs pair against actual same-p full-native captures, never derived replacements.
    links=[]
    for vid,cfg in configs.items():
        s=cfg['attention_rs']
        if s['mode']=='msb_topp':links.append((vid,'rs_full_q8k8_topp_p'+vid[-3:],'actual_native_full_same_p'))
        elif s['mode']=='full_topp':links.append((vid,'rs_dense_q8k8','actual_native_dense'))
    external={};unavailable_dense={};controls=load_json(controls_manifest) if controls_manifest else {}
    # Explicit manifest maps each candidate to compatible actual TC/SM same-p controls.
    for vid,mapping in controls.get('top_p',{}).items():
        check(vid in configs and configs[vid]['attention_rs']['mode']=='msb_topp','control candidate must be MSB')
        for role,path in mapping.items():
            check(role in ('tc','sm'),'unsupported RS historical role');path=Path(path).resolve();check(load_json(path/'status.json')['status']=='complete','historical Top-p incomplete')
            if role=='tc':
                from benchmark.attention_slice.replay import load_trace_records as loader
            else:
                from benchmark.attention_format.replay import load_trace_records as loader
            records=loader(path);group={};expected=configs[vid]['attention_rs'];p=expected['selection']['p']
            for entry,meta,a,d in records:
                hs=meta['attention_slice'];check(hs['selection']['p']==p and hs['q_prefix_bits']==expected['q_bits'] and hs['k_prefix_bits']==expected['k_bits'],'historical same-p/shape differs')
                check(hs['estimator']==('tc_msb' if role=='tc' else 'format_msb'),'historical control estimator differs')
                if role=='sm':check(hs['predictor_format']=='sm8','historical SM role differs')
                validate_source_final(meta,role)
                predicted=selection(a['estimator_logits'],a['allowed_mask'],p)[0];compare(a['selected_mask'],predicted,'historical saved set',True)
                group[canonical_hash(identity(meta,a))]=(entry,meta,a,d)
            key=vid+':'+role;check(set(group)==set(groups[vid]),'historical protocol/hash identity differs');external[key]=group;links.append((vid,key,'historical_'+role+'_same_p'));bind_run(path,records,sources);sources[str(path/'selection_sets.jsonl')]=sha256_file(path/'selection_sets.jsonl')
    # Authentic old dense source is optional; diagnostic sets are derived explicitly from its actual saved probabilities.
    for role,path in controls.get('dense',{}).items():
        check(role in ('fp','source_int'),'unsupported dense role');path=Path(path).resolve();records=historical_records(path);group={canonical_hash(identity(meta,a)):(entry,meta,a,d) for entry,meta,a,d in records};bind_run(path,records,sources)
        if set(group)!=set(groups[first]):
            candidate=next(iter(groups[first].values()))[1];historical=records[0][1]
            unavailable_dense[role]=dict(reason='strict full protocol/environment identity mismatch; no cross-run row invented',candidate_environment_sha256=candidate['environment_sha256'],historical_environment_sha256=historical['environment_sha256'],historical_driver=load_json(path/'environment.json').get('driver'),source_run=str(path),same_input_source_dense_reference_available=True)
            continue
        external[role]=group;links.append(('rs_dense_q8k8',role,'historical_dense_'+role))
    if controls_manifest:sources[str(Path(controls_manifest).resolve())]=sha256_file(controls_manifest)
    for candidate,reference,role in links:
        reference_group=groups.get(reference,external.get(reference));expected_cross+=192
        for key,(entry,meta,a,d) in groups[candidate].items():
            re,rm,ra,rd=reference_group[key];s=configs[candidate]['attention_rs'];p=s['selection']['p']
            # Dense comparisons use all-legal execution. Full-vs-dense uses offline same-p diagnostic only.
            if p is None:ref=ra['allowed_mask'].expand_as(ra['logits_e'] if 'logits_e' in ra else ra['estimator_logits']);rp=ra.get('probs',ra.get('estimator_probs'))
            elif rm.get('rs_mode')=='dense_full':ref=selection(ra['estimator_logits'],ra['allowed_mask'],p)[0];rp=ra['estimator_probs']
            else:ref=ra['selected_mask'];rp=ra['estimator_probs']
            for head in range(32):
                pred=a['selected_mask'][0,head,0];reference_mask=ref[0,head,0];mass_p=1. if p is None else p
                rows.append(dict(comparison_kind='cross_run_trajectory',candidate_variant_id=candidate,reference_variant_id=reference,reference_role=role,p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,candidate_actual_sha256=entry['actual_sha256'],reference_actual_sha256=re['actual_sha256'],candidate_code_sha256=meta['code_sha256'],reference_code_sha256=rm['code_sha256'],candidate_final_operand_origin=s['final_operand_origin'],reference_final_operand_origin=rm.get('final_operand_origin','original_source_dense'),reference_set_role='offline_native_dense_important_keys' if p is not None and rm.get('rs_mode')=='dense_full' else 'actual_execution_set',reference_set_unused_in_forward=bool(p is not None and rm.get('rs_mode')=='dense_full'),comparison_note=NOTE,**set_metrics(pred,reference_mask,rp[0,head,0],mass_p)))
    # 3 reference directions per actual execution row, with final-format diagnostics on S.
    for vid,group in groups.items():
        s=configs[vid]['attention_rs'];p=s['selection']['p'];threshold=1. if p is None else p
        for key,(entry,meta,a,d) in group.items():
            native,source,predictor=d['native_full_logits'],d['original_source_logits'],d['actual_predictor_logits'];domain=a['allowed_mask']
            definitions=[('native_vs_source_saturation',native,source),('predictor_vs_native_prefix',predictor,native),('predictor_vs_source_combined',predictor,source),('predictor_vs_original_fp_combined',predictor,d['original_fp_logits'])]
            for label,candidate_logits,reference_logits in definitions:
                pred=selection(candidate_logits,domain,p)[0];ref=selection(reference_logits,domain,p)[0];probability=reference_logits.softmax(-1)
                for head in range(32):
                    base=dict(comparison_kind='same_input_estimator',candidate_variant_id=vid,reference_role=label,p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,actual_sha256=entry['actual_sha256'],derived_sha256=entry['derived_sha256'],unused_in_forward=True,candidate_final_operand_origin=s['final_operand_origin'],comparison_note=NOTE)
                    clipped_q=a['q_rs_saturation_mask'][0,head,0];clipped_k=a['k_rs_saturation_mask'][0,head]
                    base.update(q_saturation_count=int(clipped_q.sum()),q_saturation_denominator=clipped_q.numel(),k_saturation_count=int(clipped_k.sum()),k_saturation_denominator=clipped_k.numel(),**set_metrics(pred[0,head,0],ref[0,head,0],probability[0,head,0],threshold),**errors(candidate_logits[0,head,0],reference_logits[0,head,0]))
                    # Both final formats on exactly the actual set are offline, not extra runs.
                    mask=a['selected_mask'][0,head,0];nf=d['native_final_on_actual_set'][0,head,0];sf=d['source_final_on_actual_set'][0,head,0]
                    np=d['native_probs_on_actual_set'][0,head,0];sp=d['source_probs_on_actual_set'][0,head,0]
                    base.update(final_format_score_max_abs_on_actual_set=float((nf[mask]-sf[mask]).abs().max()),final_format_prob_max_abs_on_actual_set=float((np-sp).abs().max()),final_format_pv_max_abs_on_actual_set=float(((np-sp)@a['v_visible'][0,head].float()).abs().max()))
                    rows.append(base)
    counts={kind:sum(r['comparison_kind']==kind for r in rows) for kind in ('cross_run_trajectory','same_input_estimator')};expected=dict(cross_run_trajectory=expected_cross,same_input_estimator=7680);check(counts==expected,'RS pair coverage')
    jsonpath=out/'paired_sets.jsonl';jsonpath.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows));csvpath=out/'set_comparison.csv'
    with csvpath.open('w',newline='') as stream:
        fields=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report=dict(schema_version='attention-rs-pairing-v1',status='complete',passed=True,counts=counts,expected_counts=expected,estimands_separate=True,actual_native_full_pairs=6,historical_controls=list(external),unavailable_dense_controls=unavailable_dense,historical_controls_required_for_release=True,source_files=sources,artifacts={str(file):sha256_file(file) for file in (jsonpath,csvpath)},quality_is_measured_not_threshold_gated=True,comparison_note=NOTE)
    write_json(out/'pairing_manifest.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('matrix','runs-root','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--controls-manifest');a=p.parse_args()
    try:pair_sets(a.matrix,a.runs_root,a.output_dir,a.controls_manifest)
    except Exception as exc:write_json(Path(a.output_dir)/'failure.json',dict(status='failed',passed=False,reason=str(exc)));raise
if __name__=='__main__':main()
