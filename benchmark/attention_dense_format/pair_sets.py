"""Pair dense trajectories separately from frozen-input codec counterfactuals."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_slice.pair_sets import set_metrics
from .formal import read_matrix,THRESHOLDS
from .oracle import check,scalar_encode,formats,top_p
from .replay import load_trace_records

DENSE_CONTROLS={role:str(Path('/output/project/experiments/stp-baselines/20261027-v5-smoke')/(method+'-decode-on')) for role,method in [('fp','fp_common'),('source_int','intflash_qk8')]}


def identity(meta,a):
    required=('tokens_sha256','asset_manifest_sha256','model_file_hashes','sample_id','window_id','layer_id','stage','query_position','predicted_target','model_dtype','environment_sha256','protocol_id','profile','mode','subset_sha256','scoring_ranges_sha256','context_length','stride','query_chunk','prefill_tokens','decode_targets','Hq','Hkv','D','head_to_kv','input_ids','position_ids','cache_position','global_query_positions','cache_length','expected_cache_length')
    check(all(k in meta for k in required),'missing pairing identity')
    return {**{k:meta[k] for k in required},'key_positions':a['key_positions'].tolist()}


def quality(pred,ref,probability,logits,reference_logits,pv,reference_pv,p,allowed_count):
    result=set_metrics(pred,ref,probability,p)
    centered=logits.double()-logits.double().mean();reference_centered=reference_logits.double()-reference_logits.double().mean()
    result.update(argmax_survives=bool(pred[int(probability.argmax())]),candidate_fraction=int(pred.sum())/allowed_count,reference_fraction=int(ref.sum())/allowed_count,
                  allowed_count=allowed_count,centered_logit_max_abs=float((centered-reference_centered).abs().max()),centered_logit_rmse=float((centered-reference_centered).square().mean().sqrt()),
                  pv_relative_error=float(torch.linalg.vector_norm(pv.double()-reference_pv.double())/torch.linalg.vector_norm(reference_pv.double()).clamp_min(1e-30)))
    return result


def bind_run(root,records,sources):
    for name in ('status.json','metrics.json','resolved_config.json','asset_manifest.json','code_manifest.json','code.patch','environment.json','trace/index.json'):
        path=root/name;check(path.is_file(),'pairing source missing '+str(path));sources[str(path)]=sha256_file(path)
    for entry,_,_,_ in records:
        for name in ('actual.pt','derived.pt','meta.json'):
            path=root/'trace'/entry['path']/name;sources[str(path)]=sha256_file(path)
    listing=root/'diagnostic_sets.jsonl'
    if listing.is_file():sources[str(listing)]=sha256_file(listing)


def historical_records(root):
    """Load only genuine dense captures; never relabel a TopP execution."""
    root=Path(root).resolve();check(load_json(root/'status.json')['status']=='complete','historical control incomplete')
    cfg=load_json(root/'resolved_config.json')['algorithm'];check('attention_slice' not in cfg and 'attention_dense_format' not in cfg,'historical control is not dense')
    index=load_json(root/'trace/index.json');check(index['schema_version']=='attention-trace-v1' and len(index['records'])==6,'historical authentic dense schema/coverage')
    records=[]
    for entry in index['records']:
        path=root/'trace'/entry['path'];meta=load_json(path/'meta.json')
        for name,key in [('actual.pt','actual_sha256'),('derived.pt','derived_sha256'),('meta.json','meta_sha256')]:check(sha256_file(path/name)==entry[key],'historical trace index hash')
        for name in ('actual.pt','derived.pt'):check(meta['files'][name]==sha256_file(path/name),'historical trace meta hash')
        records.append((entry,meta,torch.load(path/'actual.pt',map_location='cpu',weights_only=True),torch.load(path/'derived.pt',map_location='cpu',weights_only=True)))
    return records


def pair_sets(matrix,runs_root,output_dir,controls_manifest=None):
    _,configs=read_matrix(matrix);root=Path(runs_root).resolve();out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    groups={};sources={};rows=[];artifacts=[];controls=load_json(controls_manifest) if controls_manifest else DENSE_CONTROLS
    for vid,cfg in configs.items():
        run=root/(vid+'-decode-on');records=load_trace_records(run);bind_run(run,records,sources);group={}
        check(load_json(run/'resolved_config.json')['algorithm']==cfg,'candidate pairing config mismatch')
        for entry,meta,a,d in records:
            ident=identity(meta,a);key=canonical_hash(ident);check(key not in group,'duplicate record pairing identity');group[key]=(entry,meta,a,d)
        groups[vid]=group
    first=next(iter(groups))
    for vid in groups:check(set(groups[vid])==set(groups[first]),'dense cross-run identity mismatch')
    historical={};unavailable={}
    for role in ('fp','source_int'):
        if not controls.get(role):unavailable[role]='No authentic dense cross-run source supplied; same-input control derived';continue
        run=Path(controls[role]).resolve()
        if not (run/'trace/index.json').is_file():unavailable[role]='Authentic dense source unavailable; same-input control derived';continue
        records=historical_records(run);bind_run(run,records,sources)
        group={canonical_hash(identity(meta,a)):(entry,meta,a,d) for entry,meta,a,d in records}
        # A mismatched real source is evidence of an unavailable pairing, never invented.
        if set(group)!=set(groups[first]):unavailable[role]='Authentic source exists but strict token/model/context/head identity differs';continue
        historical[role]=group
    if controls_manifest:sources[str(Path(controls_manifest).resolve())]=sha256_file(controls_manifest)
    # Unordered pairs cover every dense format pairing exactly once at each p.
    trajectory_pairs=list(itertools.combinations(configs,2))+[(vid,role) for vid in configs for role in historical]
    for candidate,reference in trajectory_pairs:
        candidate_group=groups[candidate];reference_group=groups.get(reference,historical.get(reference))
        for key,(entry,meta,a,d) in candidate_group.items():
            re,rm,ra,rd=reference_group[key]
            for suffix,p in THRESHOLDS.items():
                candidate_mask=d['diagnostic_p'+suffix+'_mask']
                if reference in groups:reference_mask=rd['diagnostic_p'+suffix+'_mask']
                else:reference_mask,_,_=top_p(ra['logits_e'],ra['allowed_mask'],ra['key_positions'],p)
                for head in range(32):
                    pred,ref=candidate_mask[0,head,0],reference_mask[0,head,0]
                    rows.append(dict(comparison_kind='cross_run_trajectory',candidate_variant_id=candidate,reference_variant_id=reference,p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,candidate_record_id=entry['record_id'],reference_record_id=re['record_id'],candidate_actual_sha256=entry['actual_sha256'],reference_actual_sha256=re['actual_sha256'],candidate_derived_sha256=entry['derived_sha256'],reference_derived_sha256=re['derived_sha256'],candidate_selected_key_positions=a['key_positions'][pred].tolist(),reference_selected_key_positions=ra['key_positions'][ref].tolist(),unused_in_forward=True,
                        **quality(pred,ref,ra['probs'][0,head,0],a['logits_e'][0,head,0],ra['logits_e'][0,head,0],a['pv_head'][0,head,0],ra['pv_head'][0,head,0],p,a['key_positions'].numel())))
    # Each codec is re-encoded from this candidate's *same frozen original input*.
    for vid,group in groups.items():
        for key,(entry,meta,a,d) in group.items():
            references={'fp':(a['q_post_rope'].float(),a['k_visible'].float()),'source_int':(a['q_dq'],a['k_dq'])}
            for other,cfg in configs.items():
                if other==vid:continue
                reconstructed=[scalar_encode(a[prefix+'_codes'],a[prefix+'_scales'],fmt)['full_dq'] for prefix,fmt in zip(('q','k'),formats(cfg['attention_dense_format']))]
                references[other]=tuple(reconstructed)
            saved={}
            for role,(q,k) in references.items():
                logits=(q@k.transpose(-1,-2))/math.sqrt(128);probability=logits.softmax(-1);pv=probability@a['v_visible'].float()
                saved[role+'_logits']=logits;saved[role+'_probs']=probability;saved[role+'_pv_head']=pv
                for suffix,p in THRESHOLDS.items():
                    refmask,_,_=top_p(logits,a['allowed_mask'],a['key_positions'],p);saved[role+'_p'+suffix+'_mask']=refmask
            path=out/(vid+'-'+entry['record_id']+'-counterfactual.pt');torch.save(saved,path);artifacts.append(path);digest=sha256_file(path)
            for role in references:
                for suffix,p in THRESHOLDS.items():
                    for head in range(32):
                        pred=d['diagnostic_p'+suffix+'_mask'][0,head,0];ref=saved[role+'_p'+suffix+'_mask'][0,head,0]
                        rows.append(dict(comparison_kind='same_input_codec',candidate_variant_id=vid,reference_variant_id=role,p=p,head_id=head,layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],pair_identity_sha256=key,candidate_record_id=entry['record_id'],reference_record_id=entry['record_id'],candidate_actual_sha256=entry['actual_sha256'],candidate_derived_sha256=entry['derived_sha256'],reference_artifact=str(path),reference_artifact_sha256=digest,candidate_selected_key_positions=a['key_positions'][pred].tolist(),reference_selected_key_positions=a['key_positions'][ref].tolist(),unused_in_forward=True,
                            **quality(pred,ref,saved[role+'_probs'][0,head,0],a['logits_e'][0,head,0],saved[role+'_logits'][0,head,0],a['pv_head'][0,head,0],saved[role+'_pv_head'][0,head,0],p,a['key_positions'].numel())))
    expected=dict(cross_run_trajectory=len(trajectory_pairs)*576,same_input_codec=4*5*576)
    counts={kind:sum(row['comparison_kind']==kind for row in rows) for kind in expected};check(counts==expected,'pair coverage differs')
    jsonpath=out/'paired_sets.jsonl';jsonpath.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows));artifacts.append(jsonpath)
    csvpath=out/'set_comparison.csv'
    with csvpath.open('w',newline='') as f:
        fields=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    artifacts.append(csvpath)
    report=dict(schema_version='attention-dense-format-pairing-v1',status='complete',passed=True,counts=counts,expected_counts=expected,thresholds=list(THRESHOLDS.values()),historical_dense_controls=list(historical),unavailable_cross_run_controls=unavailable,source_files=sources,artifacts={str(p):sha256_file(p) for p in artifacts},estimands_separate=True,quality_is_measured_not_threshold_gated=True)
    write_json(out/'pairing_manifest.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('matrix','runs-root','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--controls-manifest');a=p.parse_args()
    try:pair_sets(a.matrix,a.runs_root,a.output_dir,a.controls_manifest)
    except Exception as exc:write_json(Path(a.output_dir)/'failure.json',dict(status='failed',passed=False,reason=str(exc)));raise
if __name__=='__main__':main()
