"""Pair candidate with three historical trajectories and its own INT-full oracle."""
import argparse
import csv
import json
from pathlib import Path
from benchmark.attention_baseline.common import canonical_hash,load_json,sha256_file,write_json
from benchmark.attention_slice.pair_sets import identity as slice_identity,set_metrics,_selection_rows
from benchmark.attention_slice.replay import load_trace_records as load_historical,replay_run as replay_historical,check_set_fields
from .replay import load_trace_records,replay_run
from .oracle import check
from .formal import variant_contract


def identity(meta,actual,head):
    ident=slice_identity(meta,actual,head)
    for key in ('environment_sha256','model_dtype','protocol_id','profile','mode','subset_sha256',
                'scoring_ranges_sha256','context_length','stride','query_chunk','prefill_tokens','decode_targets',
                'Hq','Hkv','D','predicted_target','input_ids','position_ids','cache_position',
                'global_query_positions','cache_length','expected_cache_length'):
        check(key in meta,'missing full pairing identity: '+key);ident[key]=meta[key]
    return ident


def pair_candidate(candidate_run,int_run,fp_run,tc_run,output_dir,device='cuda'):
    paths={'candidate':Path(candidate_run).resolve(),'int':Path(int_run).resolve(),'fp':Path(fp_run).resolve(),'tc':Path(tc_run).resolve()}
    reports={'candidate':replay_run(paths['candidate'],device)};groups={};sources={};rows=[]
    cfg=load_json(paths['candidate']/'resolved_config.json');vid=cfg.get('algorithm',cfg)['attention_slice']['variant_id'];fmt,bq,bk,p=variant_contract(vid)
    suffix=next(s for s,v in [('095',.95),('090',.9),('085',.85)] if v==p)
    expected={'int':f'int_topp_p{suffix}','fp':f'fp_topp_p{suffix}','tc':f'msb_q{bq}k4_topp_p{suffix}'}
    for role,run in paths.items():
        check(load_json(run/'status.json')['status']=='complete','incomplete pairing source')
        if role!='candidate':reports[role]=replay_historical(run)
        records=(load_trace_records if role=='candidate' else load_historical)(run)
        # Validate saved masks, counts and separately consumable set listing for every source.
        _selection_rows(run,records);index=load_json(run/'trace/index.json');check(index['selection_sets']['sha256']==sha256_file(run/'selection_sets.jsonl'),'selection listing hash')
        group={}
        for entry,meta,a,derived in records:
            check(meta['variant_id']==(vid if role=='candidate' else expected[role]),'wrong same-p/shape reference')
            if role!='candidate':
                from .oracle import top_p
                mask,_,_=top_p(a['estimator_logits'].to(device),a['allowed_mask'],a['key_positions'],p)
                check_set_fields(a,mask.cpu(),a['key_positions'])
            for head in range(32):
                key=canonical_hash(identity(meta,a,head));check(key not in group,'duplicate pairing identity');group[key]=(entry,meta,a,derived,head)
        groups[role]=group
        for name in ('status.json','metrics.json','resolved_config.json','asset_manifest.json','code_manifest.json','code.patch','environment.json','selection_sets.jsonl','trace/index.json'):
            path=run/name;check(path.is_file(),'missing pairing provenance '+str(path));sources[str(path)]=sha256_file(path)
        for entry,_,_,_ in records:
            for name in ('actual.pt','derived.pt','meta.json'):
                path=run/'trace'/entry.get('path',entry['record_id'])/name;sources[str(path)]=sha256_file(path)
    for role in ('int','fp','tc'):check(set(groups[role])==set(groups['candidate']),'trajectory identity/domain differs')
    for key,(entry,meta,a,d,head) in groups['candidate'].items():
        pred=a['selected_mask'][0,head,0];ident=identity(meta,a,head)
        base=dict(predicted_variant_id=vid,pair_identity_sha256=key,**ident,predicted_record_id=entry['record_id'],predicted_actual_sha256=entry['actual_sha256'],predicted_selected_key_positions=a['key_positions'][pred].tolist())
        for role in ('int','fp','tc'):
            re,rm,ra,rd,_=groups[role][key];ref=ra['selected_mask'][0,head,0]
            rows.append(dict(base,comparison_kind='cross_run_trajectory',reference_variant_id=rm['variant_id'],reference_role=role,reference_record_id=re['record_id'],reference_actual_sha256=re['actual_sha256'],candidate_code_sha256=meta['code_sha256'],reference_code_sha256=rm['code_sha256'],cross_run_different_code_allowed=True,reference_selected_key_positions=ra['key_positions'][ref].tolist(),**set_metrics(pred,ref,ra['estimator_probs'][0,head,0],p)))
        ref=d['reference_int_top_p_mask_same_input'][0,head,0]
        rows.append(dict(base,comparison_kind='same_input_estimator',reference_variant_id='original_INT_full_same_input',reference_role='own_source_int_full',reference_record_id=entry['record_id'],reference_actual_sha256=entry['actual_sha256'],reference_selected_key_positions=a['key_positions'][ref].tolist(),**set_metrics(pred,ref,d['reference_int_probs_same_input'][0,head,0],p)))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);jsonpath=output/'paired_sets.jsonl';csvpath=output/'set_comparison.csv'
    jsonpath.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows))
    fields=sorted(set().union(*(row.keys() for row in rows)))
    with csvpath.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    counts={kind:sum(r['comparison_kind']==kind for r in rows) for kind in ('cross_run_trajectory','same_input_estimator')};check(counts==dict(cross_run_trajectory=576,same_input_estimator=192),'pair coverage')
    tails={}
    for role in ('int','fp','tc','own_source_int_full'):
        group=[r for r in rows if r['reference_role']==role];tails[role]={}
        for metric in ('recall','precision','jaccard','retained_mass','shortfall'):
            vals=sorted(r[metric] for r in group);tails[role][metric]=dict(mean=sum(vals)/len(vals),min=vals[0],max=vals[-1],p01=vals[int(.01*(len(vals)-1))],p05=vals[int(.05*(len(vals)-1))],p95=vals[int(.95*(len(vals)-1))],p99=vals[int(.99*(len(vals)-1))])
    report=dict(schema_version='attention-format-pairing-v1',status='complete',passed=True,candidate_variant_id=vid,counts=counts,num_rows=len(rows),quality_is_measured_not_gated=True,source_files=sources,artifacts={str(x.resolve()):sha256_file(x) for x in (jsonpath,csvpath)},summary=tails,replay_reports=reports,trajectory_note='Historical model trajectories may use different source hashes; full asset/token/domain/head/threshold identity is mandatory; own-source oracle is reported separately.')
    write_json(output/'pairing_manifest.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('candidate-run','int-run','fp-run','tc-run','output-dir'):p.add_argument('--'+flag,required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    try:pair_candidate(a.candidate_run,a.int_run,a.fp_run,a.tc_run,a.output_dir,a.device)
    except Exception as exc:write_json(Path(a.output_dir)/'pairing_failure.json',dict(status='failed',passed=False,reason=str(exc)));raise
if __name__=='__main__':main()
