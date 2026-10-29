"""Independent stored-byte, exact-support, score-reuse and output acceptance."""
import argparse
import json
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from benchmark.attention_dense_format.oracle import check,compare,source_quantization,scalar_encode,scalar_decode,top_p,check_sets,ATOL,RTOL
from benchmark.attention_dense_format.model_replay import ModelRowReplay
from .formal import validate_formal_config,THRESHOLDS,TARGET_HASH

SCHEMA='attention-full-format-trace-v1'


def load_trace_records(run_dir):
    root=Path(run_dir).resolve();trace=root/'trace';index=load_json(trace/'index.json');check(index['schema_version']==SCHEMA and len(index['records'])==6,'six actual full-format records required')
    records=[];seen=set()
    for entry in index['records']:
        rid=entry['record_id'];check(rid not in seen,'duplicate trace');seen.add(rid);path=(trace/entry['path']).resolve();check(path.is_relative_to(trace),'unsafe trace path')
        for name,key in [('meta.json','meta_sha256'),('actual.pt','actual_sha256'),('derived.pt','derived_sha256')]:check(sha256_file(path/name)==entry[key],'trace index SHA '+name)
        meta=load_json(path/'meta.json')
        for name in ('actual.pt','derived.pt'):check(meta['files'][name]==sha256_file(path/name),'trace meta SHA '+name)
        records.append((entry,meta,torch.load(path/'actual.pt',map_location='cpu',weights_only=True),torch.load(path/'derived.pt',map_location='cpu',weights_only=True)))
    check({(m['layer_id'],m['query_position'],m['stage']) for _,m,_,_ in records}=={(l,q,s) for l in (0,16,31) for q,s in ((127,'prefill'),(128,'decode'))},'exact six targets differ')
    listing=root/'selection_sets.jsonl';check(index['selection_sets']['sha256']==sha256_file(listing) and index['selection_sets']['rows']==192 and index['selection_sets']['unused_in_forward'] is False,'actual forward selected-key listing binding')
    rows=[json.loads(line) for line in listing.read_text().splitlines() if line];check(len(rows)==192,'selected-key listing coverage');lookup={e['record_id']:(e,m,a) for e,m,a,_ in records};seenrows=set()
    for row in rows:
        key=(row['record_id'],row['head_id']);check(key not in seenrows,'duplicate selected-key row');seenrows.add(key);entry,meta,a=lookup[row['record_id']];head=row['head_id'];check(head in range(32),'invalid head')
        mask=a['selected_mask'][0,head,0];check(row['p']==meta['p'] and row['variant_id']==meta['variant_id'] and row['unused_in_forward'] is False and row['final_operand_profile']=='decoded_full_format','listing execution identity')
        check(row['key_positions']==a['key_positions'].tolist() and row['selected_key_positions']==a['key_positions'][mask].tolist() and row['num_selected']==int(mask.sum()) and row['allowed_count']==mask.numel(),'listing set differs')
        for field in ('meta_sha256','actual_sha256','derived_sha256'):check(row[field]==entry[field],'listing source hash')
        ref=row['mask_ref'];check(ref['field']=='selected_mask' and ref['index']==[0,head,0] and ref['sha256']==entry['actual_sha256'] and Path(ref['file']).resolve()==(trace/entry['path']/'actual.pt').resolve(),'selected mask evidence path')
    return records


def check_selection(a,mask,positions):
    compare(a['selected_mask'],mask.cpu(),'exact forward selection',True);counts=mask.sum(-1).cpu();compare(a['selected_counts'],counts,'selected counts',True)
    indices=a['selected_indices'];valid=a['selected_valid_mask'];check(indices.dtype==torch.int64 and valid.dtype==torch.bool and indices.shape==valid.shape,'selected index layout')
    check(torch.equal(valid.sum(-1),counts) and (indices[~valid]==-1).all(),'selected padding/counts')
    compare(((indices[...,None]==positions)&valid[...,None]).any(-2),mask.cpu(),'actual selected IDs/mask',True)
    check((indices[...,1:][valid[...,1:]]>=indices[...,:-1][valid[...,1:]]).all(),'selected indices not ascending')


def replay_record(entry,meta,a,d,config,device,model=None,failure_dir=None):
    s=config['attention_full_format'];p=s['selection']['p'];check(meta['attention_full_format']==s and meta['selection_passes']==1 and meta['selection_mode']=='top_p','full-format execution config')
    check(meta['final_operand_profile']=='decoded_full_format' and meta['final_operand_origin']=='stored_format_bytes' and meta['selected_mask_unused_in_forward'] is False,'final operand/mask semantics')
    n=meta['query_position']+1;positions=torch.arange(n,dtype=torch.int64)
    check(meta['sample_id']==0 and meta['window_id'] is None and meta['predicted_target']==meta['query_position']+1,'sample/target mismatch')
    check((meta['Hq'],meta['Hkv'],meta['D'])==(32,8,128) and meta['head_to_kv']==[h//4 for h in range(32)],'GQA identity')
    compare(a['key_positions'],positions,'absolute causal key positions',True);compare(a['allowed_mask'],torch.ones((1,1,1,n),dtype=torch.bool),'actual causal domain',True)
    check(meta['cache_length']==meta['expected_cache_length']==n and meta['cache_storage']=='original_float','cache context mismatch')
    check(a['q_post_rope'].shape==(1,32,1,128) and a['k_visible'].shape==a['v_visible'].shape==(1,32,n,128),'actual Q/K/V shape')
    reconstructed=[]
    for prefix in ('q','k'):
        original=a['q_post_rope' if prefix=='q' else 'k_visible'];check(original.dtype==torch.bfloat16,'original model dtype changed')
        c,scale=source_quantization(original);compare(a[prefix+'_codes'],c,'source codes '+prefix,True);compare(a[prefix+'_scales'],scale,'source scales '+prefix,True)
        compare(a[prefix+'_dq'],c.float()*scale.float()[...,None],'source DQ '+prefix,True)
        expected=scalar_encode(c,scale,s['format'])
        for field,value in expected.items():compare(a[prefix+'_format_'+field],value,'scalar re-encode '+prefix+' '+field,True)
        decoded=scalar_decode(a[prefix+'_format_full_codes'],s['format'],scales=scale,e8m0=a.get(prefix+'_format_e8m0'));compare(a[prefix+'_format_full_dq'],decoded,'stored-byte decode '+prefix,True);reconstructed.append(decoded.to(device))
        if s['format']=='mxint8':compare(a[prefix+'_format_exponent'],a[prefix+'_format_e8m0'].to(torch.int32)-127,'E8M0 exponent',True)
    q,k=reconstructed;allowed=a['allowed_mask'].to(device);actual=a['full_format_logits'].to(device)
    regenerated=(q@k.transpose(-1,-2))/math.sqrt(128);errors={'full_format_logits':compare(a['full_format_logits'],regenerated.cpu(),'independent format logits')}
    mask,probs,_=top_p(actual,allowed,positions,p);check_selection(a,mask,positions);compare(a['full_format_probs'],probs.cpu(),'actual full-score softmax')
    regen_mask,_,_=top_p(regenerated,allowed,positions,p)
    if not torch.equal(mask,regen_mask):
        if failure_dir:
            path=Path(failure_dir);path.mkdir(parents=True,exist_ok=True);torch.save(dict(actual_logits=actual.cpu(),regenerated_logits=regenerated.cpu(),actual_mask=mask.cpu(),regenerated_mask=regen_mask.cpu(),p=p),path/(entry['record_id']+'-membership-failure.pt'))
        raise ValueError('Regenerated full-format selected support differs; unresolved arithmetic')
    selected=actual.masked_fill(~mask,-torch.inf);final_probs=selected.softmax(-1,dtype=torch.float32);pv=final_probs@a['v_visible'].to(device).float()
    compare(a['final_selected_logits'],a['full_format_logits'].masked_fill(~a['selected_mask'],-torch.inf),'same retained full-format scores',True)
    for field,expected in [('final_selected_logits',selected),('final_selected_probs',final_probs),('pv_head',pv)]:errors[field]=compare(a[field],expected.cpu(),'independent selected '+field)
    check((a['final_selected_probs'][~a['selected_mask']]==0).all(),'probability outside selected support')
    compare(a['o_proj_input'],a['pv_head'].to(torch.bfloat16).transpose(1,2).reshape(1,1,4096),'exact PV projection boundary',True)
    for role,qq,kk in [('fp',a['q_post_rope'].float(),a['k_visible'].float()),('source_int',a['q_dq'],a['k_dq'])]:
        reference=(qq@kk.transpose(-1,-2))/math.sqrt(128);compare(d['reference_'+role+'_logits_same_input'],reference,'same-input '+role+' logits',True);compare(d['reference_'+role+'_probs_same_input'],reference.softmax(-1),'same-input '+role+' probs',True)
        remap={key.removeprefix('reference_'+role+'_').removesuffix('_same_input'):value for key,value in d.items() if key.startswith('reference_'+role+'_') and key.endswith('_same_input')}
        for suffix,threshold in THRESHOLDS.items():
            refmask,_,_=top_p(reference,a['allowed_mask'],positions,threshold);check_sets(remap,refmask,positions,'diagnostic_p'+suffix+'_')
    compare(d['full_format_dense_pv_same_input'],a['full_format_probs']@a['v_visible'].float(),'same-input full-format dense PV',True)
    model_report=model.replay(meta,a) if model else dict(status='not_executed')
    return dict(record_id=entry['record_id'],passed=True,byte_decode_exact=True,actual_support_exact=True,regenerated_support_exact=True,retained_scores_exact=True,selection_passes=1,final_operand_profile='decoded_full_format',errors=errors,model_row_replay=model_report)


def replay_run(run_dir,device='cuda',model_weights=True):
    root=Path(run_dir).resolve();check(load_json(root/'status.json')['status']=='complete','captured run incomplete');resolved=load_json(root/'resolved_config.json');config=resolved['algorithm'];validate_formal_config(config)
    m=load_json(root/'metrics.json');assets=load_json(root/'asset_manifest.json');code=load_json(root/'code_manifest.json');check(m['semantic_config_sha256']==canonical_hash(config),'semantic config hash')
    check(m['code_sha256']==code['code_sha256']==canonical_hash(dict(head=code['head'],files=code['files'])) and code['patch_sha256']==sha256_file(root/'code.patch'),'code/patch identity')
    check(m['asset_manifest_sha256']==canonical_hash(assets) and m['capture_config_sha256']==TARGET_HASH,'asset/target identity');model=ModelRowReplay(resolved['model_path'],device) if model_weights else None;reports=[]
    for entry,meta,a,d in load_trace_records(root):
        check(meta['variant_id']==m['variant_id']==config['attention_full_format']['variant_id'],'variant mismatch')
        for key in ('code_sha256','tokens_sha256','environment_sha256','asset_manifest_sha256','subset_sha256','scoring_ranges_sha256'):check(meta[key]==m[key],'trace/run identity '+key)
        check(meta['model_file_hashes']==assets['model_file_hashes'] and meta['config_sha256']==m['semantic_config_sha256'],'model/config identity')
        reports.append(replay_record(entry,meta,a,d,config,device,model,root/'replay_failures'))
    return dict(schema_version='attention-full-format-replay-v1',status='complete',passed=True,run_dir=str(root),num_records=len(reports),records=reports,atol=ATOL,rtol=RTOL,model_weights_replayed=model_weights,device=str(device),index_sha256=sha256_file(root/'trace/index.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    try:report=replay_run(a.run_dir,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc),atol=ATOL,rtol=RTOL));raise
    write_json(a.output,report)
if __name__=='__main__':main()
