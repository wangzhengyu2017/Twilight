"""Independent acceptance from stored source codes/scales and actual format bytes."""
import argparse
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash
from .formal import validate_formal_config,THRESHOLDS,TARGET_HASH
from .oracle import check,compare,source_quantization,scalar_encode,scalar_decode,formats,top_p,check_sets,ATOL,RTOL

SCHEMA='attention-dense-format-trace-v1'


def load_trace_records(run_dir):
    root=Path(run_dir).resolve();trace=root/'trace';index=load_json(trace/'index.json')
    check(index['schema_version']==SCHEMA and len(index['records'])==6,'six authentic dense records required')
    records=[];seen=set()
    for entry in index['records']:
        rid=entry['record_id'];check(rid not in seen,'duplicate trace');seen.add(rid)
        path=(trace/entry['path']).resolve();check(path.is_relative_to(trace),'unsafe trace path')
        for name,key in [('meta.json','meta_sha256'),('actual.pt','actual_sha256'),('derived.pt','derived_sha256')]:check(sha256_file(path/name)==entry[key],'trace index hash '+name)
        meta=load_json(path/'meta.json')
        for name,key in [('actual.pt','actual_sha256'),('derived.pt','derived_sha256')]:check(meta['files'][name]==entry[key],'trace meta hash')
        records.append((entry,meta,torch.load(path/'actual.pt',map_location='cpu',weights_only=True),torch.load(path/'derived.pt',map_location='cpu',weights_only=True)))
    check({(m['layer_id'],m['query_position'],m['stage']) for _,m,_,_ in records}=={(l,q,s) for l in (0,16,31) for q,s in ((127,'prefill'),(128,'decode'))},'exact trace target coverage')
    check(index['diagnostic_sets']['unused_in_forward'] is True and index['diagnostic_sets']['rows']==576,'diagnostic listing metadata')
    listing=root/'diagnostic_sets.jsonl';check(sha256_file(listing)==index['diagnostic_sets']['sha256'],'diagnostic listing changed')
    rows=[__import__('json').loads(s) for s in listing.read_text().splitlines() if s]
    check(len(rows)==576,'diagnostic listing coverage');lookup={e['record_id']:(e,m,a,d) for e,m,a,d in records};seenrows=set()
    for row in rows:
        key=(row['record_id'],row['head_id'],row['p']);check(key not in seenrows,'duplicate diagnostic listing');seenrows.add(key)
        e,m,a,d=lookup[row['record_id']];head=row['head_id'];check(head in range(32) and row['p'] in THRESHOLDS.values(),'diagnostic listing identity')
        label=next(k for k,v in THRESHOLDS.items() if v==row['p']);prefix='diagnostic_p'+label+'_';mask=d[prefix+'mask'][0,head,0]
        check(row['unused_in_forward'] is True and row['selected_key_positions']==a['key_positions'][mask].tolist() and row['num_selected']==int(mask.sum()),'diagnostic listing differs from actual saved set')
        check(row['actual_execution_count']==int(a['execution_counts'][0,head,0])==row['allowed_count'],'diagnostic altered execution count')
        for field in ('meta_sha256','actual_sha256','derived_sha256'):check(row[field]==e[field],'diagnostic listing trace binding')
        ref=row['mask_ref'];check(ref['field']==prefix+'mask' and ref['sha256']==e['derived_sha256'] and ref['index']==[0,head,0],'diagnostic mask binding')
        check(Path(ref['file']).resolve()==(trace/e['path']/'derived.pt').resolve(),'diagnostic mask path')
    return records


def replay_record(entry,meta,a,derived,config,device,model=None):
    s=config['attention_dense_format'];check(meta['attention_dense_format']==s,'trace dense config')
    check(meta['selection_mode']=='dense' and meta['selection_passes']==0 and meta['final_operand_origin']=='stored_format_bytes','dense execution semantics')
    check(meta['diagnostic_sets']['unused_in_forward'] is True,'diagnostic must not execute')
    check(meta['sample_id']==0 and meta['window_id'] is None and meta['predicted_target']==meta['query_position']+1,'sample/target identity')
    check((meta['Hq'],meta['Hkv'],meta['D'])==(32,8,128) and meta['head_to_kv']==[h//4 for h in range(32)],'GQA metadata')
    n=meta['query_position']+1;positions=torch.arange(n,dtype=torch.int64)
    compare(a['key_positions'],positions,'absolute causal positions',True)
    check(meta['cache_length']==meta['expected_cache_length']==n,'cache context length')
    check(meta['cache_storage']=='original_float' and a['k_visible'].dtype==a['v_visible'].dtype==a['q_post_rope'].dtype==torch.bfloat16,'original Q/K/V/cache dtype')
    check(a['q_post_rope'].shape==(1,32,1,128) and a['k_visible'].shape==a['v_visible'].shape==(1,32,n,128),'Q/K/V shape')
    compare(a['allowed_mask'],torch.ones((1,1,1,n),dtype=torch.bool),'captured causal domain',True)
    execution=a['allowed_mask'].expand(1,32,1,n);compare(a['execution_mask'],execution,'all legal execution',True)
    compare(a['execution_counts'],execution.sum(-1),'actual execution count',True)
    compare(a['execution_indices'],positions.expand(1,32,1,n),'actual execution indices',True)
    compare(a['execution_valid_mask'],execution,'actual execution validity',True)
    reconstructed=[];errors={}
    for prefix,fmt in zip(('q','k'),formats(s)):
        original=a['q_post_rope' if prefix=='q' else 'k_visible'];codes,scales=source_quantization(original)
        compare(a[prefix+'_codes'],codes,'independent source '+prefix+' codes',True);compare(a[prefix+'_scales'],scales,'independent source '+prefix+' scales',True)
        compare(a[prefix+'_dq'],codes.float()*scales.float()[...,None],'source dq',True)
        expected=scalar_encode(codes,scales,fmt)
        for name,value in expected.items():compare(a[prefix+'_format_'+name],value,'scalar encoder '+prefix+' '+name,True)
        decoded=scalar_decode(a[prefix+'_format_full_codes'],fmt,scales=scales,e8m0=a.get(prefix+'_format_e8m0'))
        compare(a[prefix+'_format_full_dq'],decoded,'stored bytes decode '+prefix,True);reconstructed.append(decoded.to(device))
        if fmt!='sm8':compare(a[prefix+'_format_exponent'],a[prefix+'_format_e8m0'].to(torch.int32)-127,'E8M0 actual exponent',True)
        groups=original.reshape(1,8,4,*original.shape[2:]) if prefix=='k' else None
        if groups is not None:check(torch.equal(groups,groups[:,:,:1].expand_as(groups)),'GQA cache head replication')
    q,k=reconstructed
    logits=(q@k.transpose(-1,-2))/math.sqrt(128);probs=logits.softmax(-1);pv=probs@a['v_visible'].to(device).float()
    for name,expected in [('logits_e',logits),('probs',probs),('pv_head',pv)]:errors[name]=compare(a[name],expected.cpu(),'independent decoded '+name)
    compare(a['o_proj_input'],a['pv_head'].to(torch.bfloat16).transpose(1,2).reshape(1,1,4096),'exact PV output cast/reshape',True)
    for label,p in THRESHOLDS.items():
        mask,_,_=top_p(a['logits_e'],a['allowed_mask'],positions,p);check_sets(derived,mask,positions,'diagnostic_p'+label+'_')
    for label,qq,kk in [('fp',a['q_post_rope'].float(),a['k_visible'].float()),('source_int',a['q_dq'],a['k_dq'])]:
        reference=(qq@kk.transpose(-1,-2))/math.sqrt(128);reference_probs=reference.softmax(-1)
        compare(derived['reference_'+label+'_logits_same_input'],reference,'same-input reference logits',True)
        compare(derived['reference_'+label+'_probs_same_input'],reference_probs,'same-input reference probs',True)
        for suffix,p in THRESHOLDS.items():
            mask,_,_=top_p(reference,a['allowed_mask'],positions,p)
            remap={key.removeprefix('reference_'+label+'_').removesuffix('_same_input'):value for key,value in derived.items() if key.startswith('reference_'+label+'_') and key.endswith('_same_input')}
            check_sets(remap,mask,positions,'diagnostic_p'+suffix+'_')
    model_report=model.replay(meta,a) if model is not None else dict(status='not_executed')
    return dict(record_id=entry['record_id'],passed=True,source_and_bytes_exact=True,all_legal_execution=True,selection_passes=0,diagnostics_unused_in_forward=True,errors=errors,model_row_replay=model_report)


def replay_run(run_dir,device='cuda',model_weights=True):
    root=Path(run_dir).resolve();check(load_json(root/'status.json')['status']=='complete','incomplete captured run')
    resolved=load_json(root/'resolved_config.json');config=resolved['algorithm'];validate_formal_config(config)
    m=load_json(root/'metrics.json');assets=load_json(root/'asset_manifest.json');code=load_json(root/'code_manifest.json')
    check(m['semantic_config_sha256']==canonical_hash(config),'run config hash')
    check(m['code_sha256']==code['code_sha256']==canonical_hash(dict(head=code['head'],files=code['files'])),'run code hash')
    check(code['patch_sha256']==sha256_file(root/'code.patch'),'run patch hash')
    check(m['asset_manifest_sha256']==canonical_hash(assets),'run assets hash')
    check(m['capture_config_sha256']==TARGET_HASH,'frozen trace target source')
    model=None
    if model_weights:
        from .model_replay import ModelRowReplay
        model=ModelRowReplay(resolved['model_path'],device)
    reports=[]
    for entry,meta,a,d in load_trace_records(root):
        check(meta['variant_id']==m['variant_id']==config['attention_dense_format']['variant_id'],'trace variant')
        for key in ('code_sha256','tokens_sha256','environment_sha256','asset_manifest_sha256','subset_sha256','scoring_ranges_sha256'):check(meta[key]==m[key],'trace/run identity '+key)
        check(meta['model_file_hashes']==assets['model_file_hashes'],'trace model identity')
        check(meta['config_sha256']==m['semantic_config_sha256'],'trace config identity')
        reports.append(replay_record(entry,meta,a,d,config,device,model))
    return dict(schema_version='attention-dense-format-replay-v1',status='complete',passed=True,run_dir=str(root),num_records=len(reports),records=reports,atol=ATOL,rtol=RTOL,byte_decode_exact=True,model_weights_replayed=model_weights,device=str(device),index_sha256=sha256_file(root/'trace/index.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    try:report=replay_run(a.run_dir,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc),atol=ATOL,rtol=RTOL));raise
    write_json(a.output,report)
if __name__=='__main__':main()
