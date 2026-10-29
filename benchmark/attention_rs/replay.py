"""Independent source, packed-RS, selector, final-score and model-row replay."""
import argparse
import json
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file,canonical_hash,configure_runtime,environment
from .formal import validate_formal_config,TARGET_HASH,PAPER
from .oracle import check,compare,source_quantization,source_sidecar,decode_words,selection,semantic_fp64,check_set,ATOL,RTOL
SCHEMA='attention-rs-trace-v1'
THRESHOLDS={'095':.95,'090':.9,'085':.85}

def load_trace_records(run_dir):
    root=Path(run_dir).resolve();trace=root/'trace';index=load_json(trace/'index.json');check(index['schema_version']==SCHEMA and len(index['records'])==6,'six actual RS captures required');records=[];seen=set()
    for entry in index['records']:
        rid=entry['record_id'];check(rid not in seen,'duplicate RS record');seen.add(rid);path=(trace/entry['path']).resolve();check(path.is_relative_to(trace),'unsafe RS trace path')
        for file,key in [('meta.json','meta_sha256'),('actual.pt','actual_sha256'),('derived.pt','derived_sha256')]:check(sha256_file(path/file)==entry[key],'RS trace hash '+file)
        meta=load_json(path/'meta.json');check(meta['files']['actual.pt']==entry['actual_sha256'] and meta['files']['derived.pt']==entry['derived_sha256'],'RS meta tensor bindings')
        records.append((entry,meta,torch.load(path/'actual.pt',map_location='cpu',weights_only=True),torch.load(path/'derived.pt',map_location='cpu',weights_only=True)))
    check({(m['layer_id'],m['query_position'],m['stage']) for _,m,_,_ in records}=={(l,q,s) for l in (0,16,31) for q,s in ((127,'prefill'),(128,'decode'))},'RS exact target coverage')
    lookup={entry['record_id']:(entry,meta,a,d) for entry,meta,a,d in records}
    for name,total in [('selection_sets',192),('diagnostic_sets',576)]:
        file=root/(name+'.jsonl');reference=index[name];check(reference['rows']==total and Path(reference['path']).resolve()==file and sha256_file(file)==reference['sha256'],'RS set listing identity')
        rows=[json.loads(line) for line in file.read_text().splitlines() if line];check(len(rows)==total,'RS listing row coverage');keys=set()
        for row in rows:
            entry,meta,a,d=lookup[row['record_id']];head=row['head_id'];key=(row['record_id'],head,row['p']);check(key not in keys and head in range(32),'RS set duplicate/head');keys.add(key)
            for field in ('meta_sha256','actual_sha256','derived_sha256'):check(row[field]==entry[field],'RS set tensor binding')
            check(row['selection_passes']==meta['selection_passes'] and row['variant_id']==meta['variant_id'] and row['final_operand_origin']==meta['final_operand_origin'],'RS set metadata')
            if name=='selection_sets':
                mask=a['selected_mask'][0,head,0];check(row['set_role']==('actual_all_legal_execution' if meta['rs_mode']=='dense_full' else 'actual_used_prediction_set'),'RS execution role')
                check(row['p']==meta['attention_rs']['selection']['p'],'RS actual selection p')
            else:
                check(row['unused_in_forward'] is True and row['set_role']=='offline_native_full_important_keys' and row['p'] in THRESHOLDS.values(),'RS diagnostic role')
                suffix=next(k for k,p in THRESHOLDS.items() if p==row['p']);field='native_full_diagnostic_p'+suffix+'_mask';mask=d[field][0,head,0];ref=row['mask_ref']
                check(Path(ref['file']).resolve()==(trace/entry['path']/'derived.pt') and ref['sha256']==entry['derived_sha256'] and ref['field']==field and ref['index']==[0,head,0],'RS diagnostic mask binding')
            check(row['selected_key_positions']==a['key_positions'][mask].tolist() and row['num_selected']==int(mask.sum()),'RS set listing/mask mismatch')
    return records

def selected_original_logits(qc,qs,kc,ks,indices,valid,device):
    # Reconstruct selected source operands independently; never import production selected helper.
    qcodes=qc.to(device);qscale=qs.to(device);kcodes=kc.to(device);kscale=ks.to(device);ix=indices.to(device);vi=valid.to(device)
    final=torch.full((*qc.shape[:-1],kc.shape[-2]),-torch.inf,device=device,dtype=torch.float32)
    for batch in range(qc.shape[0]):
        for head in range(qc.shape[1]):
            for row in range(qc.shape[2]):
                selected=ix[batch,head,row][vi[batch,head,row]];q=qcodes[batch,head,row].float()*qscale[batch,head,row].float();k=kcodes[batch,head,selected].float()*kscale[batch,head,selected].float()[:,None]
                final[batch,head,row,selected]=torch.bmm(q.reshape(1,1,-1),k[None].transpose(-1,-2)).reshape(-1)/math.sqrt(q.shape[-1])
    return final

def replay_record(entry,meta,a,d,config,device,model=None):
    s=config['attention_rs'];check(meta['attention_rs']==s and meta['variant_id']==s['variant_id'],'RS config observation')
    for key in ('mode','selection_passes','final_operand_origin','final_scores','final_execution'):check(meta['rs_mode' if key=='mode' else key]==s[key],'RS actual '+key)
    check(meta['paper_sha256']==PAPER and meta['diagnostic_sets_unused_in_forward'] is True,'RS paper/diagnostic identity')
    check(meta['sample_id']==0 and meta['window_id'] is None and meta['predicted_target']==meta['query_position']+1,'RS sample/target')
    check((meta['Hq'],meta['Hkv'],meta['D'])==(32,8,128) and meta['head_to_kv']==[h//4 for h in range(32)],'RS GQA metadata')
    n=meta['query_position']+1;positions=torch.arange(n,dtype=torch.int64);compare(a['key_positions'],positions,'RS key positions',True)
    check(meta['cache_length']==meta['expected_cache_length']==n and meta['cache_storage']=='original_float','RS cache state')
    check(a['q_post_rope'].shape==(1,32,1,128) and a['k_visible'].shape==a['v_visible'].shape==(1,32,n,128),'RS original shapes')
    check(a['q_post_rope'].dtype==a['k_visible'].dtype==a['v_visible'].dtype==torch.bfloat16,'RS original BF16 operands')
    compare(a['allowed_mask'],torch.ones((1,1,1,n),dtype=torch.bool),'RS legal domain',True)
    sides=[];errors={}
    for prefix,bits in [('q',s['q_bits']),('k',s['k_bits'])]:
        original=a['q_post_rope' if prefix=='q' else 'k_visible'];codes,scales=source_quantization(original)
        compare(a[prefix+'_codes'],codes,'independent source codes '+prefix,True);compare(a[prefix+'_scales'],scales,'independent unchanged scale '+prefix,True);compare(a[prefix+'_dq'],codes.float()*scales.float()[...,None],'original source DQ '+prefix,True)
        expected=source_sidecar(codes,scales,bits)
        for field,value in expected.items():compare(a[prefix+'_rs_'+field],value,'independent RS '+prefix+' '+field,True)
        decoded=decode_words(a[prefix+'_rs_rs_words']);compare(decoded,expected['native_int'],'actual packed bytes inverse '+prefix,True)
        stored=a[prefix+'_rs_retained_bytes'];retained=decode_words(stored) if bits==8 else 16*torch.where(stored<8,stored.to(torch.int16),stored.to(torch.int16)-16)
        compare(a[prefix+'_rs_predictor_dq'],retained.float()*scales.float()[...,None],'actual retained bytes DQ '+prefix,True);sides.append(expected)
        budget=meta['storage'][prefix];check(budget['logical_payload_bits']==codes.numel()*bits and budget['required_source_scale_bits']==scales.numel()*16 and budget['bits_per_element_with_scale']==bits+.125,'RS truthful logical budgets')
        for field,value in expected.items():check(budget['fields'][field]['bytes']==value.numel()*value.element_size(),'RS actual allocation '+field)
        if prefix=='k':
            for name in ('k_visible','v_visible','k_codes','k_scales'):
                value=a[name];group=value.reshape(1,8,4,*value.shape[2:]);check(torch.equal(group,group[:,:,:1].expand_as(group)),'RS GQA expanded '+name)
    qs,ks=sides;q,k=qs['predictor_dq'].to(device),ks['predictor_dq'].to(device)
    logits=(q@k.transpose(-1,-2))/torch.tensor(math.sqrt(128),device=device,dtype=torch.float32)
    errors['estimator_logits']=compare(a['estimator_logits'],logits.cpu(),'RS DQ-first logits')
    # Membership acceptance is exact against replay of execution FP32 logits, not tolerance masks.
    mask,indices,counts,valid,ep=selection(logits,a['allowed_mask'].to(device),s['selection']['p'],positions)
    compare(a['selected_mask'],mask.cpu(),'independent FP32 selected membership',True);check_set(a,mask.cpu(),positions)
    errors['estimator_probs']=compare(a['estimator_probs'],ep.cpu(),'RS estimator softmax')
    if s['mode']=='msb_topp':
        safe=torch.where(a['selected_valid_mask'],a['selected_indices'],a['selected_indices'][...,:1]);batch=torch.arange(1)[:,None,None,None];head=torch.arange(32)[None,:,None,None]
        compare(a['selected_source_k_codes'],a['k_codes'][batch,head,safe],'selected original K codes',True);compare(a['selected_source_k_scales'],a['k_scales'][batch,head,safe],'selected original K scales',True)
        compare(a['selected_source_q_codes'],a['q_codes'],'selected original Q codes',True);compare(a['selected_source_q_scales'],a['q_scales'],'selected original Q scales',True)
        # Production padding belongs to the entire original query chunk. A captured
        # row can select fewer keys than that chunk's largest row; preserve every
        # saved slot and independently verify its prefix-valid/sentinel contract.
        saved_indices=a['selected_indices'];saved_valid=a['selected_valid_mask'];width=saved_indices.shape[-1]
        check(0<width<=n,'saved selected chunk width')
        expected_valid=torch.arange(width)<counts.cpu()[...,None]
        compare(saved_valid,expected_valid,'saved chunk padding validity',True)
        expected_indices=positions.expand_as(mask.cpu()).masked_fill(~mask.cpu(),n).sort(-1).values[...,:width].masked_fill(~expected_valid,-1)
        compare(saved_indices,expected_indices,'saved chunk padded original positions',True)
        final=selected_original_logits(a['q_codes'],a['q_scales'],a['k_codes'],a['k_scales'],saved_indices,saved_valid,device)
        selected_expected=final.gather(-1,saved_indices.to(device).clamp_min(0)).masked_fill(~saved_valid.to(device),-torch.inf)
        compare(a['selected_source_logits'],selected_expected.cpu(),'actual selected-source logits')
    else:
        check(not any(name.startswith('selected_source_') for name in a),'native final cannot contain selected-source recompute operands')
        # Reuse must be bitwise to the actual saved native selector logits on S.
        compare(a['final_selected_logits'],a['estimator_logits'].masked_fill(~a['selected_mask'],-torch.inf),'native full actual logits reused',True)
        final=logits.masked_fill(~mask,-torch.inf)
    probs=final.softmax(-1);pv=probs@a['v_visible'].to(device).float()
    for field,value in [('final_selected_logits',final),('final_selected_probs',probs),('pv_head',pv)]:errors[field]=compare(a[field],value.cpu(),'RS replay '+field)
    compare(a['o_proj_input'],a['pv_head'].to(torch.bfloat16).transpose(1,2).reshape(1,1,4096),'RS actual PV cast/head reshape',True)
    # Verify offline diagnostics independently; all are explicitly unused in forward.
    references={'native_full':(qs['full_dq']@ks['full_dq'].transpose(-1,-2))/math.sqrt(128),'original_source':(a['q_dq']@a['k_dq'].transpose(-1,-2))/math.sqrt(128),'original_fp':(a['q_post_rope'].float()@a['k_visible'].float().transpose(-1,-2))/math.sqrt(128),'actual_predictor':a['estimator_logits']}
    for label,reference in references.items():
        compare(d[label+'_logits'],reference,'offline '+label+' logits',True);compare(d[label+'_probs'],reference.softmax(-1),'offline '+label+' probabilities',True)
        for suffix,p in THRESHOLDS.items():
            expected=selection(reference,a['allowed_mask'],p)[0];check_set(d,expected,positions,label+'_diagnostic_p'+suffix+'_')
    for label,reference in [('native',references['native_full']),('source',references['original_source'])]:
        expected=reference.masked_fill(~a['selected_mask'],-torch.inf);compare(d[label+'_final_on_actual_set'],expected,'same-input final-format '+label,True);compare(d[label+'_probs_on_actual_set'],expected.softmax(-1),'same-input final probabilities '+label,True)
    fp64,boundaries=semantic_fp64(a['estimator_logits'],a['allowed_mask'],s['selection']['p']);differences=(fp64!=a['selected_mask']).any(-1)
    model_report=model.replay(meta,a) if model is not None else dict(status='not_executed')
    return dict(record_id=entry['record_id'],passed=True,source_and_bytes_exact=True,final_origin_verified=True,selection_passes=s['selection_passes'],final_execution=s['final_execution'],errors=errors,
        fp64_semantic_disagreement_heads=differences.nonzero().tolist(),selector_boundaries=boundaries,diagnostics_unused_in_forward=True,model_row_replay=model_report)

def replay_run(run_dir,device='cuda',model_weights=True):
    # This public entry is used by a fresh replay subprocess. Pin cuBLAS and
    # deterministic arithmetic before environment() or model/tensor CUDA work.
    configure_runtime()
    runtime=environment()
    check(all(runtime.get(key)==value for key,value in dict(cublas_workspace_config=':4096:8',tf32_matmul=False,tf32_cudnn=False,float32_matmul_precision='highest',deterministic_algorithms=True).items()),'RS replay pinned runtime differs')
    root=Path(run_dir).resolve();check(load_json(root/'status.json')['status']=='complete','RS capture incomplete');resolved=load_json(root/'resolved_config.json');config=resolved['algorithm'];validate_formal_config(config)
    m=load_json(root/'metrics.json');assets=load_json(root/'asset_manifest.json');code=load_json(root/'code_manifest.json')
    recorded_runtime=load_json(root/'environment.json')
    check(recorded_runtime.get('environment_sha256')==canonical_hash({key:value for key,value in recorded_runtime.items() if key!='environment_sha256'}),'RS captured environment manifest hash')
    check(runtime['environment_sha256']==m['environment_sha256']==recorded_runtime['environment_sha256'],'RS replay runtime/environment differs from capture')
    check(m['semantic_config_sha256']==canonical_hash(config) and m['asset_manifest_sha256']==canonical_hash(assets),'RS config/asset hash')
    check(m['code_sha256']==code['code_sha256']==canonical_hash(dict(head=code['head'],files=code['files'])) and code['patch_sha256']==sha256_file(root/'code.patch'),'RS code/patch binding')
    check(m['capture_config_sha256']==TARGET_HASH,'RS target file binding');model=None
    if model_weights:
        from .model_replay import ModelRowReplay
        model=ModelRowReplay(resolved['model_path'],device)
    reports=[]
    for entry,meta,a,d in load_trace_records(root):
        for key in ('code_sha256','tokens_sha256','environment_sha256','asset_manifest_sha256','subset_sha256','scoring_ranges_sha256'):check(meta[key]==m[key],'RS trace/run '+key)
        check(meta['model_file_hashes']==assets['model_file_hashes'] and meta['config_sha256']==m['semantic_config_sha256'],'RS trace model/config hashes')
        reports.append(replay_record(entry,meta,a,d,config,device,model))
    return dict(schema_version='attention-rs-replay-v1',status='complete',passed=True,num_records=6,run_dir=str(root),records=reports,atol=ATOL,rtol=RTOL,byte_decode_exact=True,model_weights_replayed=model_weights,device=str(device),runtime_initialized=True,environment_sha256=runtime['environment_sha256'],index_sha256=sha256_file(root/'trace/index.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    try:report=replay_run(a.run_dir,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc),atol=ATOL,rtol=RTOL));raise
    write_json(a.output,report)
if __name__=='__main__':main()
