"""Independent FP64 mathematics and pinned FP32 execution replay of real traces."""
import argparse
import math
from pathlib import Path
import torch
from benchmark.attention_baseline.common import canonical_hash,load_json,sha256_file,write_json,semantic_config
from benchmark.attention_slice.replay import check_set_fields
from .formal import validate_formal_config
from .diagnostics import source_format_statistics,logit_statistics,decomposition_statistics
from .oracle import check,compare,source_quantization,sidecars,predictor_logits,top_p,mathematical_top_p,ATOL,RTOL


def load_trace_records(run_dir,verify=True):
    root=Path(run_dir).resolve();trace=root/'trace';index=load_json(trace/'index.json')
    check(index.get('schema_version')=='attention-format-trace-v1','format trace schema mismatch')
    check(len(index['records'])==6,'six trace records required');result=[];seen=set()
    for entry in index['records']:
        rid=entry['record_id'];check(rid not in seen,'duplicate record');seen.add(rid)
        path=(trace/entry.get('path',rid)).resolve();check(path.is_relative_to(trace),'unsafe trace path')
        meta=load_json(path/'meta.json')
        for name,key in [('actual.pt','actual_sha256'),('derived.pt','derived_sha256'),('meta.json','meta_sha256')]:
            if verify:check(entry[key]==sha256_file(path/name),'trace index digest '+name)
            if verify and name!='meta.json':check(meta['files'][name]==entry[key],'trace metadata digest')
        result.append((entry,meta,torch.load(path/'actual.pt',map_location='cpu',weights_only=True),torch.load(path/'derived.pt',map_location='cpu',weights_only=True)))
    targets={(l,q,s) for l in (0,16,31) for q,s in ((127,'prefill'),(128,'decode'))}
    check({(m['layer_id'],m['query_position'],m['stage']) for _,m,_,_ in result}==targets,'target coverage mismatch')
    return result


def validate_mx_reference(qside,kside,actual,derived,config):
    """Validate the saved CPU full-sidecar reference independently of production.

    This diagnostic reference never replaces the actual-logit/device execution
    selector. Conditional FP4 paths use their native/mixed full-sidecar meaning.
    """
    if config['predictor_format']=='sm8':return None
    allowed=actual['allowed_mask'];positions=actual['key_positions'];p=config['selection']['p']
    q=qside['full_dq'].float().contiguous();k=kside['full_dq'].float().contiguous()
    logits=torch.div(q@k.transpose(-1,-2),torch.tensor(math.sqrt(q.shape[-1]),dtype=torch.float32))
    logits=logits.masked_fill(~allowed,-torch.inf)
    mask,probs,_=top_p(logits,allowed,positions,p)
    for field,expected in [('reference_mx_full_logits_same_input',logits),('reference_mx_full_probs_same_input',probs)]:
        check(field in derived,'missing saved CPU full-sidecar reference: '+field);compare(derived[field],expected,field,True)
    check_set_fields(derived,mask,positions,'reference_mx_full_top_p_','_same_input')
    original_logits=derived['reference_int_logits_same_input'];original_probs=derived['reference_int_probs_same_input'];original_mask=derived['reference_int_top_p_mask_same_input']
    return dict(reference_kind='MXINT8 full8 sidecar' if config['predictor_format']=='mxint8_tc4' else 'native/mixed full sidecar',reference_scope='same-input CPU FP32 diagnostic independently decoded from original source; never production attention or exact device selector',saved_full_sidecar_reference_validated=True,full_sidecar_vs_original_int=decomposition_statistics(logits,mask,original_logits,original_mask,original_probs,allowed,p),actual_predictor_vs_full_sidecar=decomposition_statistics(actual['estimator_logits'],actual['selected_mask'],logits,mask,probs,allowed,p))


def replay_record(entry,meta,a,derived,device='cuda',failure_dir=None):
    required=('layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','allowed_mask','key_positions','selected_mask','selected_indices','selected_counts','valid_mask','estimator_logits','estimator_probs','final_selected_logits','final_selected_probs','pv_head','sqrt_D_f32')
    check(all(name in a and a[name] is not None for name in required),'missing actual trace capture')
    rid=entry['record_id']; config=meta['attention_slice'];validate_formal_config(meta.get('algorithm',{'attention_slice':config})) if 'algorithm' in meta else None
    check((meta['Hq'],meta['Hkv'],meta['D'])==(32,8,128),'architecture mismatch')
    check(meta['sample_id']==0 and meta['window_id'] is None and meta['predicted_target']==meta['query_position']+1,'sample target mismatch')
    check(meta['backend']=='torch_qdq' and meta['cache_storage']=='original_float' and meta['quantization_at']=='attention_read','execution provenance')
    check(meta['model_dtype']=='torch.bfloat16' and meta['cache_length']==meta['expected_cache_length']==meta['query_position']+1,'model/cache mismatch')
    check(meta['source_revision']=='7f7bfb00bcd26b2cef49e7783f51ef610e05abf7' and meta['input_cast_policy']=='native_fp16_allow_underflow_to_zero','source provenance')
    check(meta['zero_point']==0 and meta['statistics_axis']=='D' and meta['scale_broadcast']=='[B,H,L,1]' and meta['rounding']=='torch_round_ties_to_even','source convention')
    check(meta.get('tf32_allowed') is False and meta.get('float32_matmul_precision')=='highest' and meta.get('deterministic_algorithms') is True,'unpinned FP32 execution')
    check(meta['head_to_kv']==[h//4 for h in range(32)],'GQA mismatch')
    check(meta['selection_passes']==1 and meta['scope']=='prefill_and_decode','scope/pass mismatch')
    check(meta['final_operand_origin']=='original_intflash_codes','final source mismatch')
    check(meta.get('source_quantization_unchanged') is True,'source quantizer preservation not captured')
    check(meta.get('predictor_requantized')==(config['predictor_format']!='sm8'),'requantization disclosure')
    check(all(a[name].dtype==torch.bfloat16 for name in ('q_post_rope','k_visible','v_visible')),'original operand dtype changed')
    compare(a['sqrt_D_f32'],torch.tensor(math.sqrt(128),dtype=torch.float32),'sqrt D bits',True)
    n=meta['query_position']+1;positions=a['key_positions'];allowed=a['allowed_mask'];p=config['selection']['p']
    check(torch.equal(positions,torch.arange(n)),'key domain mismatch')
    check(a['q_post_rope'].shape==(1,32,1,128) and a['k_visible'].shape==(1,32,n,128),'QK shape mismatch')
    for inventory,fields in ((a,meta['fields']),(derived,meta.get('derived_fields',{}))):
        for name,value in inventory.items():
            if value is None:continue
            info=fields.get(name);check(info and info.get('layout') and info.get('capture_point'),name+': missing capture metadata')
            check(info['shape']==list(value.shape) and info['dtype'].removeprefix('torch.')==str(value.dtype).removeprefix('torch.'),name+': metadata mismatch')
    errors={}
    for name,original in [('q',a['q_post_rope']),('k',a['k_visible'])]:
        # Quantizer operations execute on the pinned device, independently of production.
        codes,scales=source_quantization(original.to(device));compare(a[name+'_codes'],codes.cpu(),name+' source',True);compare(a[name+'_scales'],scales.cpu(),name+' scales',True)
    qc,kc,qs,ks=(a[k] for k in ('q_codes','k_codes','q_scales','k_scales'))
    for prefix,codes,scales in [('q',qc,qs),('k',kc,ks)]:compare(a[prefix+'_dq'],codes.float()*scales.float()[...,None],prefix+' original DQ bits',True)
    qside,kside=sidecars(qc,kc,qs,ks,config)
    for prefix,side in [('q',qside),('k',kside)]:
        for name,value in side.items():
            key=prefix+'_format_'+name;check(key in a,'missing actual sidecar '+key);compare(a[key],value,key,True)
    actual=a['estimator_logits'].to(device).contiguous(); check(actual.dtype==torch.float32,'actual logits not binary32')
    savedmask=a['selected_mask'];actual_mask,actual_probs,selection=top_p(actual,allowed,positions,p)
    check_set_fields(a,actual_mask.cpu(),positions)
    compare(a['estimator_probs'],actual_probs.cpu(),'actual-logit softmax')
    regen=predictor_logits(qside,kside,qs,ks,config,device).masked_fill(~allowed.to(device),-torch.inf)
    repeat=predictor_logits(qside,kside,qs,ks,config,device).masked_fill(~allowed.to(device),-torch.inf)
    compare(regen,repeat,'deterministic FP32 replay',True)
    regenerated_mask,regenerated_probs,regenerated_selection=top_p(regen,allowed,positions,p)
    if not torch.equal(regenerated_mask.cpu(),savedmask):
        if failure_dir:
            path=Path(failure_dir);path.mkdir(parents=True,exist_ok=True)
            divergence=[]
            actual_order=selection['order'].reshape(-1,n);regen_order=regenerated_selection['order'].reshape(-1,n)
            am=actual_mask.reshape(-1,n);rm=regenerated_mask.reshape(-1,n)
            for row in range(am.shape[0]):
                different=(am[row]!=rm[row]).nonzero().reshape(-1)
                if not different.numel():continue
                rank_difference=(actual_order[row]!=regen_order[row]).nonzero().reshape(-1)
                na,nr=int(am[row].sum()),int(rm[row].sum())
                divergence.append(dict(row=row,first_divergent_key_position=int(positions[int(different[0])]),first_divergent_sorted_rank=int(rank_difference[0]) if rank_difference.numel() else None,actual_crossing_key=int(positions[int(actual_order[row,na-1])]),regenerated_crossing_key=int(positions[int(regen_order[row,nr-1])]),actual_selected_positions=positions[am[row].cpu()].tolist(),regenerated_selected_positions=positions[rm[row].cpu()].tolist()))
            actual_selection={k:v.cpu() for k,v in selection.items()};regenerated_selection_saved={k:v.cpu() for k,v in regenerated_selection.items()}
            torch.save(dict(actual_logits=actual.cpu(),regenerated_logits=regen.cpu(),actual_bits=actual.view(torch.int32).cpu(),regenerated_bits=regen.view(torch.int32).cpu(),actual_mask=savedmask,regenerated_mask=regenerated_mask.cpu(),actual_selection=actual_selection,regenerated_selection=regenerated_selection_saved,actual_selection_binary32={k:v.view(torch.int32) for k,v in actual_selection.items() if v.dtype==torch.float32},regenerated_selection_binary32={k:v.view(torch.int32) for k,v in regenerated_selection_saved.items() if v.dtype==torch.float32},p=p,p_binary32=torch.tensor(p,dtype=torch.float32).view(torch.int32),divergence=divergence),path/(rid+'-membership-failure.pt'))
        raise ValueError(rid+': regenerated FP32 membership mismatch; unresolved arithmetic')
    check(allowed.dtype==torch.bool and allowed.shape==(1,1,1,n) and allowed.all(),'allowed domain mismatch')
    errors['predictor_logits']=compare(a['estimator_logits'],regen.cpu(),'regenerated logits')
    math_logits=predictor_logits(qside,kside,qs,ks,config,'cpu',True)
    ideal_mask,ideal_probs=mathematical_top_p(math_logits,allowed,positions,p)
    ideal_different_rows=int((ideal_mask!=savedmask).any(-1).sum())
    ideal_artifact=None
    if ideal_different_rows and failure_dir:
        path=Path(failure_dir);path.mkdir(parents=True,exist_ok=True);artifact=path/(rid+'-fp64-semantic-difference.pt')
        ideal_scores=math_logits.masked_fill(~allowed,-torch.inf);ideal_order=torch.argsort(ideal_scores,descending=True,stable=True);ideal_cumsum=torch.gather(ideal_probs,-1,ideal_order).cumsum(-1)
        differences=[]
        for row,(am,im,ao,io) in enumerate(zip(savedmask.reshape(-1,n),ideal_mask.reshape(-1,n),selection['order'].cpu().reshape(-1,n),ideal_order.reshape(-1,n))):
            changed=(am!=im).nonzero().reshape(-1)
            if not changed.numel():continue
            rank=(ao!=io).nonzero().reshape(-1)
            differences.append(dict(row=row,first_divergent_key_position=int(positions[int(changed[0])]),first_divergent_sorted_rank=int(rank[0]) if rank.numel() else None,actual_crossing_key=int(positions[int(ao[int(am.sum())-1])]),ideal_crossing_key=int(positions[int(io[int(im.sum())-1])]),actual_selected_positions=positions[am].tolist(),ideal_selected_positions=positions[im].tolist()))
        torch.save(dict(actual_logits=actual.cpu(),ideal_logits=ideal_scores,actual_logits_binary32=actual.view(torch.int32).cpu(),ideal_logits_binary64=ideal_scores.view(torch.int64),actual_probabilities=actual_probs.cpu(),ideal_probabilities=ideal_probs,actual_probabilities_binary32=actual_probs.view(torch.int32).cpu(),ideal_probabilities_binary64=ideal_probs.view(torch.int64),actual_order=selection['order'].cpu(),ideal_order=ideal_order,actual_cumsum=selection['cumsum'].cpu(),ideal_cumsum=ideal_cumsum,actual_cumsum_binary32=selection['cumsum'].view(torch.int32).cpu(),ideal_cumsum_binary64=ideal_cumsum.view(torch.int64),p=p,actual_mask=savedmask,ideal_mask=ideal_mask,divergence=differences,scope='diagnostic semantic difference; exact execution membership remains independently mandatory'),artifact)
        ideal_artifact=dict(path=str(artifact.resolve()),sha256=sha256_file(artifact))

    finite=torch.isfinite(a['estimator_logits']); math_error=float((a['estimator_logits'][finite].double()-math_logits[finite]).abs().max())
    qfull=(qc.float()*qs.float()[...,None]).to(device);kfull=(kc.float()*ks.float()[...,None]).to(device)
    full=qfull@kfull.transpose(-1,-2)/math.sqrt(128)
    reference_mask,reference_probs,_=top_p(full,allowed,positions,p)
    for field,expected in [('reference_int_logits_same_input',full),('reference_int_probs_same_input',reference_probs)]:
        check(field in derived,'missing same-input original INT reference');errors[field]=compare(derived[field],expected.cpu(),field)
    check_set_fields(derived,reference_mask.cpu(),positions,'reference_int_top_p_','_same_input')
    mx_decomposition=validate_mx_reference(qside,kside,a,derived,config)
    # Match the pinned selected-full executor: 8 rows/chunk, batched head matmul,
    # sorted selected positions, padded with a real key then zero-masked.
    selected=torch.full_like(full,-torch.inf)
    indices=a['selected_indices'].to(device);valid=a.get('valid_mask',a.get('selected_valid_mask')).to(device)
    width=indices.shape[-1];batch,heads,rows,_=indices.shape
    # Independent gather/bmm implementation, with the declared head parallelism.
    for start in range(0,rows,8):
        end=min(start+8,rows);ix=indices[...,start:end,:];vi=valid[...,start:end,:]
        expanded=kfull.unsqueeze(-3).expand(batch,heads,end-start,kfull.shape[-2],128)
        keys=torch.gather(expanded,-2,torch.where(vi,ix,ix[...,:1])[...,None].expand(*ix.shape,128)).contiguous()
        vals=torch.bmm(qfull[...,start:end,:].reshape(-1,1,128),keys.reshape(-1,width,128).transpose(-1,-2)).reshape(batch,heads,end-start,width)/math.sqrt(128)
        dest=selected[...,start:end,:];rr=torch.arange(dest.numel()//n,device=device)[:,None].expand(-1,width)
        dest.reshape(-1,n)[rr[vi.reshape(-1,width)],ix.reshape(-1,width)[vi.reshape(-1,width)]]=vals.reshape(-1,width)[vi.reshape(-1,width)]
    probs=torch.softmax(selected,-1,dtype=torch.float32);pv=probs@a['v_visible'].to(device).float()
    for name,expected in [('final_selected_logits',selected),('final_selected_probs',probs),('pv_head',pv)]:errors[name]=compare(a[name],expected.cpu(),name)
    check((a['final_selected_probs'][~savedmask]==0).all(),'final mass outside S')
    compare(a['o_proj_input'],a['pv_head'].transpose(1,2).reshape(1,1,-1).to(a['o_proj_input'].dtype),'PV to projection',True)
    sorted_actual=torch.gather(actual,-1,selection['order']);gaps=sorted_actual[...,:-1]-sorted_actual[...,1:]
    cumulative_distance=(selection['cumsum']-p).abs()
    return dict(record_id=rid,passed=True,mx_same_input_decomposition=mx_decomposition,source_format_statistics=source_format_statistics(a,config),same_input_logit_statistics=logit_statistics(a,full.cpu()),minimum_sorted_logit_gap=float(gaps.min()),exact_logit_ties=int((gaps==0).sum()),minimum_cumsum_distance_to_p=float(cumulative_distance.min()),exact_equality_crossings=int((cumulative_distance==0).sum()),errors=errors,mathematical_fp64_max_abs=math_error,mathematical_fp64_different_membership_rows=ideal_different_rows,mathematical_fp64_difference_artifact=ideal_artifact,actual_logit_membership_exact=True,regenerated_logit_membership_exact=True,regenerated_logits_bitwise_equal=torch.equal(actual,regen),device=str(device))


def replay_run(run_dir,device='cuda',failure_dir=None):
    root=Path(run_dir).resolve();check(load_json(root/'status.json')['status']=='complete','incomplete run')
    config=load_json(root/'resolved_config.json');algorithm=config.get('algorithm',config);validate_formal_config(algorithm)
    metrics=load_json(root/'metrics.json');assets=load_json(root/'asset_manifest.json');code=load_json(root/'code_manifest.json')
    check(code['code_sha256']==metrics['code_sha256']==canonical_hash({'head':code['head'],'files':code['files']}),'code hash mismatch')
    check(code['patch_sha256']==sha256_file(root/'code.patch'),'patch mismatch');check(canonical_hash(assets)==metrics['asset_manifest_sha256'],'assets mismatch')
    reports=[]
    for entry,meta,actual,derived in load_trace_records(root):
        check(meta['variant_id']==metrics['variant_id']==algorithm['attention_slice']['variant_id'],'variant mismatch')
        check(meta['attention_slice']==algorithm['attention_slice'],'config mismatch')
        for key in ('code_sha256','tokens_sha256'):check(meta[key]==metrics[key],'trace identity '+key)
        check(meta['model_file_hashes']==assets['model_file_hashes'],'model mismatch')
        check(meta['config_sha256']==metrics['semantic_config_sha256']==canonical_hash(semantic_config(algorithm)),'semantic hash mismatch')
        reports.append(replay_record(entry,meta,actual,derived,device,failure_dir or root/'replay_failures'))
    return dict(status='complete',passed=True,schema_version='attention-format-replay-v1',run_dir=str(root),num_records=len(reports),records=reports,atol=ATOL,rtol=RTOL,index_sha256=sha256_file(root/'trace/index.json'),execution_device=str(device))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    try:report=replay_run(args.run_dir,args.device,Path(args.output).parent/'failures')
    except Exception as exc:
        write_json(args.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(args.output,report)
if __name__=='__main__':main()
