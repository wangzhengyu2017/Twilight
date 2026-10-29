"""Native dense/full Top-p RS and RS-prefix/original-source selected attention."""
import math
import copy
import torch
from .rs_codec import PROFILE,PARAMETERS,sidecar,allocation_report
from .slice_backend import exact_top_p,selected_full_logits

VARIANTS={'rs_dense_q8k8':('dense_full',8,8,None)}
for _mode,_shape,_q,_k in [('full_topp','q8k8',8,8),('msb_topp','q8k4',8,4),('msb_topp','q4k4',4,4)]:
    for _label,_p in [('095',.95),('090',.9),('085',.85)]:
        VARIANTS['rs_'+('full' if _mode=='full_topp' else 'msb')+'_'+_shape+'_topp_p'+_label]=(_mode,_q,_k,_p)

def make_config(variant_id):
    mode,q,k,p=VARIANTS[variant_id];native=mode!='msb_topp'
    return dict(variant_id=variant_id,mode=mode,predictor_format=PROFILE,q_bits=q,k_bits=k,
        selection=dict(kind='none' if p is None else 'top_p',p=p,tie_break=None if p is None else 'key_position_ascending'),
        selection_passes=int(p is not None),final_operand_origin='rs_full_decoded' if native else 'original_intflash_codes',
        final_scores='rs_full_dq' if native else 'full_int8_qk_dq',
        final_execution={'dense_full':'dense_rs_full','full_topp':'reuse_full_rs_scores','msb_topp':'selected_full_qk_recompute'}[mode],
        scope='prefill_and_decode',diagnostics='targets_only',format_parameters=copy.deepcopy(PARAMETERS))

def validate_rs_config(config,method):
    if not isinstance(config,dict) or config.get('variant_id') not in VARIANTS:raise ValueError('Unknown RS variant')
    if method!='intflash_qk8' or config!=make_config(config['variant_id']):raise ValueError('RS configuration differs from frozen contract')
    for key in ('q_bits','k_bits','selection_passes'):
        if type(config[key]) is not int:raise ValueError('RS integer config type '+key)
    if any(type(value) is not int for value in config['format_parameters']['native_range']):raise ValueError('RS native bound types')
    if type(config['format_parameters']['zero_point']) is not int or type(config['format_parameters']['slice_bits']) is not int:raise ValueError('RS parameter types')
    if config['selection']['p'] is not None and type(config['selection']['p']) is not float:raise ValueError('RS p must be frozen float')

def legal_set(domain):
    counts=domain.sum(-1);width=int(counts.max().item());n=domain.shape[-1]
    positions=torch.arange(n,device=domain.device).expand_as(domain)
    indices=positions.masked_fill(~domain,n).sort(-1).values[...,:width]
    valid=torch.arange(width,device=domain.device)<counts[...,None]
    return domain,indices.masked_fill(~valid,-1),counts,valid

def attention_rs(q,k,v,allowed,qq,kk,config,query_chunk=64,observer=None,stats=None,layer_id=None):
    validate_rs_config(config,'intflash_qk8')
    if q.ndim!=4 or k.shape!=v.shape or q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1] or type(query_chunk) is not int or query_chunk<1:raise ValueError('Invalid RS layout/chunk')
    if qq is None or kk is None or qq.codes.shape!=q.shape or kk.codes.shape!=k.shape:raise ValueError('RS requires actual source quantization once per forward')
    if allowed.dtype!=torch.bool or allowed.shape[-2:]!=(q.shape[-2],k.shape[-2]) or not allowed.any(-1).all():raise ValueError('Nonempty boolean RS causal domain required')
    if torch.is_autocast_enabled(q.device.type) or (q.is_cuda and torch.backends.cuda.matmul.allow_tf32):raise ValueError('RS requires autocast/TF32 disabled')
    with torch.autocast(device_type=q.device.type,enabled=False):
        qs,ks=sidecar(qq,config['q_bits']),sidecar(kk,config['k_bits'])
        if stats is not None and hasattr(stats,'record_rs'):
            for operand,source,side in [('q',qq,qs),('k',kk,ks)]:stats.record_rs(layer_id,operand,source,side)
        if stats is not None:
            nq,nk=qq.codes.numel(),kk.codes.numel();elements=nq+nk
            full_decodes=nq*(1+int(config['q_bits']==8))+nk*(1+int(config['k_bits']==8))
            prefix_decodes=nq*(1+int(config['q_bits']==4))+nk*(1+int(config['k_bits']==4))
            stats.record_slice_statistics(rs_source_clamp_elements=elements,rs_encode_elements=elements,rs_full_decode_elements=full_decodes,rs_prefix_extract_elements=prefix_decodes,rs_prefix_decode_elements=prefix_decodes,rs_saturation_check_elements=elements)
        result=torch.empty_like(q,dtype=torch.float32);d,n=q.shape[-1],k.shape[-2];denominator=torch.tensor(math.sqrt(d),device=q.device,dtype=torch.float32)
        for start in range(0,q.shape[-2],query_chunk):
            end=min(start+query_chunk,q.shape[-2]);domain=allowed[...,start:end,:]
            logits=(qs['predictor_dq'][...,start:end,:]@ks['predictor_dq'].transpose(-1,-2))/denominator
            logits=logits.masked_fill(~domain,-torch.inf);domain=domain.expand_as(logits)
            if config['mode']=='dense_full':
                mask,indices,counts,valid=legal_set(domain);ep=logits.softmax(-1,dtype=torch.float32)
            else:mask,indices,counts,valid,ep=exact_top_p(logits,domain,config['selection']['p'])
            padded=0
            if config['mode']=='msb_topp':
                selected,padded=selected_full_logits(qq,kk,start,end,indices,valid)
                final=torch.full_like(logits,-torch.inf);flat=final.reshape(-1,n);ix=indices.reshape(-1,indices.shape[-1]);vi=valid.reshape_as(ix)
                rr=torch.arange(flat.shape[0],device=q.device)[:,None].expand_as(ix);flat[rr[vi],ix[vi]]=selected.reshape_as(ix)[vi]
            else:final=logits.masked_fill(~mask,-torch.inf)
            probs=final.softmax(-1,dtype=torch.float32);pv=probs@v.float();result[...,start:end,:]=pv
            if stats is not None:
                stats.record_slice_statistics(rows=counts.numel(),selected_count=counts.sum(),selected_count_squared=(counts*counts).sum(),selected_count_min=counts.min(),selected_count_max=counts.max(),allowed_count=domain.sum(),
                    estimator_qk_macs=logits.numel()*d,selected_full_qk_macs=counts.sum()*d if config['mode']=='msb_topp' else 0,actual_selected_full_qk_macs=padded*d,actual_pv_macs=probs.numel()*v.shape[-1],
                    selection_passes=config['selection_passes'],final_execution=config['final_execution'],actual_pv='dense_zero_masked',selected_row_chunk=8,heads_parallel=q.shape[1])
            if observer is not None:
                payload=dict(production_allocations={prefix:allocation_report(source,side,bits) for prefix,source,side,bits in [('q',qq,qs,config['q_bits']),('k',kk,ks,config['k_bits'])]},floating_coexistence_bytes=dict(q_post_rope=q.numel()*q.element_size(),original_k_cache=k.numel()*k.element_size(),original_v_cache=v.numel()*v.element_size()),estimator_logits=logits,estimator_probs=ep,selected_mask=mask,selected_indices=indices,selected_counts=counts,selected_valid_mask=valid,final_selected_logits=final,final_selected_probs=probs,pv_head=pv,sqrt_D_f32=denominator)
                if config['mode']=='msb_topp':
                    # Observational gather of the immutable source operands actually selected by selected_full_logits.
                    safe=torch.where(valid,indices,indices[...,:1]);batch=torch.arange(q.shape[0],device=q.device)[:,None,None,None];head=torch.arange(q.shape[1],device=q.device)[None,:,None,None]
                    payload.update(selected_source_k_codes=kk.codes[batch,head,safe],selected_source_k_scales=kk.scales[batch,head,safe],selected_source_q_codes=qq.codes[...,start:end,:],selected_source_q_scales=qq.scales[...,start:end],selected_source_logits=selected)
                for prefix,side in [('q',qs),('k',ks)]:
                    for name,value in side.items():payload[prefix+'_rs_'+name]=value[...,start:end] if prefix=='q' and name=='scales' else (value[...,start:end,:] if prefix=='q' else value)
                observer(start,end,payload)
        return result
