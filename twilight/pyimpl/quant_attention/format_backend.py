"""One format predictor selection followed by original full INT8 recomputation."""
import math
import torch
from .predictor_formats import operands
from .slice_backend import exact_top_p, selected_full_logits


def validate_format_config(config, method):
    from .slice_backend import validate_slice_config
    if set(config) != {'variant_id','baseline_name','estimator','encoding','q_prefix_bits','k_prefix_bits','selection','final_scores','final_execution','scope','diagnostics','predictor_format','format_parameters'}:
        raise ValueError('Unknown/missing format fields')
    fmt = config['predictor_format']
    if fmt == 'rsint8':
        raise ValueError('RS blocked: primary encoding definition unavailable')
    if fmt not in ('sm8','mxint8_tc4','mxfp4','mxint8q_mxfp4k'):
        raise ValueError('Unknown predictor format')
    expected = {} if fmt == 'sm8' else {'block_size':32}
    if fmt in ('mxint8_tc4','mxint8q_mxfp4k'):
        expected['endpoint_policy'] = 'symmetric_omit_min_negative'
    if config['format_parameters'] != expected or config['encoding'] != ('sign_magnitude' if fmt == 'sm8' else 'microscaling'):
        raise ValueError('Unsupported format parameters/encoding')
    if (config['q_prefix_bits'],config['k_prefix_bits']) not in ((4,4),(8,4)):
        raise ValueError('Only Q4K4 and Q8K4 supported')
    if (fmt == 'mxfp4' and config['q_prefix_bits'] != 4) or (fmt == 'mxint8q_mxfp4k' and config['q_prefix_bits'] != 8):
        raise ValueError('Invalid conditional native/mixed FP4 width')
    base = {k:v for k,v in config.items() if k not in ('predictor_format','format_parameters')}
    base.update(estimator='tc_msb',encoding='twos_complement')
    validate_slice_config(base,method)


def attention_format(q,k,v,allowed,qq,kk,config,query_chunk=64,observer=None,stats=None):
    if q.ndim != 4 or k.shape != v.shape or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Invalid format attention layout')
    if qq is None or kk is None or query_chunk < 1:
        raise ValueError('Format predictor requires original quantization')
    if torch.is_autocast_enabled() or (q.is_cuda and torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Format predictor requires autocast and TF32 disabled')
    with torch.autocast(device_type=q.device.type, enabled=False):
        qs,ks = operands(qq,kk,config)
        result = torch.empty_like(q,dtype=torch.float32)
        d,n = q.shape[-1],k.shape[-2]
        denominator = torch.tensor(math.sqrt(d),device=q.device,dtype=torch.float32)
        for start in range(0,q.shape[-2],query_chunk):
            end = min(start+query_chunk,q.shape[-2]); domain=allowed[...,start:end,:]
            if config['predictor_format'] == 'sm8':
                dot=torch.matmul(qs['hi'][...,start:end,:].float(),ks['hi'].float().transpose(-1,-2))
                if d*127*127 >= 2**24 or not torch.equal(dot,dot.round()):
                    raise ValueError('SM prefix dot outside exact FP32 integer domain')
                aq,ak=1 << (8-config['q_prefix_bits']),1 << (8-config['k_prefix_bits'])
                logits=(((dot*(aq*ak))*qq.scales[...,start:end,None].float())*kk.scales[...,None,:].float())/math.sqrt(d)
                dot_hh=dot.to(torch.int32)
            else:
                dot=torch.matmul(qs['predictor_dq'][...,start:end,:],ks['predictor_dq'].transpose(-1,-2))
                logits=torch.div(dot,denominator); dot_hh=None
            logits=logits.masked_fill(~domain,-torch.inf)
            mask,indices,counts,valid,ep=exact_top_p(logits,domain,config['selection']['p'])
            selected,padded=selected_full_logits(qq,kk,start,end,indices,valid)
            final=torch.full_like(logits,-torch.inf); flat=final.reshape(-1,n)
            ix=indices.reshape(-1,indices.shape[-1]); vi=valid.reshape_as(ix)
            rr=torch.arange(flat.shape[0],device=q.device)[:,None].expand_as(ix)
            flat[rr[vi],ix[vi]]=selected.reshape_as(ix)[vi]
            probs=torch.softmax(final,-1,dtype=torch.float32); pv=torch.matmul(probs,v.float())
            result[...,start:end,:]=pv
            if stats is not None:
                stats.record_slice_statistics(rows=counts.numel(),selected_count=counts.sum(),selected_count_squared=(counts*counts).sum(),selected_count_min=counts.min(),selected_count_max=counts.max(),allowed_count=domain.expand_as(mask).sum(),estimator_qk_macs=dot.numel()*d,selected_full_qk_macs=counts.sum()*d,actual_selected_full_qk_macs=padded*d,actual_pv_macs=probs.numel()*d,selection_passes=1,actual_pv='dense_zero_masked',selected_row_chunk=8,heads_parallel=q.shape[1])
            if observer is not None:
                payload=dict(estimator_logits=logits,estimator_probs=ep,selected_mask=mask,selected_indices=indices,selected_counts=counts,selected_valid_mask=valid,final_selected_logits=final,final_selected_probs=probs,dot_hh=dot_hh,q_hi=qs['hi'][...,start:end,:],k_hi=ks['hi'],q_lo=None,k_lo=None,pv_head=pv,sqrt_D_f32=denominator)
                for prefix,side in (('q',qs),('k',ks)):
                    for field,value in side.items():
                        payload[prefix+'_format_'+field]=value[...,start:end,:] if prefix=='q' else value
                observer(start,end,payload)
        return result
