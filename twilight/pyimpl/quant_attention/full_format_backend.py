"""Full SM8/MXINT8 TopP: one selector, final scores from the same stored bytes."""
import math
import torch
from .predictor_formats import dense_sidecar
from .slice_backend import exact_top_p

THRESHOLDS={'095':.95,'090':.9,'085':.85}
VARIANTS={f'{fmt}_q8k8_topp_p{suffix}':(fmt,p) for fmt in ('sm8','mxint8') for suffix,p in THRESHOLDS.items()}


def validate_full_format_config(config,method):
    if method!='intflash_qk8' or not isinstance(config,dict) or config.get('variant_id') not in VARIANTS:
        raise ValueError('Unsupported full-format TopP source/variant')
    fmt,p=VARIANTS[config['variant_id']]
    expected=dict(variant_id=config['variant_id'],format=fmt,q_bits=8,k_bits=8,selection_mode='top_p',selection_passes=1,
                  selection=dict(kind='top_p',p=p,tie_break='key_position_ascending'),scope='prefill_and_decode',
                  final_operand_profile='decoded_full_format',final_scores='reuse_decoded_full_format_logits',
                  final_execution='reuse_full_scores',format_parameters={} if fmt=='sm8' else dict(block_size=32,endpoint_policy='symmetric_omit_min_negative'))
    if config!=expected or any(type(config[k]) is not int for k in ('q_bits','k_bits','selection_passes')):
        raise ValueError('Full-format TopP configuration differs from approved matrix')


def attention_full_format(q,k,v,allowed,qq,kk,config,query_chunk=64,observer=None,stats=None):
    validate_full_format_config(config,'intflash_qk8')
    if query_chunk<1 or q.ndim!=4 or k.shape!=v.shape or q.shape[:2]!=k.shape[:2] or q.shape[-1]!=k.shape[-1]:
        raise ValueError('Invalid full-format attention shape/chunk')
    if qq is None or kk is None or allowed.dtype!=torch.bool or allowed.shape[-2:]!=(q.shape[-2],k.shape[-2]) or not allowed.any(-1).all():
        raise ValueError('Full-format TopP requires source quantization and nonempty legal domains')
    if torch.is_autocast_enabled() or (q.is_cuda and torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Full-format TopP requires autocast and TF32 disabled')
    with torch.autocast(device_type=q.device.type,enabled=False):
        qs,ks=[dense_sidecar(source,config['format']) for source in (qq,kk)]
        result=torch.empty_like(q,dtype=torch.float32)
        for start in range(0,q.shape[-2],query_chunk):
            end=min(start+query_chunk,q.shape[-2]);domain=allowed[...,start:end,:]
            full=(qs['full_dq'][...,start:end,:]@ks['full_dq'].transpose(-1,-2))/math.sqrt(q.shape[-1])
            full=full.masked_fill(~domain,-torch.inf)
            mask,indices,counts,valid,full_probs=exact_top_p(full,domain,config['selection']['p'])
            # Retain the exact selected full-format logits. No re-encoding, QK
            # recomputation, source-INT substitution, or second selector occurs.
            selected=full.masked_fill(~mask,-torch.inf)
            probs=selected.softmax(-1,dtype=torch.float32);pv=probs@v.float();result[...,start:end,:]=pv
            if stats is not None:
                stats.record_slice_statistics(rows=counts.numel(),selected_count=counts.sum(),selected_count_squared=(counts*counts).sum(),selected_count_min=counts.min(),selected_count_max=counts.max(),allowed_count=domain.expand_as(mask).sum(),selection_passes=1,selection_mode='top_p',actual_pv='dense_zero_masked',variant_id=config['variant_id'],final_operand_profile='decoded_full_format',full_qk_macs=full.numel()*q.shape[-1],actual_pv_macs=probs.numel()*q.shape[-1])
            if observer is not None:
                payload=dict(full_format_logits=full,full_format_probs=full_probs,selected_mask=mask,selected_indices=indices,selected_counts=counts,selected_valid_mask=valid,final_selected_logits=selected,final_selected_probs=probs,pv_head=pv)
                for prefix,side in (('q',qs),('k',ks)):
                    for name,value in side.items():payload[prefix+'_format_'+name]=value[...,start:end,:] if prefix=='q' else value
                observer(start,end,payload)
        return result
