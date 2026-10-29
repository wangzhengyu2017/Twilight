"""All-key attention using actual byte-decoded format operands; no selector."""
import torch
from .predictor_formats import dense_sidecar
from .torch_backend import attention

VARIANTS={'dense_sm8_q8k8':('sm8',8,8),'dense_mxint8_q8k8':('mxint8',8,8),
          'dense_mxint8q_mxfp4k_q8k4':('mxint8q_mxfp4k',8,4),'dense_mxfp4_q4k4':('mxfp4',4,4)}


def validate_dense_config(config, method):
    if method != 'intflash_qk8': raise ValueError('Dense format requires original INT8 source')
    if config.get('variant_id') not in VARIANTS: raise ValueError('Unknown dense format variant')
    fmt,bq,bk=VARIANTS[config['variant_id']]
    expected=dict(variant_id=config['variant_id'],format=fmt,q_bits=bq,k_bits=bk,
                  selection_mode='dense',selection_passes=0,final_scores='actual_decoded_format_qk',
                  scope='prefill_and_decode',diagnostic_only_p=[.95,.9,.85],
                  format_parameters={} if fmt=='sm8' else dict(block_size=32,endpoint_policy='symmetric_omit_min_negative'))
    if config != expected or type(config['q_bits']) is not int or type(config['k_bits']) is not int or type(config['selection_passes']) is not int:
        raise ValueError('Dense configuration differs from fixed contract')


def operand_formats(config):
    fmt=config['format']
    return ('mxint8','mxfp4') if fmt=='mxint8q_mxfp4k' else (fmt,fmt)


def attention_dense_format(q,k,v,allowed,qq,kk,config,query_chunk=64,observer=None,stats=None):
    validate_dense_config(config,'intflash_qk8')
    if qq is None or kk is None: raise ValueError('Dense formats require original quantized source')
    if allowed.dtype != torch.bool or allowed.shape[-2:] != (q.shape[-2],k.shape[-2]) or not allowed.any(-1).all():
        raise ValueError('Every dense query requires a nonempty legal key domain')
    if torch.is_autocast_enabled() or (q.is_cuda and torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Dense format requires autocast and TF32 disabled')
    with torch.autocast(device_type=q.device.type,enabled=False):
        qs,ks=[dense_sidecar(source,fmt) for source,fmt in zip((qq,kk),operand_formats(config))]
        def observe(start,end,logits,probs,pv):
            domain=allowed[...,start:end,:].expand_as(logits)
            counts=domain.sum(-1)
            if stats is not None:
                stats.record_slice_statistics(rows=counts.numel(),selected_count=counts.sum(),selected_count_squared=(counts*counts).sum(),selected_count_min=counts.min(),selected_count_max=counts.max(),allowed_count=domain.sum(),selection_passes=0,selection_mode='dense',actual_pv='dense',variant_id=config['variant_id'],full_qk_macs=logits.numel()*q.shape[-1],actual_pv_macs=probs.numel()*q.shape[-1])
            if observer is not None:
                # Index enumeration records actual execution; it does not rank or select.
                width=int(counts.max())
                positions=torch.arange(k.shape[-2],device=q.device).expand_as(domain)
                indices=positions.masked_fill(~domain,k.shape[-2]).sort(-1).values[...,:width]
                valid=torch.arange(width,device=q.device)<counts[...,None]
                payload=dict(logits_e=logits,probs=probs,pv_head=pv,execution_mask=domain,
                             execution_indices=indices.masked_fill(~valid,-1),execution_counts=counts,execution_valid_mask=valid)
                for prefix,side in (('q',qs),('k',ks)):
                    for name,value in side.items():payload[prefix+'_format_'+name]=value[...,start:end,:] if prefix=='q' else value
                observer(start,end,payload)
        return attention(qs['full_dq'],ks['full_dq'],v,allowed,query_chunk,observe)
