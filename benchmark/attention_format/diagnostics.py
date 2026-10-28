"""Actual-source/format diagnostics; quality measurements never authorize execution."""
import torch


def histogram(x):
    values,counts=x.reshape(-1).unique(sorted=True,return_counts=True)
    return {'values':values.tolist(),'counts':counts.tolist()}


def errors(value):
    x=value.double()
    return dict(bias=float(x.mean()),mae=float(x.abs().mean()),mse=float(x.square().mean())) if x.numel() else dict(bias=None,mae=None,mse=None)


def source_format_statistics(actual,config):
    rows=[]
    for operand in ('q','k'):
        codes=actual[operand+'_codes'];original=actual[operand+'_dq'];prefix=actual[operand+'_format_prefix_bytes'];full=actual[operand+'_format_full_dq'];predictor=actual[operand+'_format_predictor_dq']
        allfields={name:value for name,value in actual.items() if name.startswith(operand+'_format_') and value is not None}
        bytes_by_field={name:value.numel()*value.element_size() for name,value in allfields.items()}
        logical_bits=config[operand+'_prefix_bits'];mx=config['predictor_format']!='sm8'
        for head in range(codes.shape[1]):
            source=codes[0,head].reshape(-1);real=original[0,head].reshape(-1).double();pred=predictor[0,head].reshape(-1).double();fq=full[0,head].reshape(-1).double();px=prefix[0,head].reshape(-1)
            if mx:
                metadata=actual[operand+'_format_exponent'][0,head].repeat_interleave(32,-1).reshape(-1).long()
            else:
                metadata=actual[operand+'_scales'][0,head].contiguous().view(torch.int16).long()[...,None].expand_as(codes[0,head]).reshape(-1)
            payload=torch.stack((px.long(),metadata),-1);unique,inverse=payload.unique(dim=0,return_inverse=True)
            residual=real-pred;nunique=len(unique);counts=torch.bincount(inverse,minlength=nunique)
            sums=torch.zeros(nunique,dtype=torch.float64).scatter_add_(0,inverse,residual)
            squares=torch.zeros(nunique,dtype=torch.float64).scatter_add_(0,inverse,residual.square())
            variances=(squares/counts-(sums/counts).square()).clamp_min(0)
            def distinct_per_group(values):
                pairs=torch.stack((inverse,values),-1).unique(dim=0)
                return torch.bincount(pairs[:,0],minlength=nunique)
            distinct_codes=distinct_per_group(source.long());distinct_real=distinct_per_group(real.view(torch.int64))
            conditional=[dict(prefix_byte=int(key[0]),metadata=int(key[1]),count=int(counts[group]),distinct_source_codes=int(distinct_codes[group]),distinct_source_real_values=int(distinct_real[group]),residual_variance_dq=float(variances[group])) for group,key in enumerate(unique)]
            weighted_variance=float((counts*variances).sum()/source.numel());collisions=int((distinct_real-1).clamp_min(0).sum())
            signs={}
            for sign,selection in [('negative',source<0),('zero',source==0),('positive',source>0)]:
                signs[sign]=dict(count=int(selection.sum()),predictor_vs_original=errors((pred-real)[selection]),full_sidecar_vs_original=errors((fq-real)[selection]),predictor_vs_full_sidecar=errors((pred-fq)[selection]))
            row=dict(operand=operand,head_id=head,count=source.numel(),source_histogram=histogram(source),full_format_code_histogram=histogram(actual[operand+'_format_full_codes'][0,head]),prefix_histogram=histogram(px),negative_count=int((source<0).sum()),positive_count=int((source>0).sum()),zero_count=int((source==0).sum()),predictor_zero_count=int((pred==0).sum()),unique_prefix_payloads_including_metadata=len(unique),prefix_collision_excess_real_values=collisions,conditional_residual_variance_dq=weighted_variance,conditional=conditional,sign_errors=signs,logical_element_bits=logical_bits,metadata_bits_per_block=8 if mx else 0,logical_bits_per_element_including_metadata=logical_bits+.25 if mx else logical_bits,full_sidecar_bits_per_element_including_metadata=(4.25 if (config['predictor_format']=='mxfp4' or (config['predictor_format']=='mxint8q_mxfp4k' and operand=='k')) else 8.25) if mx else 8,actual_all_heads_bytes_by_field=bytes_by_field,actual_all_heads_sidecar_tensor_bytes=sum(bytes_by_field.values()),actual_all_heads_original_source_codes_bytes=codes.numel()*codes.element_size(),actual_all_heads_original_source_scales_bytes=actual[operand+'_scales'].numel()*actual[operand+'_scales'].element_size(),logical_retained_packed_bytes=(codes.numel()*logical_bits+7)//8+(codes.numel()//32 if mx else 0),logical_full_sidecar_packed_bytes=((codes.numel()*(4 if (config['predictor_format']=='mxfp4' or (config['predictor_format']=='mxint8q_mxfp4k' and operand=='k')) else 8)+7)//8)+(codes.numel()//32 if mx else 0),actual_storage_note='Saved field payload bytes are summed; aliases, capture copies and allocator peak are not inferred. Prefix bytes and hi tensors are unpacked. Original final source codes/scales are additional storage.')
            if mx:
                for field in ('clamp_mask','zero_blocks','underflow_mask'):row[field+'_count']=int(actual[operand+'_format_'+field][0,head].sum())
                row['exponent_histogram']=histogram(actual[operand+'_format_exponent'][0,head])
            rows.append(row)
    return rows


def logit_statistics(actual,reference):
    result=[];allowed=actual['allowed_mask'].expand_as(actual['estimator_logits'])
    for head in range(actual['estimator_logits'].shape[1]):
        legal=allowed[0,head,0];pred=actual['estimator_logits'][0,head,0,legal].double();ref=reference[0,head,0,legal].double()
        rp=torch.argsort(torch.argsort(pred,stable=True),stable=True).double();rr=torch.argsort(torch.argsort(ref,stable=True),stable=True).double();rp-=rp.mean();rr-=rr.mean()
        error=pred-ref
        result.append(dict(head_id=head,legal_keys=pred.numel(),logit_error=errors(error),centered_rmse=float((error-error.mean()).square().mean().sqrt()),stable_rank_correlation=float((rp*rr).sum()/(rp.norm()*rr.norm()).clamp_min(1e-30)),predictor_tie_fraction=1-pred.unique().numel()/pred.numel(),reference_tie_fraction=1-ref.unique().numel()/ref.numel(),selected_count=int(actual['selected_mask'][0,head,0].sum())))
    return result


def decomposition_statistics(predicted_logits,predicted_mask,reference_logits,reference_mask,reference_probs,allowed,p):
    """Same-input diagnostics; caller declares each CPU/device distribution origin."""
    rows=logit_statistics(dict(estimator_logits=predicted_logits,selected_mask=predicted_mask,allowed_mask=allowed),reference_logits)
    for row in rows:
        head=row['head_id'];pred=predicted_mask[0,head,0];ref=reference_mask[0,head,0];prob=reference_probs[0,head,0]
        intersection=int((pred&ref).sum());union=int((pred|ref).sum());npred=int(pred.sum());nref=int(ref.sum());mass=float(prob[pred].double().sum())
        row.update(num_predicted=npred,num_reference=nref,recall=intersection/nref,precision=intersection/npred,jaccard=intersection/union,retained_reference_mass=mass,mass_shortfall=max(0.,p-mass),reference_retained_mass=float(prob[ref].double().sum()))
    return rows
