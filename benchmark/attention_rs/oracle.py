"""Independent RS arithmetic and selection: no production helper imports."""
import math
import torch
ATOL=1e-5
RTOL=1e-4

def check(condition,message):
    if not bool(condition):raise ValueError(message)

def compare(actual,expected,label,exact=False):
    actual=actual.cpu();expected=expected.cpu();check(actual.shape==expected.shape and actual.dtype==expected.dtype,label+' shape/dtype mismatch')
    if exact:check(torch.equal(actual,expected),label+' exact mismatch');return dict(max_abs=0.,coordinate=None)
    check(torch.equal(torch.isinf(actual),torch.isinf(expected)) and torch.equal(torch.isnan(actual),torch.isnan(expected)),label+' nonfinite mismatch')
    finite=torch.isfinite(expected);error=torch.where(finite,(actual-expected).abs(),torch.zeros_like(expected));worst=int(error.reshape(-1).argmax()) if error.numel() else 0
    coordinate=list(__import__('numpy').unravel_index(worst,error.shape)) if error.numel() else []
    check(torch.allclose(actual,expected,atol=ATOL,rtol=RTOL),label+' numerical mismatch max='+str(float(error.max()))+' coordinate='+str(coordinate))
    return dict(max_abs=float(error.max()) if error.numel() else 0.,coordinate=[int(x) for x in coordinate])

def source_quantization(original):
    check(original.ndim==4 and original.is_floating_point() and torch.isfinite(original).all(),'invalid independent source input')
    x=original.to(torch.float16,copy=True);check(torch.isfinite(x).all(),'source FP16 overflow')
    maximum=x.abs().amax(-1);raw=maximum/127;scale=torch.where(maximum==0,torch.ones_like(raw),raw)
    check(torch.isfinite(scale).all() and (scale>0).all(),'source nonzero scale underflow')
    value=torch.round(x/scale[...,None]);check(torch.isfinite(value).all() and (value.abs()<=127).all(),'source code overflow')
    return value.to(torch.int8),scale

def scalar_native(r):
    check(type(r) is int and -136<=r<=119,'native domain')
    high=(r+8)//16;low=r-16*high
    return ((high%16)<<4)|(low%16),high,low

def decode_words(words):
    check(words.dtype==torch.uint8,'stored RS words must be uint8')
    values=[]
    for word in words.reshape(-1).tolist():
        high,low=divmod(word,16)
        if high>=8:high-=16
        if low>=8:low-=16
        values.append(high*16+low)
    return torch.tensor(values,dtype=torch.int16).reshape(words.shape)

def source_sidecar(codes,scales,bits):
    check(codes.dtype==torch.int8 and codes.ndim==4 and (codes>=-127).all(),'source code contract')
    check(scales.dtype==torch.float16 and scales.shape==codes.shape[:-1] and torch.isfinite(scales).all() and (scales>0).all(),'source scale contract')
    check(bits in (4,8),'oracle width');rows={name:[] for name in ('raw_input_bytes','rs_words','H','L','retained_bytes','native_int','predictor_int','saturation_mask','conversion_residual','prefix_residual','total_prefix_residual')}
    for source in codes.reshape(-1).tolist():
        native=min(source,119);word,high,low=scalar_native(native);prefix=high*16
        values=dict(raw_input_bytes=native%256,rs_words=word,H=high,L=low,retained_bytes=(word>>4) if bits==4 else word,native_int=native,predictor_int=prefix if bits==4 else native,saturation_mask=source!=native,conversion_residual=source-native,prefix_residual=low,total_prefix_residual=source-prefix)
        for name,value in values.items():rows[name].append(value)
    byte={'raw_input_bytes','rs_words','retained_bytes'}
    result={name:torch.tensor(value,dtype=torch.uint8 if name in byte else (torch.bool if name=='saturation_mask' else torch.int16)).reshape(codes.shape) for name,value in rows.items()}
    # Decode the stored bytes independently, so corrupted/bypassed packed data cannot pass.
    result['native_int']=decode_words(result['rs_words']);result['scales']=scales
    result['full_dq']=result['native_int'].float()*scales.float()[...,None];result['predictor_dq']=result['predictor_int'].float()*scales.float()[...,None]
    return result

def selection(logits,allowed,p,positions=None):
    domain=allowed.expand_as(logits);check(domain.dtype==torch.bool and domain.any(-1).all(),'oracle legal domain')
    check(torch.isfinite(logits[domain]).all(),'oracle legal scores nonfinite')
    scores=logits.float().masked_fill(~domain,-torch.inf);probs=scores.softmax(-1)
    if p is None:mask=domain.clone()
    else:
        check(type(p) is float and 0<p<=1,'oracle p')
        ranked=torch.argsort(scores,dim=-1,descending=True,stable=True);sums=torch.cumsum(torch.gather(probs,-1,ranked),-1)
        preceding=torch.cat((torch.zeros_like(sums[...,:1]),sums[...,:-1]),-1)
        keep=(preceding<p)&torch.gather(domain,-1,ranked);mask=torch.zeros_like(domain).scatter(-1,ranked,keep)
    counts=mask.sum(-1);width=int(counts.max());n=logits.shape[-1]
    ix=torch.arange(n,device=logits.device).expand_as(mask).masked_fill(~mask,n).sort(-1).values[...,:width]
    valid=torch.arange(width,device=logits.device)<counts[...,None];ix=ix.masked_fill(~valid,-1)
    if positions is not None:ix=positions.to(ix.device)[ix.clamp_min(0)].masked_fill(~valid,-1)
    return mask,ix,counts,valid,probs

def semantic_fp64(logits,allowed,p):
    """Scalar float64 oracle records every ranked boundary for diagnosis."""
    masks=[];boundaries=[]
    for row,legal in zip(logits.cpu().reshape(-1,logits.shape[-1]).tolist(),allowed.expand_as(logits).cpu().reshape(-1,logits.shape[-1]).tolist()):
        order=sorted((i for i,ok in enumerate(legal) if ok),key=lambda i:(-row[i],i));maximum=max(row[i] for i in order);weights=[math.exp(row[i]-maximum) for i in order];total=math.fsum(weights);mass=0.;selected=[];ranked=[]
        for i,weight in zip(order,weights):
            probability=weight/total;before=mass;mass+=probability;ranked.append(dict(key_position=i,score=row[i],probability=probability,cumulative=mass))
            if p is None or before<p:selected.append(i)
        masks.append([i in selected for i in range(len(row))]);count=len(selected);boundaries.append(dict(selected_count=count,before=ranked[max(0,count-2)],crossing=ranked[count-1],after=ranked[count] if count<len(ranked) else None,ranked=ranked))
    return torch.tensor(masks,dtype=torch.bool).reshape(logits.shape),boundaries

def check_set(data,mask,positions,prefix='selected_'):
    compare(data[prefix+'mask'],mask,prefix+'mask',True);counts=mask.sum(-1);compare(data[prefix+'counts'],counts,prefix+'counts',True)
    indices=data[prefix+'indices'];valid=data[prefix+'valid_mask'];check(indices.dtype==torch.int64 and valid.dtype==torch.bool,'set dtype')
    check(torch.equal(valid.sum(-1),counts) and (indices[~valid]==-1).all(),'set padding/count')
    check(torch.equal(((indices[...,None]==positions)&valid[...,None]).any(-2),mask),'set indices/mask')
    for row,v in zip(indices.reshape(-1,indices.shape[-1]),valid.reshape(-1,valid.shape[-1])):check((row[v][1:]>row[v][:-1]).all(),'indices must ascend uniquely')
