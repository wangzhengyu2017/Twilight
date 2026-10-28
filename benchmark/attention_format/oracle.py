"""Source/format codecs and selectors; no production implementation imports."""
import math
import torch

ATOL, RTOL = 1e-5, 1e-4

def check(ok, message):
    if not ok: raise ValueError(message)

def source_quantization(original):
    x=original.to(torch.float16)
    check(torch.isfinite(x).all(), 'FP16 source overflow')
    maximum=x.abs().amax(-1); scales=torch.where(maximum==0,torch.ones_like(maximum),maximum/127)
    check(torch.isfinite(scales).all() and (scales>0).all(), 'source scale underflow')
    codes=torch.round(x/scales[...,None]); check(torch.isfinite(codes).all() and (codes.abs()<=127).all(),'invalid original source')
    return codes.to(torch.int8), scales

def sm8(codes,bits):
    c=codes.to(torch.int64); check((c.abs()<=127).all(),'SM source range')
    word=(c.abs()+128*(c<0)).to(torch.uint8)
    a=2**(8-bits); hi=torch.sign(c)*(c.abs()//a)
    return {'full_codes':word,'prefix_bytes':(word.to(torch.int64)//a).to(torch.uint8),
        'hi':hi.to(torch.int16),'full_dq':c.float(),'predictor_dq':(hi*a).float()}

def mx_sidecar(source,bits,fp4=False):
    """Scalar Python frexp/RNE codec; FP32 source values converted exactly to FP64."""
    check(source.shape[-1]%32==0,'incomplete MX block')
    x=source.detach().cpu().float(); shape=x.shape; blocks=x.reshape(-1,32)
    codes=[]; prefixes=[]; his=[]; full=[]; pred=[]; exps=[]; clamps=[]; zeros=[]; under=[]
    table=(0.,.5,1.,1.5,2.,3.,4.,6.)
    for block in blocks:
        vals=block.double().tolist(); check(all(math.isfinite(v) for v in vals),'nonfinite MX source')
        maximum=max(abs(v) for v in vals); e=math.frexp(maximum)[1]-1-(2 if fp4 else 0) if maximum else 0
        check(-127<=e<=127,'MX exponent outside finite E8M0'); exps.append(e); zeros.append(maximum==0)
        for v in vals:
            if fp4:
                u=math.ldexp(v,-e); mag=min(range(8),key=lambda j:(abs(abs(u)-table[j]),j%2,j))
                c=mag+(8 if u<0 and mag else 0); h=math.copysign(table[mag],-1 if c>=8 else 1)
                f=math.ldexp(h,e); z=f; prefix=c; clamp=abs(u)>6
            else:
                u=math.ldexp(v,6-e); rounded=round(u); c=max(-127,min(127,rounded)); h=c//16 if bits==4 else c
                prefix=((c&255)>>4) if bits==4 else c&255; f=math.ldexp(c,e-6); z=math.ldexp(h,e-(2 if bits==4 else 6)); clamp=rounded!=c
            codes.append(c);prefixes.append(prefix);his.append(h);full.append(f);pred.append(z);clamps.append(clamp);under.append(v!=0 and f==0)
    blockshape=(*shape[:-1],shape[-1]//32)
    tensor=lambda v,dtype:torch.tensor(v,dtype=dtype).reshape(shape)
    # Rounding to binary32 occurs only at the explicitly declared reconstructed operands.
    full_tensor=tensor(full,torch.float32); pred_tensor=tensor(pred,torch.float32)
    check(torch.isfinite(full_tensor).all() and torch.isfinite(pred_tensor).all(),'nonfinite reconstruction')
    return dict(full_codes=tensor(codes,torch.uint8 if fp4 else torch.int8),prefix_bytes=tensor(prefixes,torch.uint8),
        hi=tensor(his,torch.float32 if fp4 else torch.int16),full_dq=full_tensor,predictor_dq=pred_tensor,
        exponent=torch.tensor(exps,dtype=torch.int32).reshape(blockshape),e8m0=torch.tensor([e+127 for e in exps],dtype=torch.uint8).reshape(blockshape),
        clamp_mask=tensor(clamps,torch.bool),zero_blocks=torch.tensor(zeros,dtype=torch.bool).reshape(blockshape),underflow_mask=tensor(under,torch.bool))

def sidecars(qc,kc,qs,ks,config):
    result=[]; fmt=config['predictor_format']
    for name,c,s in [('q',qc,qs),('k',kc,ks)]:
        bits=config[name+'_prefix_bits']
        item=sm8(c,bits) if fmt=='sm8' else mx_sidecar(c.float()*s.float()[...,None],bits,fmt=='mxfp4' or (fmt=='mxint8q_mxfp4k' and name=='k'))
        if fmt=='sm8':
            item['full_dq']=item['full_dq']*s.float()[...,None]
            item['predictor_dq']=item['predictor_dq']*s.float()[...,None]
        result.append(item)
    return result

def int64_dot(q,k):
    result=torch.zeros((*q.shape[:-1],k.shape[-2]),dtype=torch.int64)
    for d in range(q.shape[-1]):result+=q[...,d].long()[...,None]*k[...,d].long().unsqueeze(-2)
    return result

def predictor_logits(qside,kside,qs,ks,config,device='cpu',mathematical=False):
    dtype=torch.float64 if mathematical else torch.float32
    if config['predictor_format']=='sm8':
        dot=int64_dot(qside['hi'],kside['hi']).to(device=device,dtype=dtype) if mathematical else torch.matmul(qside['hi'].to(device=device,dtype=dtype),kside['hi'].to(device=device,dtype=dtype).transpose(-1,-2))
        return (((dot*(2**(16-config['q_prefix_bits']-config['k_prefix_bits'])))*qs.to(device=device,dtype=dtype)[...,None])*ks.to(device=device,dtype=dtype).unsqueeze(-2))/math.sqrt(qside['hi'].shape[-1])
    if mathematical:
        # Independent block-factor formula, with exact integer block dots.
        # E2M1 values multiplied by two are integers; MXINT8 uses saved c or h.
        fmt=config['predictor_format'];d=qside['hi'].shape[-1]
        result=torch.zeros((*qside['hi'].shape[:-1],kside['hi'].shape[-2]),dtype=torch.float64)
        for b in range(d//32):
            factors=[];integers=[]
            for name,side in [('q',qside),('k',kside)]:
                fp4=fmt=='mxfp4' or (fmt=='mxint8q_mxfp4k' and name=='k')
                integers.append((side['hi'][...,b*32:(b+1)*32]*(2 if fp4 else 1)).long())
                factors.append(side['exponent'][...,b].long()-(1 if fp4 else (2 if config[name+'_prefix_bits']==4 else 6)))
            dots=int64_dot(*integers).double()
            exponent=factors[0][...,None]+factors[1].unsqueeze(-2)
            result+=torch.ldexp(dots,exponent)
        return result/math.sqrt(d)
    q=qside['predictor_dq'].to(device=device,dtype=dtype).contiguous(); k=kside['predictor_dq'].to(device=device,dtype=dtype).contiguous()
    denominator=torch.tensor(math.sqrt(q.shape[-1]),device=device,dtype=dtype)
    return torch.div(q@k.transpose(-1,-2),denominator)

def top_p(logits,allowed,positions,p):
    """Independent stable selector on the supplied device and FP32 logits."""
    check(0<p<=1,'invalid threshold'); check(logits.dtype==torch.float32,'execution selector requires actual FP32 logits')
    domain=allowed.to(logits.device).expand_as(logits);check(domain.any(-1).all(),'empty legal domain')
    posorder=torch.argsort(positions.to(logits.device),stable=True)
    scores=logits.masked_fill(~domain,-torch.inf);probs=torch.softmax(scores,-1,dtype=torch.float32)
    order=posorder[torch.argsort(scores[...,posorder],descending=True,stable=True)]
    sortedprobs=torch.gather(probs,-1,order);cumulative=sortedprobs.cumsum(-1)
    # Previous cumulative below p includes the crossing item exactly once.
    keep=torch.cat([torch.ones_like(cumulative[...,:1],dtype=torch.bool),cumulative[...,:-1]<p],-1)
    if p==1:keep=torch.gather(domain,-1,order)
    mask=torch.zeros_like(domain).scatter(-1,order,keep)&domain
    return mask,probs,{'order':order,'cumsum':cumulative,'probabilities':probs}

def compare(got,expected,name,exact=False):
    check(torch.is_tensor(got) and got.shape==expected.shape,name+': shape')
    expected=expected.to(got.device)
    if exact or not expected.is_floating_point():
        check(got.dtype==expected.dtype and torch.equal(got.contiguous().reshape(-1).view(torch.uint8),expected.contiguous().reshape(-1).view(torch.uint8)),name+': exact mismatch');return {'exact':True}
    check(torch.equal(torch.isneginf(got),torch.isneginf(expected)),name+': illegal domain')
    legal=torch.isfinite(expected);check(torch.isfinite(got[legal]).all(),name+': nonfinite')
    err=float((got[legal].double()-expected[legal].double()).abs().max()) if legal.any() else 0.
    check(torch.allclose(got[legal],expected[legal],atol=ATOL,rtol=RTOL),name+f': tolerance max_abs={err}')
    return {'max_abs':err}


def mathematical_top_p(logits,allowed,positions,p):
    """CPU FP64 ideal semantics, reported separately from execution membership."""
    check(logits.device.type=='cpu' and logits.dtype==torch.float64,'mathematical selector requires CPU FP64')
    domain=allowed.expand_as(logits);mask=torch.zeros_like(domain);probs=torch.zeros_like(logits)
    for row,(scores,legal,out,pr) in enumerate(zip(logits.reshape(-1,logits.shape[-1]),domain.reshape(-1,logits.shape[-1]),mask.reshape(-1,logits.shape[-1]),probs.reshape(-1,logits.shape[-1]))):
        keys=[j for j in range(scores.numel()) if legal[j]];check(keys,'empty mathematical domain')
        order=sorted(keys,key=lambda j:(-float(scores[j]),int(positions[j])))
        maximum=max(float(scores[j]) for j in keys);weights={j:math.exp(float(scores[j])-maximum) for j in keys};denominator=math.fsum(weights.values())
        cumulative=0.
        for j in keys:pr[j]=weights[j]/denominator
        for j in order:
            out[j]=True;cumulative+=float(pr[j])
            if p<1 and cumulative>=p:break
    return mask,probs
