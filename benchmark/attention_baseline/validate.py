"""Small independent numerical and scoring acceptance; no model checkpoint needed."""
import argparse
import math
from pathlib import Path
import torch
from .common import configure_runtime,write_json,environment,scoring_ranges,code_provenance
from .replay import error,ATOL,RTOL
from twilight.pyimpl.quant_attention.intflash import quantize_per_token
from twilight.pyimpl.quant_attention.torch_backend import attention,causal_allowed,validate_mask


def source_quantize(x):
    # Fixed source formula, intentionally independent of production quantizer.
    maximum=torch.max(torch.abs(x),dim=-1)[0]
    scales=torch.where(maximum == 0, torch.ones_like(maximum), maximum/127)
    return torch.round(x/scales.unsqueeze(-1)).to(torch.int8),scales


def must_fail(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('invalid numerical input accepted')


def validate_device(device):
    tests=[]
    torch.manual_seed(0)
    for dtype in (torch.float16,torch.bfloat16):
        x=torch.randn(2,4,7,128,device=device,dtype=dtype)
        x[...,0]=170
        # Noncontiguous source exercises an independent cast and strided D axis.
        x=x.transpose(-1,-2).contiguous().transpose(-1,-2)
        assert not x.is_contiguous()
        qc,qs=source_quantize(x.to(torch.float16))
        got=quantize_per_token(x)
        assert torch.equal(got.codes,qc) and torch.equal(got.scales,qs), 'source mismatch'
        assert torch.equal(got.dq,qc.float()*qs.float().unsqueeze(-1)), 'DQ mismatch'
        tests.append('source_exact_'+str(dtype))
    # Constant, ties-to-even, signed values and nonzero groups adjacent to zero.
    x=torch.tensor([0,0.5,1.5,2.5,-0.5,-1.5,-2.5,127],device=device,dtype=torch.float16).reshape(1,1,1,8)
    c,s=source_quantize(x);g=quantize_per_token(x)
    assert torch.equal(g.codes,c) and torch.equal(g.scales,s)
    z=quantize_per_token(torch.zeros(1,2,3,128,device=device,dtype=torch.bfloat16))
    assert (z.codes==0).all() and (z.scales==1).all() and (z.dq==0).all()
    regression=quantize_per_token(torch.full((1,1,1,128),170.,device=device,dtype=torch.bfloat16))
    assert (regression.codes==127).all(), 'BF16 170 overflow regression'
    tests += ['rounding_boundaries','zero_groups','bf16_170']
    for label,value,dtype in [('nan',float('nan'),torch.float32),('inf',float('inf'),torch.float32),('cast_overflow',65536.,torch.float32),('scale_underflow',2**-24,torch.float16),('rounded_code_overflow',2**-17,torch.float16)]:
        must_fail(lambda:quantize_per_token(torch.full((1,1,1,128),value,device=device,dtype=dtype)))
        tests.append(label+'_rejected')
    # Native FP16 conversion may round tiny BF16 values to signed zero. Match
    # the fixed FP16 source formula exactly, including a wholly cast-zero group.
    for label,values in [('mixed_cast_underflow',[-2.211e-9, 1., -2., 0.]),
                         ('allzero_cast_group',[-2.211e-9, 1e-10, -1e-10, 0.])]:
        x=torch.tensor(values,device=device,dtype=torch.bfloat16).reshape(1,1,1,4)
        converted=x.to(torch.float16)
        lost=(x != 0) & (converted == 0)
        assert lost.any(), 'test must witness native cast underflow'
        c,s=source_quantize(converted);g=quantize_per_token(x)
        assert torch.equal(g.codes,c) and torch.equal(g.scales,s), label+' source mismatch'
        assert torch.equal(g.dq,c.float()*s.float().unsqueeze(-1)), label+' DQ mismatch'
        assert g.diagnostics['input_cast_nonzero_to_zero_count']==int(lost.sum())
        tests.append(label+'_allowed_source_exact')
    reports=[]
    for n in (1,63,64,65,127,128,129):
        for m in sorted({1,n}):
            # Explicit independent Hkv->Hq expansion; each source KV head repeated twice.
            q=torch.randn(1,4,m,128,device=device,dtype=torch.bfloat16)
            base_k=torch.randn(1,2,n,128,device=device,dtype=torch.bfloat16)
            base_v=torch.randn_like(base_k)
            k=torch.stack([base_k[:,h//2] for h in range(4)],dim=1)
            v=torch.stack([base_v[:,h//2] for h in range(4)],dim=1)
            qpos=torch.arange(n-m,n,device=device);kpos=torch.arange(n,device=device)
            mask=causal_allowed(qpos,kpos,device=device)
            independent=(kpos.view(1,-1)<=qpos.view(-1,1)).view(1,1,m,n)
            assert torch.equal(mask,independent)
            additive=torch.zeros((1,1,m,n+3),device=device);additive[...,:n].masked_fill_(~mask,-torch.inf)
            validate_mask(additive,mask)
            bad=torch.zeros((1,1,m,n),device=device);bad[...,0]=1
            must_fail(lambda:validate_mask(bad,mask))
            ref_logits=torch.einsum('bhmd,bhnd->bhmn',q.float(),k.float())/math.sqrt(128)
            ref_logits=ref_logits.masked_fill(~independent,-torch.inf)
            ref_probs=torch.exp(ref_logits-torch.logsumexp(ref_logits,dim=-1,keepdim=True))
            ref_pv=torch.einsum('bhmn,bhnd->bhmd',ref_probs,v.float())
            captured=[]
            def observer(start,end,l,p,pv):captured.append((start,end,l.clone(),p.clone(),pv.clone()))
            out=attention(q,k,v,mask,query_chunk=64,observer=observer)
            for name,actual,expected in [('pv',out,ref_pv),('logits',torch.cat([r[2] for r in captured],-2),ref_logits),('probs',torch.cat([r[3] for r in captured],-2),ref_probs)]:
                finite=torch.isfinite(expected)
                assert torch.equal(torch.isinf(actual),torch.isinf(expected))
                assert torch.allclose(actual[finite],expected[finite],atol=ATOL,rtol=RTOL), f'{device}/{n}/{m}/{name}: {error(actual[finite],expected[finite])}'
                reports.append({'n':n,'m':m,'field':name,**error(actual[finite],expected[finite])})
    tests+=['causal_square','causal_decode','gqa_expansion','chunk64_reference','mask_bias_rejected']
    return {'device':str(device),'tests':tests,'backend_errors':reports,'status':'complete'}


def run(output_dir):
    out=Path(output_dir)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True);write_json(out/'status.json',{'status':'running'})
    try:
        configure_runtime();code_provenance(out);devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        reports=[validate_device(d) for d in devices]
        lengths=(2,17,2048,2049,4096,4103)
        for t in lengths:
            ranges=scoring_ranges(t)
            targets=[j for r in ranges for j in range(r['lo'],r['target_end'])]
            assert targets==list(range(1,t)), f'inexact scoring coverage T={t}'
        must_fail(lambda:scoring_ranges(1))
        decode=[(127,128)]+[(q,q+1) for q in range(128,143)]
        assert len(decode)==16 and [t for _,t in decode]==list(range(128,144))
        report={'status':'complete','passed':True,'devices':reports,'scoring_lengths':list(lengths),'decode_targets':[t for _,t in decode],'atol':ATOL,'rtol':RTOL}
        write_json(out/'report.json',report);write_json(out/'environment.json',environment());write_json(out/'status.json',{'status':'complete','passed':True})
        return report
    except Exception as e:
        write_json(out/'status.json',{'status':'failed','passed':False,'reason':str(e)});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.output_dir)

if __name__=='__main__':main()
