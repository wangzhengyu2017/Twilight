"""Exhaustive scalar codecs and independent CPU/device numerical acceptance."""
import argparse
from pathlib import Path
import torch
from benchmark.attention_baseline.common import write_json,code_provenance,environment,canonical_hash,configure_runtime
from .formal import validate_formal_config
from .oracle import check,compare,sm8,mx_sidecar,source_quantization,sidecars,predictor_logits,top_p,int64_dot,mathematical_top_p


def check_device(device,configs):
    # Production imports stay in this test driver; the independent oracle imports none.
    from twilight.pyimpl.quant_attention.predictor_formats import sm8 as actual_sm,mx_sidecar as actual_mx
    from twilight.pyimpl.quant_attention.format_backend import attention_format
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    checks=[];source=torch.arange(-127,128,dtype=torch.int16)
    for bits in (4,8):
        expected=sm8(source,bits);got=actual_sm(source.to(device),bits)
        for name,value in expected.items():compare(got[name].cpu(),value,'SM '+name,True)
        check(torch.equal(expected['prefix_bytes'].long(),expected['full_codes'].long()//2**(8-bits)),'SM physical bit budget')
    check(torch.equal(sm8(source,8)['predictor_dq'],source.float()),'SM full identity');checks.append('all_255_SM_roundtrip_and_four_total_bits_including_sign')
    # Power boundaries, exact RNE ties, symmetric endpoints, zero blocks, E8M0 range.
    rows=[]
    for exponent in (-127,-126,-20,0,20,120,127):
        base=torch.tensor([0.,.5,1.,1.5,2.,2.5,63.5,64.,126.5,127.,127.5,-.5,-1.5,-126.5,-127.,-127.5]+[0.]*16)
        rows.append(torch.ldexp(base,torch.tensor(exponent-6)))
    rows.extend([torch.zeros(32),torch.tensor([0.,.25,.5,.75,1.,1.25,1.5,1.75,2.,2.5,3.,3.5,4.,5.,6.,7.]+[0.]*16)])
    values=torch.stack(rows)
    for fp4 in (False,True):
        # Native FP4 exponent -129 at tiny input is deliberately rejected.
        for row in values:
            for bits in ((4,) if fp4 else (4,8)):
                try:expected=mx_sidecar(row,bits,fp4)
                except ValueError:
                    try:actual_mx(row.to(device),bits,fp4)
                    except ValueError:continue
                    raise ValueError('production accepted forbidden exponent')
                got=actual_mx(row.to(device),bits,fp4)
                for name,value in expected.items():compare(got[name].cpu(),value,'MX '+name,True)
    # Fixed block maxima keep each tie/endpoint probe at the intended scale.
    endpoint=torch.tensor(127.5/64,dtype=torch.float32)
    endpoint_row=torch.zeros(32);endpoint_row[0]=torch.nextafter(torch.tensor(2.),torch.tensor(0.))
    probes=[torch.nextafter(endpoint,torch.tensor(0.)),endpoint,torch.nextafter(endpoint,torch.tensor(torch.inf))]
    endpoint_row[1:7]=torch.stack(probes+[-x for x in probes])
    fp4_rows=[]
    for midpoint in (.25,.75,1.25,1.75,2.5,3.5,5.,6.):
        x=torch.tensor(midpoint);probes=[torch.nextafter(x,torch.tensor(0.)),x,torch.nextafter(x,torch.tensor(torch.inf))]
        row=torch.zeros(32);row[0]=6.;row[1:7]=torch.stack(probes+[-v for v in probes]);fp4_rows.append(row)
    for row,fp4 in [(endpoint_row,False)]+[(row,True) for row in fp4_rows]:
        expected=mx_sidecar(row,4,fp4);got=actual_mx(row.to(device),4,fp4)
        for name,value in expected.items():compare(got[name].cpu(),value,'endpoint/tie neighbors '+name,True)
    for dimension in (1,31,33,127,129):
        for fn in (mx_sidecar,actual_mx):
            try:fn(torch.zeros(dimension,device=device),4)
            except ValueError:pass
            else:raise ValueError('incomplete MX tail accepted')
    # Every representable power transition and immediate binary32 neighbors.
    transition_rows=[]
    for exponent in range(-127,128):
        power=torch.tensor(2.**exponent,dtype=torch.float32)
        for maximum in (power,torch.nextafter(power,torch.tensor(0.)),torch.nextafter(power,torch.tensor(torch.inf))):
            if torch.isfinite(maximum) and maximum>0:
                row=torch.zeros(32);row[0]=maximum;row[1]=-maximum
                try:mx_sidecar(row,4)
                except ValueError:continue
                transition_rows.append(row)
    transitions=torch.stack(transition_rows)
    for bits in (4,8):
        expected=mx_sidecar(transitions,bits);got=actual_mx(transitions.to(device),bits)
        for name,value in expected.items():compare(got[name].cpu(),value,'all power boundaries '+name,True)
    # Every original source code, with representative exact dyadic real scales.
    original=torch.cat([source.to(torch.int8),torch.zeros(1,dtype=torch.int8)]).reshape(8,32)
    for scale_exponent in (-100,-20,0,20,100):
        dq=original.float()*2.**scale_exponent
        for bits in (4,8):
            expected=mx_sidecar(dq,bits);got=actual_mx(dq.to(device),bits)
            for name,value in expected.items():compare(got[name].cpu(),value,'all source255 scaled '+name,True)
    for exponent in (-128,128):
        row=torch.full((32,),2.**exponent,dtype=torch.float64).float()
        try:mx_sidecar(row,4)
        except ValueError:pass
        else:raise ValueError('oracle accepted invalid exponent')
        try:actual_mx(row.to(device),4)
        except ValueError:pass
        else:raise ValueError('production accepted invalid exponent')
    checks.append('MX_block32_exact_frexp_E8M0_RNE_endpoints_zero_exponent_edges_and_E2M1_ties')
    generator=torch.Generator().manual_seed(713);q=torch.randn((1,2,7,128),generator=generator).to(device);k=torch.randn((1,2,11,128),generator=generator).to(device);v=torch.randn((1,2,11,128),generator=generator).to(device)
    qq,kk=quantize_per_token(q),quantize_per_token(k)
    for original,quantized in [(q,qq),(k,kk)]:
        c,s=source_quantization(original);compare(quantized.codes,c,'original codes',True);compare(quantized.scales,s,'original scales',True)
    # BF16-to-FP16 native underflow may erase all input and selects scale one.
    zeroed=torch.full((1,1,1,128),2.**-100,dtype=torch.bfloat16,device=device)
    z=quantize_per_token(zeroed);c,s=source_quantization(zeroed)
    compare(z.codes,c,'native FP16 underflow codes',True);compare(z.scales,s,'native FP16 underflow scales',True)
    check((c==0).all() and (s==1).all(),'zero group convention')
    # A nonzero FP16 group whose scale underflows must fail, rather than fabricate codes.
    tiny=torch.full((1,1,1,128),2.**-24,dtype=torch.bfloat16,device=device)
    for fn in (source_quantization,quantize_per_token):
        try:fn(tiny)
        except (ValueError,RuntimeError):pass
        else:raise ValueError('nonzero source scale underflow accepted')
    allowed=(torch.arange(11,device=device)[None,:]<=torch.arange(4,11,device=device)[:,None])[None,None]
    errors={}
    for vid,config in configs.items():
        validate_formal_config(config);s=config['attention_slice'];qside,kside=sidecars(qq.codes.cpu(),kk.codes.cpu(),qq.scales.cpu(),kk.scales.cpu(),s);captured=[]
        mathematical=predictor_logits(qside,kside,qq.scales.cpu(),kk.scales.cpu(),s,'cpu',True)
        independently_decoded=(qside['predictor_dq'].double()@kside['predictor_dq'].double().transpose(-1,-2))/128**.5
        compare(mathematical,independently_decoded,'FP64 block factor reconstruction')
        ideal,_=mathematical_top_p(mathematical,allowed.cpu(),torch.arange(11),s['selection']['p'])
        check((ideal.sum(-1)>0).all(),'empty ideal mathematical selection')
        before=[t.clone() for t in (qq.codes,kk.codes,qq.scales,kk.scales)]
        pv=attention_format(q,k,v,allowed,qq,kk,s,query_chunk=3,observer=lambda start,end,payload:captured.append((start,end,payload)))
        for start,end,a in captured:
            sliced={name:(value[...,start:end,:] if value.ndim>=4 else value) for name,value in qside.items()}
            expected=predictor_logits(sliced,kside,qq.scales.cpu()[...,start:end],kk.scales.cpu(),s,device).masked_fill(~allowed[...,start:end,:],-torch.inf)
            compare(a['estimator_logits'],expected,'independent FP32 logits')
            chosen,probs,_=top_p(a['estimator_logits'],allowed[...,start:end,:],torch.arange(11),s['selection']['p'])
            compare(a['selected_mask'],chosen,'actual-logit membership',True)
            regenerated,_,_=top_p(expected,allowed[...,start:end,:],torch.arange(11),s['selection']['p']);compare(a['selected_mask'],regenerated,'regenerated membership',True)
            full=(qq.dq[...,start:end,:]@kk.dq.transpose(-1,-2))/128**.5;final=full.masked_fill(~chosen,-torch.inf)
            compare(a['final_selected_logits'],final,'original selected full logits')
            expected_pv=torch.softmax(final,-1)@v.float();compare(pv[...,start:end,:],expected_pv,'selected full PV')
            for prefix,side in [('q',sliced),('k',kside)]:
                for name,value in side.items():compare(a[prefix+'_format_'+name].cpu(),value,prefix+' sidecar '+name,True)
        for value,snapshot in zip((qq.codes,kk.codes,qq.scales,kk.scales),before):compare(value,snapshot,'source preservation',True)
        import copy
        p1=copy.deepcopy(s);p1['selection']['p']=1.
        dense=torch.softmax((qq.dq@kk.dq.transpose(-1,-2)/128**.5).masked_fill(~allowed,-torch.inf),-1)@v.float()
        compare(attention_format(q,k,v,allowed,qq,kk,p1,query_chunk=3),dense,'p1 original INT dense')
        errors[vid]='passed'
    # Constructed arithmetic: zero blocks, all-negative codes, cancellation,
    # disparate Q/K scales and sentinel dimensions on both sides of block32.
    from twilight.pyimpl.quant_attention.intflash import QuantizedTensor
    boundary=torch.tensor([-127,-126,-64,-63,-17,-16,-15,-1,0,1,15,16,17,63,64,126,127],dtype=torch.int8)
    for kind in ('zero','negative','cancellation'):
        qc=torch.zeros((1,2,2,128),dtype=torch.int8);kc=torch.zeros((1,2,5,128),dtype=torch.int8)
        if kind!='zero':
            qc.copy_(boundary[torch.arange(128)%len(boundary)].expand_as(qc));kc.copy_(boundary[(torch.arange(128)+3)%len(boundary)].expand_as(kc))
            if kind=='negative':qc=-qc.abs();kc=-kc.abs()
            else:kc[...,1::2]=-kc[...,1::2]
            for dimension in (0,30,31,32,33,126,127):
                value=boundary[dimension%len(boundary)]
                qc[...,dimension]=-value.abs() if kind=='negative' else value
            if kind=='negative':check((qc<=0).all() and (kc<=0).all(),'allnegative fixture')
        qc=qc.to(device);kc=kc.to(device);qs=torch.tensor([[[.125,.5],[2.,.25]]],dtype=torch.float16,device=device);ks=torch.tensor([[[2.,.25,1.,.5,.125],[.125,.5,2.,.25,1.]]],dtype=torch.float16,device=device)
        fq=qc.float()*qs.float()[...,None];fk=kc.float()*ks.float()[...,None]
        qqfixture=QuantizedTensor(qc,qs,fq);kkfixture=QuantizedTensor(kc,ks,fk);vv=torch.arange(128,device=device).float().expand(1,2,5,128)/128
        domain=torch.ones((1,1,2,5),dtype=torch.bool,device=device);domain[...,0,0]=False
        for vid,config in configs.items():
            s=config['attention_slice'];captured=[]
            actual=attention_format(fq,fk,vv,domain,qqfixture,kkfixture,s,query_chunk=2,observer=lambda st,en,a:captured.append(a))
            a=captured[0];qside,kside=sidecars(qc.cpu(),kc.cpu(),qs.cpu(),ks.cpu(),s)
            regen=predictor_logits(qside,kside,qs.cpu(),ks.cpu(),s,device).masked_fill(~domain,-torch.inf)
            compare(a['estimator_logits'],regen,kind+' constructed logits')
            mask,_,_=top_p(regen,domain,torch.arange(5),s['selection']['p']);compare(a['selected_mask'],mask,kind+' constructed membership',True)
            final=(fq@fk.transpose(-1,-2)/128**.5).masked_fill(~mask,-torch.inf)
            compare(a['final_selected_logits'],final,kind+' final original QK');compare(actual,torch.softmax(final,-1)@vv,kind+' PV')
    checks.append('constructed_zero_negative_cancellation_boundarycodes_blockboundaries_disparate_scales_tail_rejection')
    # Stable ties and exact equality crossing on actual execution device.
    scores=torch.zeros((1,1,1,4),device=device);domain=torch.ones_like(scores,dtype=torch.bool)
    chosen,_,_=top_p(scores,domain,torch.arange(4),.5);check(chosen.flatten().tolist()==[True,True,False,False],'stable tie/equality crossing')
    gauge=torch.tensor([[[[2.,1.,0.]]]],device=device);alllegal=torch.ones_like(gauge,dtype=torch.bool)
    first,_,_=top_p(gauge,alllegal,torch.arange(3),.85);second,_,_=top_p(2*gauge,alllegal,torch.arange(3),.85)
    check(not torch.equal(first,second),'softmax gauge sensitivity missing')
    full,_,_=top_p(gauge,alllegal,torch.arange(3),1.);check(torch.equal(full,alllegal),'p1 domain mismatch')
    checks.append('all_matrix_variants_independent_device_operands_logits_exact_sets_original_final_PV')
    return dict(status='passed',device=str(device),checks=checks,variants=errors)


def validate(matrix,device='cuda'):
    from .gate import read_matrix
    _,configs=read_matrix(matrix);configure_runtime();cpu=check_device('cpu',configs)
    gpu=check_device(device,configs) if device!='cpu' else {'status':'not_executed'}
    return dict(status='passed' if gpu['status']=='passed' else 'cpu_only',cpu=cpu,gpu=gpu,variant_ids=list(configs),semantic_config_sha256={vid:canonical_hash(c) for vid,c in configs.items()},code_sha256=code_provenance()['code_sha256'],environment_sha256=environment()['environment_sha256'],oracle_sha256=__import__('hashlib').sha256(Path(__file__).with_name('oracle.py').read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    try:report=validate(a.matrix,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
