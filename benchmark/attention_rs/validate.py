"""Exhaustive CPU oracle plus optional explicit CUDA numerical acceptance."""
import argparse
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import torch
from benchmark.attention_baseline.common import write_json,code_provenance,environment,canonical_hash,configure_runtime
from .formal import read_matrix,VARIANTS
from .oracle import check,compare,scalar_native,decode_words,source_sidecar,source_quantization,selection,semantic_fp64,ATOL,RTOL

def expect_failure(function,label):
    try:function()
    except (ValueError,TypeError,RuntimeError,AssertionError):return
    raise ValueError('accepted invalid input '+label)

def validate_device(device,configs):
    from twilight.pyimpl.quant_attention import rs_codec as codec
    from twilight.pyimpl.quant_attention import rs_backend as backend
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token,QuantizedTensor
    from twilight.pyimpl.rs_trace import RSCallCounters
    dev=torch.device(device);native=torch.arange(-136,120,dtype=torch.int16,device=dev);words=codec.encode_native(native)
    check(words.unique().numel()==256,'native bijection');compare(codec.decode_native(words).cpu(),native.cpu(),'native256 roundtrip',True)
    fixtures=[scalar_native(r) for r in range(-136,120)]
    compare(words.cpu(),torch.tensor([x[0] for x in fixtures],dtype=torch.uint8),'circuit versus independent quotient',True)
    compare(decode_words(words.cpu()),native.cpu(),'independent decoder256',True)
    subset=torch.arange(-127,120,dtype=torch.int16,device=dev).to(torch.int8);compare(codec.decode_native(codec.encode_native(subset)).cpu(),subset.cpu().to(torch.int16),'native valid int8 subset',True)
    check(int(codec.encode_native(torch.tensor([8],device=dev))[0])==0x18 and int(codec.encode_native(torch.tensor([-8],device=dev))[0])==0x08,'TC high-nibble shortcut fixtures')
    for invalid in (-137,120,128):expect_failure(lambda:codec.encode_native(torch.tensor([invalid],device=dev)),'native domain')
    for invalid in (-1,256):expect_failure(lambda:codec.decode_native(torch.tensor([invalid],device=dev)),'packed byte domain')
    expect_failure(lambda:codec.encode_native(torch.tensor([8.],device=dev)),'floating native')
    source=torch.arange(-127,128,dtype=torch.int16,device=dev).to(torch.int8).reshape(1,1,1,255);scales=torch.tensor([[[.125]]],device=dev,dtype=torch.float16);quant=QuantizedTensor(source,scales,source.float()*scales.float()[...,None])
    for bits in (4,8):
        side=codec.sidecar(quant,bits);reference=source_sidecar(source.cpu(),scales.cpu(),bits)
        for name,value in reference.items():compare(side[name].cpu(),value,'source255 '+str(bits)+' '+name,True)
        check(side['scales'].data_ptr()==scales.data_ptr(),'source scales must alias unchanged');check(side['saturation_mask'].sum()==8,'exactly eight source saturation codes')
        check(side['prefix_residual'].min()==-8 and side['prefix_residual'].max()==7,'signed low residual')
    # Original source FP16 conversion and underflow semantics remain unchanged.
    for x in (torch.zeros(1,2,2,8),torch.tensor([1e-20,1.,-.5,0.]).reshape(1,1,1,4)):
        original=x.to(dev);actual=quantize_per_token(original);c,s=source_quantization(original.cpu());compare(actual.codes.cpu(),c,'source codes',True);compare(actual.scales.cpu(),s,'source scale',True)
    expect_failure(lambda:quantize_per_token(torch.full((1,1,1,4),2**-24,device=dev)),'nonzero source scale underflow')
    expect_failure(lambda:quantize_per_token(torch.full((1,1,1,4),1e10,device=dev)),'source FP16 overflow')
    generator=torch.Generator().manual_seed(123);q=torch.randn((1,4,3,16),generator=generator).to(dev);k=torch.randn((1,4,7,16),generator=generator).to(dev);v=torch.randn((1,4,7,16),generator=generator).to(dev)
    # Unequal Q/K scales, repeated GQA heads, saturated endpoints and cancellation.
    q[...,0]=q.abs().amax(-1);k[...,0]=k.abs().amax(-1);k[:,1]=k[:,0];k[:,3]=k[:,2];v[:,1]=v[:,0];v[:,3]=v[:,2]
    qq,kk=quantize_per_token(q),quantize_per_token(k);domain=torch.tensor([[[[1,1,1,0,0,0,0],[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]]]],device=dev,dtype=torch.bool)
    report=[]
    for vid,config in configs.items():
        cfg=config['attention_rs'];backend.validate_rs_config(cfg,'intflash_qk8');observations=[];stats=RSCallCounters()
        originals=[tensor.clone() for tensor in (qq.codes,qq.scales,qq.dq,kk.codes,kk.scales,kk.dq)]
        with patch.object(backend,'exact_top_p',wraps=backend.exact_top_p) as selector,patch.object(backend,'selected_full_logits',wraps=backend.selected_full_logits) as selected:
            actual=backend.attention_rs(q,k,v,domain,qq,kk,cfg,3,lambda a,b,p:observations.append(p),stats,0)
            check(selector.call_count==cfg['selection_passes'],'actual selector count');check(selected.call_count==int(cfg['mode']=='msb_topp'),'selected original recompute count')
        for saved,tensor in zip(originals,(qq.codes,qq.scales,qq.dq,kk.codes,kk.scales,kk.dq)):compare(tensor.cpu(),saved.cpu(),'original source immutable',True)
        a=observations[0];qs=source_sidecar(qq.codes.cpu(),qq.scales.cpu(),cfg['q_bits']);ks=source_sidecar(kk.codes.cpu(),kk.scales.cpu(),cfg['k_bits'])
        hh=qs['H'].long()@ks['H'].long().transpose(-1,-2);hl=qs['H'].long()@ks['L'].long().transpose(-1,-2);lh=qs['L'].long()@ks['H'].long().transpose(-1,-2);ll=qs['L'].long()@ks['L'].long().transpose(-1,-2)
        compare(qs['native_int'].long()@ks['native_int'].long().transpose(-1,-2),256*hh+16*hl+16*lh+ll,'independent algebraic signed-slice expansion',True)
        qp,kp=qs['predictor_dq'].to(dev),ks['predictor_dq'].to(dev);logits=(qp@kp.transpose(-1,-2))/torch.tensor(math.sqrt(16),device=dev,dtype=torch.float32);logits=logits.masked_fill(~domain,-torch.inf)
        mask,indices,counts,valid,ep=selection(logits,domain,cfg['selection']['p'])
        for name,expected in [('selected_mask',mask),('selected_indices',indices),('selected_counts',counts),('selected_valid_mask',valid)]:compare(a[name].cpu(),expected.cpu(),vid+' '+name,True)
        if cfg['mode']=='msb_topp':
            final=torch.full_like(logits,-torch.inf)
            for h in range(4):
                for row in range(3):
                    ix=indices[0,h,row][valid[0,h,row]];qdq=qq.codes[0,h,row].float()*qq.scales[0,h,row].float();kdq=kk.codes[0,h,ix].float()*kk.scales[0,h,ix].float()[:,None]
                    # Match independent selected bmm reduction, DQ-first.
                    final[0,h,row,ix]=torch.bmm(qdq.reshape(1,1,16),kdq.reshape(1,-1,16).transpose(-1,-2)).reshape(-1)/math.sqrt(16)
        else:final=logits.masked_fill(~mask,-torch.inf)
        probs=final.softmax(-1);pv=probs@v.float();errors={name:compare(a[name].cpu(),expected.cpu(),vid+' '+name) for name,expected in [('estimator_logits',logits),('final_selected_logits',final),('final_selected_probs',probs),('pv_head',pv)]};compare(actual.cpu(),pv.cpu(),vid+' result')
        costs=stats.slice_costs();check((costs['selected_full_qk_macs']>0)==(cfg['mode']=='msb_topp'),'native final source-QK counts zero')
        # Corrupting packed decode must change output; no original-DQ bypass.
        with patch.object(codec,'encode_native',side_effect=lambda native:torch.zeros_like(native,dtype=torch.uint8)):
            corrupted=backend.attention_rs(q,k,v,domain,qq,kk,cfg,3)
        check(not torch.equal(corrupted,actual),'packed byte numerical dependency')
        for field,value in [('final_operand_origin','wrong'),('final_scores','wrong'),('final_execution','wrong'),('q_bits',True),('selection_passes',True)]:
            bad=copy.deepcopy(cfg);bad[field]=value;expect_failure(lambda:backend.validate_rs_config(bad,'intflash_qk8'),vid+' '+field)
        report.append(dict(variant_id=vid,passed=True,errors=errors,selector_calls=cfg['selection_passes'],selected_original_calls=int(cfg['mode']=='msb_topp')))
    # Ties, all-negative cancellation, crossing item, illegal high scores, nonempty domain.
    boundary=[]
    for scores in ([0.,0.,0.,0.],[-100.,-100.,-100.,-100.],[1.,-1.,1.,-1.],[100.,0.,0.,0.]):
        logits=torch.tensor(scores,device=dev).reshape(1,1,1,4);allowed=torch.tensor([[[[0,1,1,1]]]],device=dev,dtype=torch.bool)
        for p in (.95,.9,.85):
            production=backend.exact_top_p(logits,allowed,p);oracle=selection(logits,allowed,p)
            compare(production[0].cpu(),oracle[0].cpu(),'selector boundary',True);fp64,detail=semantic_fp64(logits,allowed,p)
            boundary.append(dict(scores=scores,p=p,fp32_fp64_same=torch.equal(production[0].cpu(),fp64),boundaries=detail))
    expect_failure(lambda:backend.attention_rs(q,k,v,torch.zeros_like(domain),qq,kk,next(iter(configs.values()))['attention_rs']),'empty causal rows')
    return dict(status='passed',device=str(device),native_values=256,source_values=255,source_saturation_codes=list(range(120,128)),variants=report,selector_boundary_artifacts=boundary,independent_helpers=True)

def validate(matrix,device='cpu'):
    # Configure before the first CUDA query/transfer in standalone validation.
    configure_runtime()
    runtime=environment()
    check(all(runtime.get(key)==value for key,value in dict(cublas_workspace_config=':4096:8',tf32_matmul=False,tf32_cudnn=False,float32_matmul_precision='highest',deterministic_algorithms=True).items()),'RS validation pinned runtime differs')
    m,configs=read_matrix(matrix);torch.set_num_threads(2);cpu=validate_device('cpu',configs)
    result=dict(status='passed' if device=='cuda' else 'cpu_only',cpu=cpu,gpu=validate_device('cuda',configs) if device=='cuda' else dict(status='not_executed'),variant_ids=list(configs),semantic_config_sha256={vid:canonical_hash(cfg) for vid,cfg in configs.items()},atol=ATOL,rtol=RTOL,code_sha256=code_provenance()['code_sha256'],environment_sha256=runtime['environment_sha256'],runtime_initialized=True)
    check(environment()['environment_sha256']==runtime['environment_sha256'],'RS validation runtime/environment changed during numerical checks')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    try:report=validate(a.matrix,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report);print(json.dumps(dict(status=report['status'],cpu=report['cpu']['status'],gpu=report['gpu']['status'])))
if __name__=='__main__':main()
