"""Independent scalar codec and small attention/cache validation; no model run."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from benchmark.attention_baseline.common import configure_runtime,write_json,canonical_hash,code_provenance,environment,scoring_ranges
from .formal import read_matrix,validate_formal_config
from .oracle import check,compare,scalar_decode,scalar_encode,source_quantization,formats,mx_sidecar,ATOL,RTOL


def rejects(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise ValueError('Unsupported input accepted')


def cache_check(device,config):
    """Exercise real common_forward with independent projections/RoPE/cache checks."""
    from transformers.cache_utils import DynamicCache
    from twilight.pyimpl.quant_attention.forward import common_forward
    from twilight.pyimpl.trace import forward_context
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.quant_attention.dense_format_backend import attention_dense_format
    from twilight.pyimpl.quant_attention.torch_backend import causal_allowed
    torch.manual_seed(51)
    mod=SimpleNamespace(layer_id=0,layer_idx=0,num_heads=4,num_key_value_heads=2,num_key_value_groups=2,head_dim=32,hidden_size=128)
    mod.state=SimpleNamespace(attention_quantization=config['attention_quantization'],attention_slice=None,attention_dense_format=config.get('attention_dense_format'),attention_full_format=config.get('attention_full_format'),query_chunk=3)
    for name,nout in [('q_proj',128),('k_proj',64),('v_proj',64),('o_proj',128)]:setattr(mod,name,torch.nn.Linear(128,nout,bias=False,device=device,dtype=torch.bfloat16))
    cache=DynamicCache();allk=[];allv=[]
    for positions in (list(range(5)),[5],[6]):
        hidden=torch.randn(1,len(positions),128,device=device,dtype=torch.bfloat16)
        pos=torch.tensor(positions,device=device,dtype=torch.long);freq=torch.arange(16,device=device).float()/32+.01
        angles=pos.float()[:,None]*freq[None,:];angles=torch.cat([angles,angles],-1)[None]
        cos,sin=angles.cos().to(torch.bfloat16),angles.sin().to(torch.bfloat16)
        q=mod.q_proj(hidden).view(1,-1,4,32).transpose(1,2);k=mod.k_proj(hidden).view(1,-1,2,32).transpose(1,2);v=mod.v_proj(hidden).view(1,-1,2,32).transpose(1,2)
        rotate=lambda x:torch.cat([-x[...,16:],x[...,:16]],-1)
        q=q*cos[:,None]+rotate(q)*sin[:,None];k=k*cos[:,None]+rotate(k)*sin[:,None]
        k=k.repeat_interleave(2,1);v=v.repeat_interleave(2,1);allk.append(k);allv.append(v)
        fullk=torch.cat(allk,-2);fullv=torch.cat(allv,-2);keys=list(range(fullk.shape[-2]))
        domain=causal_allowed(pos,torch.tensor(keys,device=device))
        if config.get('attention_full_format'):
            from twilight.pyimpl.quant_attention.full_format_backend import attention_full_format
            backend=attention_full_format;s=config['attention_full_format']
        else:backend=attention_dense_format;s=config['attention_dense_format']
        pv=backend(q,fullk,fullv,domain,quantize_per_token(q),quantize_per_token(fullk),s,3)
        expected=mod.o_proj(pv.to(hidden.dtype).transpose(1,2).reshape(1,-1,128).contiguous())
        with forward_context(run_id='test',sample_id=0,window_id=None,stage='prefill' if len(positions)>1 else 'decode',global_query_positions=positions,position_ids=positions,key_positions=keys,expected_cache_length=len(keys),cache_position=positions):
            actual,_,returned=common_forward(mod,hidden,position_ids=pos[None],past_key_value=cache,use_cache=True,cache_position=pos,position_embeddings=(cos,sin))
        compare(actual,expected,'prefill/decode output exact',True)
        check(returned is cache and cache.get_seq_length(0)==len(keys),'cache identity/length')
        compare(cache.key_cache[0],fullk,'original floating K cache',True);compare(cache.value_cache[0],fullv,'original floating V cache',True)
        check(cache.key_cache[0].dtype==torch.bfloat16,'cache dtype altered')


def check_device(device,configs):
    from twilight.pyimpl.quant_attention.predictor_formats import sm8,dense_sidecar,mx_sidecar as encode_mx,decode_sm8_bytes,decode_mx_bytes
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.quant_attention.dense_format_backend import attention_dense_format
    from twilight.pyimpl.quant_attention.torch_backend import attention,causal_allowed,validate_mask
    from twilight.pyimpl.quant_attention.forward import validate_config
    tests=[]
    codes=torch.arange(-127,128,dtype=torch.int16).to(torch.int8).reshape(1,1,1,255)
    words=sm8(codes.to(device),8)['full_codes'];scales=torch.ones((1,1,1),dtype=torch.float16,device=device)
    compare(decode_sm8_bytes(words,scales).cpu(),codes.float(),'all255 SM roundtrip',True)
    allwords=torch.arange(256,dtype=torch.int16).to(torch.uint8).reshape(1,1,1,256)
    expected=scalar_decode(allwords,'sm8',scales.cpu())
    got=decode_sm8_bytes(allwords.to(device),scales);compare(got.cpu(),expected,'all256 SM stored bytes',True)
    check(not torch.signbit(got[...,128]).any(),'SM negative zero must decode positive')
    tests.append('exhaustive_SM255_roundtrip_SM256_decode_negative_zero')
    # Every FP4 code, signs, both zero encodings, finite exponent boundaries.
    fpwords=torch.arange(16,dtype=torch.uint8).repeat(2).reshape(1,1,1,32)
    for e in (-127,-126,-1,0,1,124):
        metadata=torch.tensor([[[[e+127]]]],dtype=torch.uint8)
        expected=scalar_decode(fpwords,'mxfp4',e8m0=metadata)
        compare(decode_mx_bytes(fpwords.to(device),metadata.to(device),True).cpu(),expected,'all16 FP4 codes e'+str(e),True)
    tests.append('exhaustive_FP4_16_codes_E8M0_bounds_subnormal_negative_zero')
    # Scalar Python oracle uses binary64 math.frexp, round (RNE), and E2M1 tie-even.
    fixtures=[torch.zeros(32),torch.tensor([-127,-126,-64,-63,-17,-16,-15,-1,0,1,15,16,17,63,64,126,127]*2)[:32].float()/64,
              torch.tensor([0,.25,.5,.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,7]*2).float()]
    for exponent in (-126,-120,-1,0,1,126):fixtures.append(torch.ldexp(torch.tensor([0.,1.,-1.,1.5,-1.5,1.984375,-1.984375,1.9921875]*4),torch.tensor(exponent)))
    for values in fixtures:
        for fp4 in (False,True):
            try:expected=mx_sidecar(values.reshape(1,1,1,32),4 if fp4 else 8,fp4)
            except ValueError:
                rejects(lambda:encode_mx(values.to(device).reshape(1,1,1,32),4 if fp4 else 8,fp4));continue
            got=encode_mx(values.to(device).reshape(1,1,1,32),4 if fp4 else 8,fp4)
            for field,value in expected.items():compare(got[field].cpu(),value,'MX scalar '+field,True)
            compare(decode_mx_bytes(got['full_codes'],got['e8m0'],fp4).cpu(),expected['full_dq'],'actual MX byte decode',True)
    for bad in (torch.full((32,),float('nan')),torch.full((32,),float('inf')),torch.full((32,),2.**-128),torch.ones(33)):
        rejects(lambda:encode_mx(bad.to(device),8))
    rejects(lambda:decode_mx_bytes(torch.full((32,),-128,dtype=torch.int8,device=device),torch.tensor([127],dtype=torch.uint8,device=device)))
    rejects(lambda:decode_mx_bytes(fpwords.to(device),torch.tensor([[[[255]]]],dtype=torch.uint8,device=device),True))
    rejects(lambda:decode_mx_bytes(torch.full_like(fpwords,16).to(device),torch.tensor([[[[127]]]],dtype=torch.uint8,device=device),True))
    tests.append('MX_RNE_saturation_endpoints_ties_zero_block_exponent_flags_nonfinite_tail_rejections')
    torch.manual_seed(713);q=torch.randn((1,4,7,128),device=device,dtype=torch.bfloat16)
    k0=torch.randn((1,2,11,128),device=device,dtype=torch.bfloat16);v0=torch.randn_like(k0);k=k0.repeat_interleave(2,1);v=v0.repeat_interleave(2,1)
    qq,kk=quantize_per_token(q),quantize_per_token(k)
    for source,quant in ((q,qq),(k,kk)):
        c,s=source_quantization(source);compare(quant.codes,c,'source codes',True);compare(quant.scales,s,'source scale',True)
    for value in (0.,2.**-100):
        zero=quantize_per_token(torch.full((1,1,1,128),value,device=device,dtype=torch.bfloat16));check((zero.codes==0).all() and (zero.scales==1).all(),'native underflow zero group')
    rejects(lambda:quantize_per_token(torch.full((1,1,1,128),2.**-24,device=device,dtype=torch.bfloat16)))
    allowed=causal_allowed(torch.arange(4,11,device=device),torch.arange(11,device=device));additive=torch.zeros((1,1,7,11),device=device).masked_fill(~allowed,-torch.inf)
    validate_mask(additive,allowed)
    rejects(lambda:validate_mask(torch.ones_like(additive),allowed))
    padding=additive.clone();padding[...,0]=-torch.inf;rejects(lambda:validate_mask(padding,allowed))
    rejects(lambda:causal_allowed([-1],[0],device))
    checks={}
    for vid,config in configs.items():
        validate_formal_config(config);validate_config(config['attention_quantization'],None,config['attention_dense_format'])
        s=config['attention_dense_format'];side=[scalar_encode(quant.codes.cpu(),quant.scales.cpu(),fmt) for quant,fmt in zip((qq,kk),formats(s))]
        before=[t.clone() for t in (qq.codes,qq.scales,kk.codes,kk.scales)];captured=[]
        with patch('twilight.pyimpl.quant_attention.slice_backend.exact_top_p',side_effect=AssertionError('dense called selector')):
            result=attention_dense_format(q,k,v,allowed,qq,kk,s,3,lambda st,en,payload:captured.append((st,en,payload)))
        qdq,kdq=[x['full_dq'].to(device) for x in side]
        for start,end,payload in captured:
            expected=(qdq[...,start:end,:]@kdq.transpose(-1,-2))/math.sqrt(128);expected.masked_fill_(~allowed[...,start:end,:],-torch.inf)
            compare(payload['logits_e'],expected,'independent dense logits')
            probability=expected.softmax(-1);compare(payload['probs'],probability,'dense softmax')
            compare(result[...,start:end,:],probability@v.float(),'dense PV')
            compare(payload['execution_mask'],allowed[...,start:end,:].expand_as(expected),'all legal executed',True)
            for prefix,reference in zip(('q','k'),side):
                for field,value in reference.items():compare(payload[prefix+'_format_'+field].cpu(),value[...,start:end,:] if prefix=='q' else value,'actual sidecar '+field,True)
        for value,snapshot in zip((qq.codes,qq.scales,kk.codes,kk.scales),before):compare(value,snapshot,'source immutable',True)
        if s['format']=='sm8':
            compare(result,attention(qq.dq,kk.dq,v,allowed,3),'SM common attention exact',True)
        rejects(lambda:attention_dense_format(q,k,v,torch.zeros_like(allowed),qq,kk,s))
        cache_check(device,config);checks[vid]='passed'
    tests+=['zero_selector_calls','causal_and_supported_additive_mask','padding_bias_empty_rows_rejected','GQA_original_cache_prefill_decode_positions_RoPE_once','SM_source_operands_attention_exact']
    return dict(status='passed',device=str(device),checks=tests,variants=checks)


def validate(matrix,device):
    _,configs=read_matrix(matrix);configure_runtime();cpu=check_device('cpu',configs)
    gpu=check_device(device,configs) if device!='cpu' else dict(status='not_executed')
    check(sum(r['target_end']-r['lo'] for r in scoring_ranges(289076))==289075 and len(scoring_ranges(289076))==282,'frozen targets/windows changed')
    return dict(status='passed' if gpu['status']=='passed' else 'cpu_only',cpu=cpu,gpu=gpu,variant_ids=list(configs),semantic_config_sha256={v:canonical_hash(c) for v,c in configs.items()},code_sha256=code_provenance()['code_sha256'],environment_sha256=environment()['environment_sha256'],oracle_sha256=__import__('hashlib').sha256(Path(__file__).with_name('oracle.py').read_bytes()).hexdigest(),atol=ATOL,rtol=RTOL)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    try:report=validate(a.matrix,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
