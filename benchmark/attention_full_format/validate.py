"""Independent full-byte TopP arithmetic/support tests; CPU-safe without a model."""
import argparse
import copy
import math
from pathlib import Path
from unittest.mock import patch
import torch
from benchmark.attention_baseline.common import configure_runtime,write_json,canonical_hash,code_provenance,environment,load_config
from benchmark.attention_dense_format.validate import rejects,cache_check
from benchmark.attention_dense_format.oracle import check,compare,scalar_encode,top_p,ATOL,RTOL
from .formal import read_matrix


def check_device(device,configs):
    from twilight.pyimpl.quant_attention.full_format_backend import attention_full_format
    from twilight.pyimpl.quant_attention.slice_backend import exact_top_p,attention_slice
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.quant_attention.torch_backend import causal_allowed,validate_mask
    from twilight.pyimpl.quant_attention.forward import validate_config
    torch.manual_seed(1029);q=torch.randn((1,4,7,128),device=device,dtype=torch.bfloat16);k0=torch.randn((1,2,11,128),device=device,dtype=torch.bfloat16);v0=torch.randn_like(k0);k=k0.repeat_interleave(2,1);v=v0.repeat_interleave(2,1)
    qq,kk=quantize_per_token(q),quantize_per_token(k);allowed=causal_allowed(torch.arange(4,11,device=device),torch.arange(11,device=device));variants={}
    for vid,cfg in configs.items():
        s=cfg['attention_full_format'];validate_config(cfg['attention_quantization'],None,None,s)
        qside,kside=[scalar_encode(quant.codes.cpu(),quant.scales.cpu(),s['format']) for quant in (qq,kk)]
        captured=[];before=[t.clone() for t in (qq.codes,qq.scales,kk.codes,kk.scales)]
        with patch('twilight.pyimpl.quant_attention.full_format_backend.exact_top_p',wraps=exact_top_p) as selector,patch('twilight.pyimpl.quant_attention.slice_backend.selected_full_logits',side_effect=AssertionError('full-format recomputed original source')):
            output=attention_full_format(q,k,v,allowed,qq,kk,s,3,lambda start,end,payload:captured.append((start,end,payload)))
            check(selector.call_count==3,'requires exactly one selector pass per query chunk')
        for start,end,payload in captured:
            full=(qside['full_dq'][...,start:end,:].to(device)@kside['full_dq'].to(device).transpose(-1,-2))/math.sqrt(128);full.masked_fill_(~allowed[...,start:end,:],-torch.inf)
            compare(payload['full_format_logits'],full,'independent full-format logits')
            mask,full_probs,_=top_p(full,allowed[...,start:end,:],torch.arange(11),s['selection']['p']);compare(payload['selected_mask'],mask,'exact independent selected support',True)
            compare(payload['full_format_probs'],full_probs,'full distribution')
            final=full.masked_fill(~mask,-torch.inf);probability=final.softmax(-1);compare(payload['final_selected_logits'],final,'retained identical format scores',True)
            compare(payload['final_selected_probs'],probability,'selected renormalization');compare(output[...,start:end,:],probability@v.float(),'selected PV')
            check((payload['final_selected_probs'][~mask]==0).all() and torch.allclose(payload['final_selected_probs'].sum(-1),torch.ones_like(mask.sum(-1),dtype=torch.float32),atol=1e-6,rtol=1e-6),'support/normalization')
            indices=payload['selected_indices'];valid=payload['selected_valid_mask'];check(torch.equal(valid.sum(-1),mask.sum(-1)) and (indices[~valid]==-1).all(),'selected count/padding')
            compare(((indices[...,None]==torch.arange(11,device=device))&valid[...,None]).any(-2),mask,'selected IDs/mask equality',True)
            for prefix,reference in (('q',qside),('k',kside)):
                for name,value in reference.items():compare(payload[prefix+'_format_'+name].cpu(),value[...,start:end,:] if prefix=='q' else value,'actual byte metadata '+name,True)
        for value,snapshot in zip((qq.codes,qq.scales,kk.codes,kk.scales),before):compare(value,snapshot,'source unchanged',True)
        if s['format']=='sm8':
            suffix=vid[-3:];control_root=Path('/output/project/SPEC/remote-gpu-platform-slice-SLICE-v2/configs') if suffix=='095' else Path('/output/project/experiments/stp-slice-thresholds/handoffs')/('p'+suffix)/'configs';control=load_config(control_root/('int_topp_p'+suffix+'.json'))['attention_slice']
            compare(output,attention_slice(q,k,v,allowed,qq,kk,control,3),'SM equals source INT TopP exact',True)
        else:
            check(not torch.equal(qside['full_dq'],qq.dq.cpu()),'MX fixture must distinguish source INT from format operands')
        rejects(lambda:attention_full_format(q,k,v,torch.zeros_like(allowed),qq,kk,s))
        invalid=copy.deepcopy(s);invalid['final_operand_profile']='source_int8';rejects(lambda:attention_full_format(q,k,v,allowed,qq,kk,invalid))
        cache_check(device,cfg);variants[vid]='passed'
    # Independent boundary examples include exact equality, ties, full legal domain.
    for p,expected in ((.5,[True,True,False,False]),(1.,[True,True,True,True])):
        logits=torch.zeros((1,1,1,4),device=device);domain=torch.ones_like(logits,dtype=torch.bool)
        mask,_,counts,_,_=exact_top_p(logits,domain,p);check(mask.flatten().tolist()==expected,'stable ties/minimal crossing p'+str(p))
        independent,_,_=top_p(logits,domain,torch.arange(4),p);compare(mask,independent,'boundary selector oracle',True)
    for p in (0.,-1.,1.01,float('nan')):rejects(lambda:exact_top_p(torch.zeros((1,1,1,1),device=device),torch.ones((1,1,1,1),device=device,dtype=torch.bool),p))
    additive=torch.zeros((1,1,7,11),device=device).masked_fill(~allowed,-torch.inf);validate_mask(additive,allowed)
    biased=additive.clone();biased[...,0]=1;rejects(lambda:validate_mask(biased,allowed))
    return dict(status='passed',device=str(device),variants=variants,checks=['independent_scalar_bytes','one_selector_per_chunk','same_retained_full_format_scores','no_source_INT_recompute','selected_indices_counts_validity_support','selected_softmax_PV','SM_source_INT_exact','MX_source_distinction','GQA_original_cache_RoPE_once_prefill_decode','stable_ties_equal_crossing_p1','empty_rows_bias_rejected'])


def validate(matrix,device='cpu'):
    _,configs=read_matrix(matrix);configure_runtime();cpu=check_device('cpu',configs);gpu=check_device(device,configs) if device!='cpu' else dict(status='not_executed')
    return dict(status='passed' if gpu['status']=='passed' else 'cpu_only',cpu=cpu,gpu=gpu,variant_ids=list(configs),semantic_config_sha256={v:canonical_hash(c) for v,c in configs.items()},code_sha256=code_provenance()['code_sha256'],environment_sha256=environment()['environment_sha256'],oracle_sha256=__import__('hashlib').sha256(Path(__file__).parents[1].joinpath('attention_dense_format/oracle.py').read_bytes()).hexdigest(),atol=ATOL,rtol=RTOL)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    try:report=validate(a.matrix,a.device)
    except Exception as exc:write_json(a.output,dict(status='failed',passed=False,reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
