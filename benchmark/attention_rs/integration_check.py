"""CPU-only shared-interface regressions: old18, dense4, full6 and RS10."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from benchmark.attention_baseline.common import ROOT,load_config,write_json,canonical_hash,code_provenance
from .formal import read_matrix
from .oracle import check,compare

def rs_cache_check(config):
    from transformers.cache_utils import DynamicCache
    from twilight.pyimpl.quant_attention import forward as shared
    from twilight.pyimpl.quant_attention.rs_backend import attention_rs
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.quant_attention.torch_backend import causal_allowed
    from twilight.pyimpl.trace import forward_context
    torch.manual_seed(51);module=SimpleNamespace(layer_id=0,layer_idx=0,num_heads=4,num_key_value_heads=2,num_key_value_groups=2,head_dim=32,hidden_size=128)
    module.state=SimpleNamespace(attention_quantization=config['attention_quantization'],attention_slice=None,attention_dense_format=None,attention_full_format=None,attention_rs=config['attention_rs'],query_chunk=3)
    for name,width in [('q_proj',128),('k_proj',64),('v_proj',64),('o_proj',128)]:setattr(module,name,torch.nn.Linear(128,width,bias=False,dtype=torch.bfloat16))
    cache=DynamicCache();allk=[];allv=[]
    for positions in (list(range(5)),[5],[6]):
        hidden=torch.randn(1,len(positions),128,dtype=torch.bfloat16);position=torch.tensor(positions);frequency=torch.arange(16).float()/32+.01;angle=position.float()[:,None]*frequency[None,:];angle=torch.cat([angle,angle],-1)[None];cos,sin=angle.cos().to(torch.bfloat16),angle.sin().to(torch.bfloat16)
        q=module.q_proj(hidden).view(1,-1,4,32).transpose(1,2);k=module.k_proj(hidden).view(1,-1,2,32).transpose(1,2);v=module.v_proj(hidden).view(1,-1,2,32).transpose(1,2)
        rotate=lambda tensor:torch.cat([-tensor[...,16:],tensor[...,:16]],-1)
        q=q*cos[:,None]+rotate(q)*sin[:,None];k=k*cos[:,None]+rotate(k)*sin[:,None];k=k.repeat_interleave(2,1);v=v.repeat_interleave(2,1);allk.append(k);allv.append(v)
        fullk,fullv=torch.cat(allk,-2),torch.cat(allv,-2);keys=list(range(fullk.shape[-2]));legal=causal_allowed(position,torch.tensor(keys),'cpu')
        pv=attention_rs(q,fullk,fullv,legal,quantize_per_token(q),quantize_per_token(fullk),config['attention_rs'],3);expected=module.o_proj(pv.to(hidden.dtype).transpose(1,2).reshape(1,-1,128).contiguous())
        with patch.object(shared,'quantize_per_token',wraps=quantize_per_token) as source_calls,forward_context(run_id='synthetic-rs-cache',sample_id=0,window_id=None,stage='prefill' if len(positions)>1 else 'decode',global_query_positions=positions,position_ids=positions,key_positions=keys,expected_cache_length=len(keys),cache_position=positions):
            actual,_,returned=shared.common_forward(module,hidden,position_ids=position[None],past_key_value=cache,use_cache=True,cache_position=position,position_embeddings=(cos,sin))
            check(source_calls.call_count==2,'one source quantization per Q/K forward operand')
        compare(actual,expected,'RS common forward dispatch exact',True);compare(cache.key_cache[0],fullk,'original floating expanded K cache',True);compare(cache.value_cache[0],fullv,'original floating expanded V cache',True);check(returned is cache,'cache ownership')

def run(matrix):
    torch.set_num_threads(2)
    from benchmark.attention_format.validate import check_device as old_check
    from benchmark.attention_dense_format.validate import check_device as dense_check
    from benchmark.attention_dense_format.formal import read_matrix as dense_matrix
    from benchmark.attention_full_format.validate import check_device as full_check
    from benchmark.attention_full_format.formal import read_matrix as full_matrix
    from twilight.pyimpl.quant_attention.forward import validate_config
    originals={path.stem:load_config(path) for path in sorted((ROOT/'benchmark/attention_format/configs').glob('*.json'))};check(len(originals)==18,'18 original format configs required')
    _,dense=dense_matrix(ROOT/'benchmark/attention_dense_format/matrix.json');_,full=full_matrix(ROOT/'benchmark/attention_full_format/matrix.json');_,rs=read_matrix(matrix)
    reports={'original_format18':old_check('cpu',originals),'dense4':dense_check('cpu',dense),'full6':full_check('cpu',full)}
    for config in rs.values():
        validate_config(config['attention_quantization'],None,None,None,config['attention_rs']);rs_cache_check(config)
        for legacy in (next(iter(originals.values())),next(iter(dense.values())),next(iter(full.values()))):
            try:validate_config(config['attention_quantization'],legacy.get('attention_slice'),legacy.get('attention_dense_format'),legacy.get('attention_full_format'),config['attention_rs'])
            except ValueError:pass
            else:raise ValueError('RS/legacy intervention conflict accepted')
    from benchmark.attention_benchmark import parser,resolve,plan,execute
    selector=[]
    with patch('benchmark.attention_baseline.common.load_model',side_effect=AssertionError('selector loaded model')),patch.object(torch.cuda,'init',side_effect=AssertionError('selector initialized CUDA')):
        for vid,config in rs.items():
            s=config['attention_rs'];flags=['--encoding','rsint8','--profile','smoke','--output-root','/tmp/rs-selector-not-executed','--q-bits',str(s['q_bits']),'--k-bits',str(s['k_bits'])]
            if s['selection']['p'] is not None:flags+=['--top-p','on','--p',str(s['selection']['p'])]
            if s['mode']=='msb_topp':flags+=['--msb','on']
            args=parser().parse_args(flags);resolved=resolve(args);check(resolved['config']==config,'unified RS resolver exact config');result=plan(args);commands=result['execution_commands'];check(len(commands)==6 and not any(row['name']=='gate' for row in commands),'single RS smoke cannot mint cohort gate')
            scorer=[row['argv'] for row in commands if row['argv'][1:3]==['-m','benchmark.ppl']];check(len(scorer)==3 and all(argv[argv.index('--algo-config-path')+1]==resolved['config_path'] for argv in scorer),'single RS smoke must use exactly selected variant')
            args.profile='full';args.smoke_root='/tmp/rs-required-cohort-gate';fullplan=plan(args);check(len(fullplan['execution_commands'])==1 and '--rs-gate-report' in fullplan['execution_commands'][0]['argv'],'full selected RS requires full cohort gate')
            args.execute=True;args.review_report=None
            try:execute(args,fullplan)
            except ValueError as error:check('review-report' in str(error),'RS must reject missing review before execution')
            else:raise ValueError('RS executed without independent acceptance')
            selector.append(vid)
    from benchmark.attention_full_format.tests.test_selector import test_dispatch,test_invalid,test_plan_and_gate
    preserved_resolutions=test_dispatch();test_invalid();test_plan_and_gate()
    return dict(status='passed',scope='synthetic_CPU_only_no_pinned_model_runs',gpu_used=False,actual_model_runs=0,preserved_numerical=reports,preserved_numerical_jobs=28,rs_cache_variants=list(rs),rs_unified_selector_variants=selector,preserved_selector_resolutions=preserved_resolutions,source_sha256=code_provenance()['code_sha256'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:report=run(a.matrix)
    except Exception as error:write_json(a.output,dict(status='failed',reason=str(error)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
