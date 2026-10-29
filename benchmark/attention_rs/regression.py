"""CPU regressions for reviewed chunk-padding and historical-source blockers."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from benchmark.attention_baseline.common import load_json,write_json,sha256_file
from .formal import read_matrix
from .historical import validate_source_final
from .oracle import check
from .replay import replay_record

def reject(call,label):
    try:call()
    except (ValueError,RuntimeError,TypeError):return
    raise ValueError('invalid evidence accepted: '+label)

def padding_case(config):
    from twilight.pyimpl.quant_attention.rs_backend import attention_rs
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.rs_trace import RSTraceRecorder
    from twilight.pyimpl.trace import forward_context
    generator=torch.Generator().manual_seed(847);q=torch.zeros((1,32,2,128),dtype=torch.bfloat16);q[:,:,1]=3
    k=torch.randn((1,8,128,128),generator=generator).to(torch.bfloat16).repeat_interleave(4,1);k[:,:,0]=3
    v=torch.randn((1,8,128,128),generator=generator).to(torch.bfloat16).repeat_interleave(4,1)
    qq,kk=quantize_per_token(q),quantize_per_token(k);hidden=torch.zeros((1,2,4096),dtype=torch.bfloat16)
    allowed=(torch.arange(128)[None,:]<=torch.tensor([126,127])[:,None]).reshape(1,1,2,128)
    target=dict(sample_id=0,window_id=None,layer_id=0,query_position=127,stage='prefill')
    module=SimpleNamespace(layer_id=0,num_heads=32,num_key_value_heads=8,num_key_value_groups=4,head_dim=128,state=SimpleNamespace(attention_rs=config['attention_rs'],attention_quantization=config['attention_quantization']))
    with tempfile.TemporaryDirectory(prefix='rs-padding-',dir='/tmp') as temporary:
        root=Path(temporary);rec=RSTraceRecorder(root/'trace',[target],metadata=dict(profile='smoke',mode='cached_decode',protocol_id='synthetic_CPU_padding_regression'))
        with forward_context(run_id='padding',sample_id=0,window_id=None,stage='prefill',global_query_positions=[126,127],position_ids=[126,127],cache_position=[126,127],key_positions=list(range(128)),expected_cache_length=128,input_ids=[1,2],recorder=rec):
            rec._put(0,'layer_input',hidden[:,1:]);rec.attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
            pv=attention_rs(q,k,v,allowed,qq,kk,config['attention_rs'],64,lambda start,end,payload:rec.attention_rs_chunk(0,start,end,payload,allowed))
            rec.attention_outputs(0,pv.to(torch.bfloat16).transpose(1,2).reshape(1,2,4096),hidden);rec._put(0,'layer_output',hidden[:,1:])
        entry=rec.finalize()['records'][0];path=root/'trace'/entry['path'];meta=load_json(path/'meta.json');a=torch.load(path/'actual.pt',weights_only=True);d=torch.load(path/'derived.pt',weights_only=True)
        width=a['selected_indices'].shape[-1];count=int(a['selected_counts'].max());check(width>count and count==1,'fixture must preserve wider original chunk padding')
        check(a['selected_source_logits'].shape[-1]==width and torch.isneginf(a['selected_source_logits'][~a['selected_valid_mask']]).all(),'all invalid source-logit slots observed')
        report=replay_record(entry,meta,a,d,config,'cpu');check(report['passed'],'valid wider padding must replay')
        negative=[]
        # Exact invalid-slot contracts are checked over the full stored width.
        for field in ('selected_source_logits','selected_indices','selected_source_k_codes'):
            damaged=dict(a);damaged[field]=a[field].clone();invalid=~a['selected_valid_mask']
            if field=='selected_source_logits':damaged[field][invalid]=0.
            elif field=='selected_indices':damaged[field][invalid]=0
            else:damaged[field][invalid]=0
            reject(lambda:replay_record(entry,meta,damaged,d,config,'cpu'),'corrupt padded '+field);negative.append(field)
        return dict(variant_id=config['attention_rs']['variant_id'],passed=True,original_query_rows=2,captured_max_count=count,stored_chunk_width=width,invalid_slots_verified=True,corrupt_padding_rejected=negative)

def run(matrix,controls_manifest):
    torch.set_num_threads(2);_,configs=read_matrix(matrix);cases=[padding_case(config) for config in configs.values() if config['attention_rs']['mode']=='msb_topp'];controls=load_json(controls_manifest);aliases=[]
    for candidate,mapping in controls['top_p'].items():
        for role,name in mapping.items():
            root=Path(name);index=load_json(root/'trace/index.json')
            for entry in index['records']:
                meta=load_json(root/'trace'/entry['path']/'meta.json');binding=validate_source_final(meta,role)
                for invalid in ('rs_full_decoded','native_prefix','unknown'):
                    wrong=copy.deepcopy(meta);wrong['final_operand_origin']=invalid;reject(lambda:validate_source_final(wrong,role),'historical '+role+' origin '+invalid)
                aliases.append(dict(candidate=candidate,record_id=entry['record_id'],**binding))
    check(len(aliases)==72,'six records in each of twelve historical groups required')
    return dict(status='passed',scope='synthetic_CPU_regressions_and_read_only_old_metadata',gpu_used=False,actual_model_runs=0,padding_cases=cases,historical_records=len(aliases),historical_aliases=aliases,atol=1e-5,rtol=1e-4)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','controls-manifest','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:report=run(a.matrix,a.controls_manifest)
    except Exception as exc:write_json(a.output,dict(status='failed',reason=str(exc)));raise
    write_json(a.output,report)
if __name__=='__main__':main()
