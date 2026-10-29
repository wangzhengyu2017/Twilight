"""Synthetic CPU trace plumbing checks; never counts as actual model captures."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from benchmark.attention_baseline.common import write_json,sha256_file
from .formal import read_matrix
from .replay import replay_record,load_trace_records
from .oracle import check

def trace_check(matrix):
    from twilight.pyimpl.quant_attention.rs_backend import attention_rs
    from twilight.pyimpl.quant_attention.intflash import quantize_per_token
    from twilight.pyimpl.rs_trace import RSTraceRecorder,RSCallCounters
    from twilight.pyimpl.trace import forward_context
    torch.set_num_threads(2);_,configs=read_matrix(matrix);reports=[];generator=torch.Generator().manual_seed(47)
    targets=[dict(sample_id=0,window_id=None,layer_id=layer,query_position=query,stage=stage) for layer in (0,16,31) for query,stage in ((127,'prefill'),(128,'decode'))]
    with tempfile.TemporaryDirectory(prefix='rs-synthetic-',dir='/tmp') as temporary:
        for vid,cfg in configs.items():
            root=Path(temporary)/vid;root.mkdir();recorder=RSTraceRecorder(root/'trace',targets,metadata=dict(profile='smoke',mode='cached_decode',protocol_id='synthetic_cpu_only'));counter=RSCallCounters()
            for target in targets:
                layer,query,stage=target['layer_id'],target['query_position'],target['stage'];n=query+1
                q=torch.randn((1,32,1,128),generator=generator).to(torch.bfloat16);k=torch.randn((1,8,n,128),generator=generator).to(torch.bfloat16).repeat_interleave(4,1);v=torch.randn((1,8,n,128),generator=generator).to(torch.bfloat16).repeat_interleave(4,1)
                hidden=torch.randn((1,1,4096),generator=generator).to(torch.bfloat16);qq,kk=quantize_per_token(q),quantize_per_token(k);allowed=torch.ones((1,1,1,n),dtype=torch.bool)
                module=SimpleNamespace(layer_id=layer,num_heads=32,num_key_value_heads=8,num_key_value_groups=4,head_dim=128,state=SimpleNamespace(attention_rs=cfg['attention_rs'],attention_quantization=cfg['attention_quantization']))
                with forward_context(run_id=vid,sample_id=0,window_id=None,stage=stage,global_query_positions=[query],position_ids=[query],cache_position=[query],key_positions=list(range(n)),expected_cache_length=n,input_ids=[1],recorder=recorder,counters=counter):
                    idx=next(i for i,_,_ in recorder.matching(layer));recorder._put(idx,'layer_input',hidden);recorder.attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
                    actual=attention_rs(q,k,v,allowed,qq,kk,cfg['attention_rs'],64,lambda start,end,payload:recorder.attention_rs_chunk(layer,start,end,payload,allowed),counter,layer)
                    project=actual.to(torch.bfloat16).transpose(1,2).reshape(1,1,4096);recorder.attention_outputs(layer,project,hidden);recorder._put(idx,'layer_output',hidden)
            index=recorder.finalize();check(len(index['records'])==6 and index['selection_sets']['rows']==192 and index['diagnostic_sets']['rows']==576,'synthetic trace coverage')
            records=load_trace_records(root);result=[]
            for entry,meta,a,d in records:
                replay=replay_record(entry,meta,a,d,cfg,'cpu',model=None);check(replay['passed'],'synthetic independent replay');result.append(dict(record_id=entry['record_id'],passed=True,mode=cfg['attention_rs']['mode']))
            reports.append(dict(variant_id=vid,passed=True,records=result))
    return dict(status='passed',scope='synthetic_CPU_trace_plumbing_only_not_actual_model_captures',actual_model_records=0,gpu_used=False,variants=reports)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--matrix',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try:report=trace_check(a.matrix)
    except Exception as exc:write_json(a.output,dict(status='failed',scope='synthetic_CPU_only',reason=str(exc)));raise
    write_json(a.output,report);print(json.dumps(dict(status=report['status'],actual_model_records=0)))
if __name__=='__main__':main()
