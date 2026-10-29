"""Synthetic CPU recorder integration only; never presented as real study traces."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from benchmark.attention_baseline.common import write_json
from benchmark.attention_full_format.formal import read_matrix
from benchmark.attention_dense_format.oracle import compare,scalar_decode
from twilight.pyimpl.trace import forward_context
from twilight.pyimpl.full_format_trace import FullFormatTraceRecorder
from twilight.pyimpl.quant_attention.full_format_backend import attention_full_format
from twilight.pyimpl.quant_attention.intflash import quantize_per_token
from twilight.pyimpl.quant_attention.torch_backend import causal_allowed


def main():
    _,configs=read_matrix('benchmark/attention_full_format/matrix.json');torch.manual_seed(67);tested=[]
    for vid in ('sm8_q8k8_topp_p095','mxint8_q8k8_topp_p085'):
        cfg=configs[vid];s=cfg['attention_full_format'];q=torch.randn(1,4,7,32,dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);qq,kk=quantize_per_token(q),quantize_per_token(k)
        allowed=causal_allowed(torch.arange(7),torch.arange(7));hidden=torch.randn(1,7,128,dtype=torch.bfloat16);module=SimpleNamespace(layer_id=0,num_heads=4,num_key_value_heads=2,num_key_value_groups=2,head_dim=32,state=SimpleNamespace(attention_quantization=cfg['attention_quantization'],attention_full_format=s))
        reference=attention_full_format(q,k,v,allowed,qq,kk,s,3)
        with tempfile.TemporaryDirectory(prefix='synthetic-full-format-trace-') as temporary:
            root=Path(temporary);target=dict(sample_id=0,window_id=None,layer_id=0,query_position=3,stage='prefill');recorder=FullFormatTraceRecorder(root/'trace',[target],metadata=dict(test_scope='synthetic_CPU_only_not_study_evidence'))
            with forward_context(run_id='synthetic-only',sample_id=0,window_id=None,stage='prefill',global_query_positions=list(range(7)),position_ids=list(range(7)),key_positions=list(range(7)),expected_cache_length=7,cache_position=list(range(7)),input_ids=list(range(7)),recorder=recorder):
                recorder._put(0,'layer_input',hidden[:,3:4]);recorder.attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
                captured=attention_full_format(q,k,v,allowed,qq,kk,s,3,lambda start,end,payload:recorder.attention_full_format_chunk(0,start,end,payload,allowed))
                projected=captured.to(torch.bfloat16).transpose(1,2).reshape(1,7,128);recorder.attention_outputs(0,projected,projected);recorder._put(0,'layer_output',projected[:,3:4])
            compare(captured,reference,'synthetic recorder invisibility',True);index=recorder.finalize();entry=index['records'][0];path=root/'trace'/entry['path'];actual=torch.load(path/'actual.pt',weights_only=True);meta=json.loads((path/'meta.json').read_text())
            assert meta['selection_passes']==1 and meta['selected_mask_unused_in_forward'] is False and meta['final_operand_profile']=='decoded_full_format'
            assert actual['key_positions'].tolist()==[0,1,2,3] and actual['k_format_full_codes'].shape==(1,4,4,32)
            decoded=[]
            for prefix in ('q','k'):
                operand=scalar_decode(actual[prefix+'_format_full_codes'],s['format'],scales=actual[prefix+'_scales'],e8m0=actual.get(prefix+'_format_e8m0'));compare(actual[prefix+'_format_full_dq'],operand,'captured scalar byte decode',True);decoded.append(operand)
            compare(actual['full_format_logits'],decoded[0]@decoded[1].transpose(-1,-2)/math.sqrt(32),'captured full scores')
            compare(actual['final_selected_logits'],actual['full_format_logits'].masked_fill(~actual['selected_mask'],-torch.inf),'captured retained scores',True)
            compare(actual['pv_head'],actual['final_selected_probs']@actual['v_visible'].float(),'captured actual PV')
            listing=[json.loads(row) for row in (root/'selection_sets.jsonl').read_text().splitlines()];assert len(listing)==4 and all(row['unused_in_forward'] is False for row in listing)
        tested.append(vid)
    report=dict(status='passed',scope='synthetic CPU recorder integration; no real study capture',variants=tested,trace_invisibility_exact=True,actual_codes_scores_masks_PV_checked=True)
    write_json('/output/project/experiments/stp-dense-formats/preparation/full-topp-trace-cpu-validation.json',report);print(report)
if __name__=='__main__':main()
