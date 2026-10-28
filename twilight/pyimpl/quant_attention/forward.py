"""Llama common FP32 attention path for FP and source-aligned QK8."""
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from .intflash import SOURCE_REVISION, quantize_per_token
from .torch_backend import attention, causal_allowed, validate_mask
from ..trace import current_context

PARAMETERS={'group_axis':'head_token','quantizer_input_dtype':'float16','source_input_cast':'bf16_to_fp16','scale_dtype':'source','rounding':'torch_round','zero_group':'scale_one'}

def validate_config(config, slice_config=None):
    if set(config)-{'method','backend','scope','cache_storage','parameters','source_revision'}: raise ValueError('Unknown attention quantization fields')
    if config.get('method') not in ('fp_common','intflash_qk8'): raise ValueError('Unsupported quantization method')
    for key,value in {'backend':'torch_qdq','scope':'prefill_and_decode','cache_storage':'original_float'}.items():
        if config.get(key)!=value: raise ValueError('Unsupported '+key)
    expected=PARAMETERS if config['method']=='intflash_qk8' else {}
    if config.get('parameters')!=expected: raise ValueError('Unsupported quantization parameters')
    if config.get('source_revision')!=(SOURCE_REVISION if expected else None): raise ValueError('Unsupported source revision')
    if slice_config is not None:
        from .slice_backend import validate_slice_config
        validate_slice_config(slice_config, config['method'])

def common_forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None,**kwargs):
    if output_attentions: raise ValueError('output_attentions=True is unsupported')
    if kwargs: raise ValueError('Unsupported forward kwargs: '+str(sorted(kwargs)))
    ctx=current_context()
    if ctx is None: raise ValueError('Common attention requires an outer forward_context')
    if self.layer_id!=self.layer_idx: raise ValueError('Attention layer identity mismatch')
    b,m,_=hidden_states.shape
    if b!=1: raise ValueError('Only batch=1 without padding is supported')
    if past_key_value is not None and not isinstance(past_key_value,DynamicCache): raise ValueError('Only DynamicCache is supported')
    if use_cache and past_key_value is None: raise ValueError('Pass a new DynamicCache explicitly')
    previous=past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
    if previous and m>1: raise ValueError('Cached multi-query prefill is unsupported')
    if previous and not use_cache: raise ValueError('Reading a cache requires use_cache=True')
    q=self.q_proj(hidden_states).view(b,m,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(hidden_states).view(b,m,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,m,self.num_key_value_heads,self.head_dim).transpose(1,2)
    if q.shape[-1]!=k.shape[-1] or self.num_heads%self.num_key_value_heads: raise ValueError('Invalid GQA layout')
    if position_ids is None: raise ValueError('Explicit position_ids required')
    given=torch.as_tensor(ctx['position_ids'],device=q.device).reshape(-1)
    if not torch.equal(position_ids.reshape(-1),given): raise ValueError('Runner and model positions disagree')
    if given.numel()!=m: raise ValueError('Query position count mismatch')
    if cache_position is None or not torch.equal(cache_position.reshape(-1),given):
        raise ValueError('Actual cache_position must equal model query positions')
    if 'cache_position' in ctx and not torch.equal(torch.as_tensor(ctx['cache_position'],device=q.device).reshape(-1),cache_position.reshape(-1)):
        raise ValueError('Runner and actual cache_position disagree')
    if position_embeddings is None: cos,sin=self.rotary_emb(v,position_ids)
    else: cos,sin=position_embeddings
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    k=repeat_kv(k,self.num_key_value_groups); v=repeat_kv(v,self.num_key_value_groups)
    if use_cache:
        k,v=past_key_value.update(k,v,self.layer_idx,{'cos':cos,'sin':sin,'cache_position':cache_position})
    n=k.shape[-2]
    keys=torch.as_tensor(ctx['key_positions'],device=q.device).reshape(-1)
    if n!=ctx['expected_cache_length'] or keys.numel()!=n: raise ValueError('Cache length/position mismatch')
    if not torch.equal(keys,torch.arange(n,device=q.device)): raise ValueError('Only contiguous unpadded key positions supported')
    if use_cache and not torch.equal(given,torch.arange(previous,previous+m,device=q.device)): raise ValueError('Cache query positions mismatch')
    allowed=causal_allowed(given,keys,q.device)
    validate_mask(attention_mask,allowed)
    method=self.state.attention_quantization['method']
    def prepare_operand(tensor, name):
        if method != 'intflash_qk8':
            return None
        try:
            return quantize_per_token(tensor)
        except ValueError as error:
            raise ValueError(f'layer={self.layer_id}, operand={name}, stage={ctx["stage"]}, global_query_positions={ctx["global_query_positions"][0]}..{ctx["global_query_positions"][-1]}: {error}') from error
    qq=prepare_operand(q, 'q')
    kk=prepare_operand(k, 'k')
    recorder=ctx.get('recorder')
    if recorder is not None: recorder.attention_inputs(self,hidden_states,q,k,v,qq,kk,allowed)
    slice_config=getattr(self.state, "attention_slice", None)
    observer=None if recorder is None else lambda start,end,logits,probs,pv: recorder.attention_chunk(self.layer_id,start,end,logits,probs,pv,allowed)
    if slice_config is None:
        pv=attention(q if qq is None else qq.dq,k if kk is None else kk.dq,v,allowed,self.state.query_chunk,observer)
    else:
        from .slice_backend import attention_slice
        observer=None if recorder is None else lambda start,end,payload: recorder.attention_slice_chunk(self.layer_id,start,end,payload,allowed)
        pv=attention_slice(q,k,v,allowed,qq,kk,slice_config,self.state.query_chunk,observer,ctx.get('counters'))
    project_input=pv.to(hidden_states.dtype).transpose(1,2).reshape(b,m,self.hidden_size).contiguous()
    output=self.o_proj(project_input)
    if recorder is not None: recorder.attention_outputs(self.layer_id,project_input,output)
    counters=ctx.get('counters')
    if counters is not None:
        counters.record(self.layer_id,method,ctx['stage'],n)
        for operand, quant in [('q',qq),('k',kk)]:
            if quant is not None: counters.record_quantization(self.layer_id,operand,quant.diagnostics,ctx)
    return output,None,past_key_value
