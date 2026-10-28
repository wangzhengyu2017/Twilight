"""Observational trace collection; no forward results are replaced."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import math
import torch
from .quant_attention.intflash import (SOURCE_REVISION, SOURCE_DELTA, INPUT_CAST_POLICY,
                                       QUANTIZATION_COUNT_SEMANTICS)

_CONTEXT = ContextVar('twilight_forward_context', default=None)

def current_context():
    return _CONTEXT.get()

@contextmanager
def forward_context(context=None, **kwargs):
    ctx = dict(context or {}, **kwargs)
    required = {'run_id','sample_id','window_id','stage','global_query_positions','position_ids','key_positions','expected_cache_length'}
    if required - ctx.keys():
        raise ValueError('Missing forward context fields: ' + str(sorted(required - ctx.keys())))
    if ctx['stage'] not in ('prefill','decode'):
        raise ValueError('Unknown forward stage')
    token = _CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CONTEXT.reset(token)

class CallCounters:
    def __init__(self):
        self.counts = {}
        self.cache_lengths = []
        self.quantization_by_layer_operand = {}
        self.quantization_events = []
        self.quantization_event_limit = 32
        self.quantization_event_count = 0
        self.slice_statistics = {}
        self.slice_cost_totals = {}
        self.slice_labels = {}
    def record(self, layer_id, method, stage, cache_length):
        stage_counts = self.counts.setdefault(str(layer_id), {}).setdefault(method, {})
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        self.cache_lengths.append({'layer_id':layer_id,'method':method,'stage':stage,'cache_length':cache_length})
    def record_quantization(self, layer_id, operand, diagnostics, context=None):
        if operand not in ('q', 'k'): raise ValueError('Unknown quantized operand')
        counts = self.quantization_by_layer_operand.setdefault(str(layer_id), {}).setdefault(operand, {'calls':0})
        counts['calls'] += 1
        for name in ('input_elements','input_nonzero_elements','input_cast_nonzero_to_zero_count','groups','zero_group_count'):
            counts[name] = counts.get(name, 0) + diagnostics.get(name, 0)
        if diagnostics.get('input_cast_nonzero_to_zero_count', 0):
            self.quantization_event_count += 1
        if diagnostics.get('input_cast_nonzero_to_zero_count', 0) and len(self.quantization_events) < self.quantization_event_limit:
            context = context or {}
            positions = _list(context.get('global_query_positions', []))
            self.quantization_events.append({
                'layer_id':layer_id, 'operand':operand,
                **{key:context.get(key) for key in ('stage','sample_id','window_id')},
                'global_query_position_start':positions[0] if positions else None,
                'global_query_position_end':positions[-1] if positions else None,
                'shape':diagnostics.get('shape'), 'source_dtype':diagnostics.get('source_dtype'),
                'nonzero_to_zero_count':diagnostics['input_cast_nonzero_to_zero_count'],
                'examples':diagnostics.get('examples', [])[:8],
                'example_limit':8, 'examples_truncated':diagnostics.get('examples_truncated', False),
            })
    def record_slice_statistics(self, **data):
        """Accumulate backend work on device; synchronize only when serializing."""
        labels = {'selection_passes','selected_row_chunk','heads_parallel','actual_pv','variant_id','final_execution'}
        for key, value in data.items():
            if isinstance(value, dict):
                self.record_slice_statistics(**value)
                continue
            if key in labels or isinstance(value, (str, bool)) or value is None:
                old = self.slice_labels.setdefault(key, value)
                if old != value: raise ValueError('Inconsistent slice statistic label: ' + key)
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1: raise ValueError('Slice statistics must be scalar')
                value = value.detach()
            elif not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Invalid slice statistic: ' + key)
            target = self.slice_cost_totals if ('mac' in key or 'cost' in key or 'bytes' in key) else self.slice_statistics
            old = target.get(key)
            if old is None: target[key] = value
            elif key.endswith('_min'):
                target[key] = torch.minimum(old,value) if isinstance(old,torch.Tensor) else min(old,value)
            elif key.endswith('_max'):
                target[key] = torch.maximum(old,value) if isinstance(old,torch.Tensor) else max(old,value)
            else: target[key] = old + value
    def _slice_serialized(self, values):
        result = {}
        for key, value in values.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            if isinstance(value, (int,float)) and not math.isfinite(value):
                raise ValueError('Nonfinite slice statistic: ' + key)
            result[key] = value
        return dict(result, **self.slice_labels)
    def slice_summary(self):
        result = self._slice_serialized(self.slice_statistics)
        rows = result.get('rows', 0)
        if rows:
            result['mean_selected_keys'] = result.get('selected_count', 0) / rows
            if 'selected_count_squared' in result:
                result['selected_count_std'] = math.sqrt(max(0, result['selected_count_squared']/rows-result['mean_selected_keys']**2))
            result['selected_fraction'] = result.get('selected_count',0)/result['allowed_count'] if result.get('allowed_count') else None
        return result
    def slice_costs(self):
        return self._slice_serialized(self.slice_cost_totals)
    def as_dict(self):
        return {'layer_calls':self.counts, 'cache_lengths':self.cache_lengths,
                'attention_slice': {'statistics':self.slice_summary(), 'costs':self.slice_costs()},
                'quantization_diagnostics':{
                    'input_cast_policy':INPUT_CAST_POLICY, 'count_semantics':QUANTIZATION_COUNT_SEMANTICS,
                    'by_layer_operand':self.quantization_by_layer_operand,
                    'events':self.quantization_events, 'event_limit':self.quantization_event_limit,
                    'event_count':self.quantization_event_count,
                    'events_truncated':self.quantization_event_count > len(self.quantization_events)}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _list(x):
    return torch.as_tensor(x).reshape(-1).tolist()

def _jsonable(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, dict): return {k:_jsonable(v) for k,v in x.items() if k not in ('recorder','counters')}
    if isinstance(x, (tuple,list)): return [_jsonable(v) for v in x]
    return x

class TraceRecorder:
    def __init__(self, output_dir, targets, metadata=None, budget_bytes=256*1024*1024):
        self.output_dir = Path(output_dir)
        if isinstance(targets, (str,Path)):
            targets = json.loads(Path(targets).read_text())
        self.targets = targets['targets'] if isinstance(targets,dict) else targets
        self.metadata = metadata or {}
        self.budget_bytes = budget_bytes
        self.used_bytes = 0
        self.records = {}
        self.hooks = []
        identities = [tuple(t.get(k) for k in ('sample_id','window_id','layer_id','query_position','stage')) for t in self.targets]
        if len(set(identities)) != len(identities): raise ValueError('Duplicate trace targets')
    def matching(self, layer_id):
        ctx = current_context()
        if ctx is None or ctx.get('recorder') is not self: return []
        positions = _list(ctx['global_query_positions'])
        return [(i,positions.index(t['query_position']),t) for i,t in enumerate(self.targets)
                if t['layer_id']==layer_id and t['sample_id']==ctx['sample_id'] and t.get('window_id')==ctx['window_id']
                and t['stage']==ctx['stage'] and t['query_position'] in positions]
    def _copy(self, value):
        if value is None: return None
        out = value.detach().clone().cpu()
        self.used_bytes += out.numel()*out.element_size()
        if self.used_bytes > self.budget_bytes: raise RuntimeError('Trace tensor budget exceeded')
        return out
    def _put(self, idx, name, value):
        self.records.setdefault(idx, {'actual':{},'meta':{}})['actual'][name] = self._copy(value)
    def attach(self, model):
        layers = model.model.layers
        if any(t['layer_id'] >= len(layers) or t['layer_id'] < 0 for t in self.targets): raise ValueError('Trace target layer out of range')
        for layer_id, layer in enumerate(layers):
            def before(mod, args, kwargs, lid=layer_id):
                hidden = args[0] if args else kwargs['hidden_states']
                for idx,local,t in self.matching(lid):
                    if idx in self.records: raise RuntimeError('Trace target hit more than once')
                    self._put(idx,'layer_input',hidden[:,local:local+1])
            def after(mod, args, output, lid=layer_id):
                hidden = output[0] if isinstance(output,tuple) else output
                for idx,local,t in self.matching(lid): self._put(idx,'layer_output',hidden[:,local:local+1])
            self.hooks.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.hooks.append(layer.register_forward_hook(after))
        return self
    def close(self):
        for hook in self.hooks: hook.remove()
        self.hooks = []
    def attention_inputs(self, module, hidden, q, k, v, qq, kk, allowed):
        ctx = current_context()
        for idx,local,target in self.matching(module.layer_id):
            if 'q_post_rope' in self.records.setdefault(idx,{'actual':{},'meta':{}})['actual']: raise RuntimeError('Repeated attention target')
            visible = allowed[0,0,local]
            values = {'attn_input':hidden[:,local:local+1], 'q_post_rope':q[:,:,local:local+1],
                      'k_visible':k[:,:,visible], 'v_visible':v[:,:,visible],
                      'allowed_mask':allowed[:,:,local:local+1,visible],
                      'key_positions':torch.as_tensor(ctx['key_positions'],device=q.device).reshape(-1)[visible]}
            for prefix,quant in [('q',qq),('k',kk)]:
                values[prefix+'_codes'] = None if quant is None else (quant.codes[:,:,local:local+1] if prefix=='q' else quant.codes[:,:,visible])
                values[prefix+'_scales'] = None if quant is None else (quant.scales[:,:,local:local+1] if prefix=='q' else quant.scales[:,:,visible])
                values[prefix+'_dq'] = None if quant is None else (quant.dq[:,:,local:local+1] if prefix=='q' else quant.dq[:,:,visible])
            for name,value in values.items(): self._put(idx,name,value)
            ids = _list(ctx.get('input_ids',[]))
            self.records[idx]['meta'] = dict({**self.metadata, **_jsonable(ctx), **target},
                local_query_position=local, token_id=ids[local] if ids else None,
                predicted_target=target['query_position']+1, cache_length=k.shape[-2], Hq=module.num_heads,
                Hkv=module.num_key_value_heads,D=module.head_dim,
                head_to_kv=[h//module.num_key_value_groups for h in range(module.num_heads)],
                backend='torch_qdq',method=module.state.attention_quantization['method'],
                model_dtype=str(q.dtype),cache_storage='original_float',quantization_at='attention_read',
                scale_broadcast='[B,H,L,1]',zero_point=0,statistics_axis='D',rounding='torch_round_ties_to_even',
                source_revision=SOURCE_REVISION if qq else None,source_delta=SOURCE_DELTA if qq else [],
                quantizer_input_dtype='float16' if qq else None,source_input_cast='bf16_to_fp16' if qq else None,
                input_cast_policy=INPUT_CAST_POLICY if qq else None,
                quantization_count_semantics=QUANTIZATION_COUNT_SEMANTICS if qq else None,
                quantization_diagnostics_scope='full attention operands including expanded K cache reads',
                quantization_diagnostics={'q':qq.diagnostics,'k':kk.diagnostics} if qq else {})
    def attention_chunk(self, layer_id, start,end,logits,probs,pv,allowed):
        for idx,local,target in self.matching(layer_id):
            if start <= local < end:
                visible=allowed[0,0,local]
                row=local-start
                for name,value in [('logits_e',logits[:,:,row:row+1,visible]),('probs',probs[:,:,row:row+1,visible]),('pv_head',pv[:,:,row:row+1])]: self._put(idx,name,value)
    def attention_outputs(self,layer_id,project_input,output):
        for idx,local,target in self.matching(layer_id):
            self._put(idx,'o_proj_input',project_input[:,local:local+1])
            self._put(idx,'attn_output',output[:,local:local+1])
    def finalize(self):
        if len(self.records)!=len(self.targets): raise RuntimeError('Missing trace targets')
        self.output_dir.mkdir(parents=True,exist_ok=True)
        index={'schema_version':'attention-trace-v1','records':[],'tensor_bytes':self.used_bytes}
        required={'layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','logits_e','probs','pv_head','allowed_mask','key_positions'}
        for idx,record in sorted(self.records.items()):
            a=record['actual']; meta=record['meta']
            if required-set(a): raise RuntimeError('Missing actual fields: '+str(required-set(a)))
            rid=f"layer{meta['layer_id']:02d}_query{meta['query_position']}_{meta['stage']}"
            path=self.output_dir/rid; path.mkdir(exist_ok=False)
            torch.save(a,path/'actual.pt')
            if a['q_codes'] is not None:
                dot=torch.matmul(a['q_codes'].long(),a['k_codes'].long().transpose(-1,-2))
                # Integer products are accumulated independently by an explicit D loop.
                ref=torch.zeros_like(dot)
                for d in range(a['q_codes'].shape[-1]): ref += a['q_codes'][...,d].long().unsqueeze(-1)*a['k_codes'][...,d].long().unsqueeze(-2)
                if not torch.equal(dot,ref): raise RuntimeError('INT64 dot reference mismatch')
                logits=dot.float()*a['q_scales'].float()[...,None]*a['k_scales'].float()[:,:,None,:]/math.sqrt(meta['D'])
            else:
                dot=None; logits=a['q_post_rope'].float()@a['k_visible'].float().transpose(-1,-2)/math.sqrt(meta['D'])
            probs=logits.softmax(-1); pv=probs@a['v_visible'].float()
            torch.save({'int64_dot':dot,'logits_e':logits,'probs':probs,'pv_head':pv},path/'derived.pt')
            layouts = {'layer_input':'B,Q,Dmodel','layer_output':'B,Q,Dmodel','attn_input':'B,Q,Dmodel','o_proj_input':'B,Q,Hq*D','attn_output':'B,Q,Dmodel', 'logits_e':'B,Hq,Q,K','probs':'B,Hq,Q,K','allowed_mask':'B,1,Q,K','key_positions':'K','q_scales':'B,Hq,Q','k_scales':'B,Hq,K'}
            points = {'layer_input':'Decoder input before first norm','layer_output':'Decoder output after MLP and residual','attn_input':'Attention input after Decoder input norm', 'o_proj_input':'PV cast to model dtype then transpose/reshape before o_proj','attn_output':'o_proj output before residual','q_post_rope':'Q after exactly one RoPE','k_visible':'Original floating post-RoPE expanded K read from cache','v_visible':'Original floating expanded V read from cache','q_codes':'Actual quantizer rounded INT8 Q','k_codes':'Actual quantizer rounded INT8 K','q_scales':'Actual FP16 source Q scale','k_scales':'Actual FP16 source K scale','q_dq':'Actual FP32 Q DQ consumed by backend','k_dq':'Actual FP32 K DQ consumed by backend','logits_e':'Actual masked FP32 QK/sqrt(D) before softmax','probs':'Actual natural FP32 softmax probabilities','pv_head':'Actual FP32 probabilities times V before dtype cast','allowed_mask':'Actual positional causal domain','key_positions':'Visible key model positions'}
            meta['fields']={k:None if v is None else {'shape':list(v.shape),'dtype':str(v.dtype),'layout':layouts.get(k,'B,Hq,Q,D' if k.startswith('q_') or k=='pv_head' else 'B,Hq,K,D'),'capture_point':points[k]} for k,v in a.items()}
            meta['files']={f:sha256(path/f) for f in ('actual.pt','derived.pt')}
            (path/'meta.json').write_text(json.dumps(meta,indent=2))
            index['records'].append({'record_id':rid,'path':rid,'meta_sha256':sha256(path/'meta.json'),'actual_sha256':sha256(path/'actual.pt'),'derived_sha256':sha256(path/'derived.pt')})
        (self.output_dir/'index.json').write_text(json.dumps(index,indent=2))
        self.close()
        return index
