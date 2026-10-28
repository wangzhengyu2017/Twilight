"""Actual format-sidecar observations and separate CPU diagnostic references."""
import json
import math
from pathlib import Path
import torch
from .slice_trace import SliceTraceRecorder, _reference_selection
from .trace import current_context, sha256

SCHEMA = 'attention-format-trace-v1'


class FormatTraceRecorder(SliceTraceRecorder):
    def attention_inputs(self, module, hidden, q, k, v, qq, kk, allowed):
        super().attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
        for idx,local,target in self.matching(module.layer_id):
            meta=self.records[idx]['meta']; config=meta['attention_slice']
            meta.update(schema_version=SCHEMA,schema=SCHEMA,set_role='predicted_k',
                        final_operand_origin='original_intflash_codes',
                        predictor_source='original_intflash_codes' if config['predictor_format']=='sm8' else 'original_intflash_dq',
                        predictor_requantized=config['predictor_format']!='sm8',
                        prefix_rounding='sign_magnitude_truncate' if config['predictor_format']=='sm8' else 'signed_floor_divide_or_native_E2M1',
                        source_quantization_unchanged=True,selection_passes=1,
                        actual_matmul='fp32_integer_prefix_ordered_scales' if config['predictor_format']=='sm8' else 'fp32_reconstructed_contiguous_operands_then_divide_sqrt_D_f32',
                        format_codec_sha256=sha256(Path(__file__).parent/'quant_attention'/'predictor_formats.py'),
                        format_backend_sha256=sha256(Path(__file__).parent/'quant_attention'/'format_backend.py'),
                        format_metadata_layout='B,Hq,Q_or_Kvisible,D/32',
                        format_data_layout='B,Hq,Q_or_Kvisible,D',
                        logical_bits_per_element={'q':config['q_prefix_bits']+(0 if config['predictor_format']=='sm8' else .25),'k':config['k_prefix_bits']+(0 if config['predictor_format']=='sm8' else .25)},
                        prefix_bits_include_sign=True,metadata_bits_per_block=0 if config['predictor_format']=='sm8' else 8,
                        mxint8_endpoint_policy='symmetric_omit_min_negative',mxint8_implicit_element_scale=2**-6,
                        tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                        float32_matmul_precision=torch.get_float32_matmul_precision(),
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    def attention_slice_chunk(self,layer_id,start,end,payload,allowed):
        super().attention_slice_chunk(layer_id,start,end,payload,allowed)
        for idx,local,target in self.matching(layer_id):
            if not start <= local < end: continue
            row=local-start; visible=allowed[0,0,local]
            for name,value in payload.items():
                if '_format_' not in name: continue
                value=value[:,:,row:row+1] if name.startswith('q_') else value[:,:,visible]
                self._put(idx,name,value)
            self._put(idx,'sqrt_D_f32',payload['sqrt_D_f32'])

    def _derived(self,a,meta):
        result={}
        references=[('int',a['q_codes'].float()*a['q_scales'].float()[...,None],a['k_codes'].float()*a['k_scales'].float()[...,None])]
        if meta['predictor_format']!='sm8':
            decoded=[]
            for prefix in ('q','k'):
                codes=a[prefix+'_format_full_codes']
                exponent=a[prefix+'_format_exponent']
                shape=codes.shape
                fp4=meta['predictor_format']=='mxfp4' or (meta['predictor_format']=='mxint8q_mxfp4k' and prefix=='k')
                if fp4:
                    table=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float32)
                    raw=table[codes.long() & 7]*torch.where(codes>=8,-1.,1.)
                    value=torch.ldexp(raw.reshape(*shape[:-1],-1,32),exponent[...,None])
                else:
                    value=torch.ldexp(codes.float().reshape(*shape[:-1],-1,32),exponent[...,None]-6)
                decoded.append(value.reshape(shape))
            references.append(('mx_full',*decoded))
        for label,q,k in references:
            logits=q@k.transpose(-1,-2)/torch.tensor(math.sqrt(meta['D']),dtype=torch.float32)
            probs,mask,indices,counts,valid=_reference_selection(logits,meta['p'])
            keyindices=a['key_positions'][indices.clamp_min(0)].masked_fill(~valid,-1)
            prefix='reference_'+label+'_'
            result.update({prefix+'logits_same_input':logits,prefix+'probs_same_input':probs,prefix+'top_p_mask_same_input':mask,prefix+'top_p_indices_same_input':keyindices,prefix+'top_p_counts_same_input':counts,prefix+'top_p_valid_mask_same_input':valid})
        return result

    @staticmethod
    def _validate(a,meta):
        SliceTraceRecorder._validate(a,meta)
        for prefix in ('q','k'):
            for field in ('full_codes','prefix_bytes','hi','full_dq','predictor_dq'):
                if prefix+'_format_'+field not in a: raise ValueError('Missing actual format observation')
            if meta['predictor_format']!='sm8':
                for field in ('exponent','e8m0','clamp_mask','zero_blocks','underflow_mask'):
                    if prefix+'_format_'+field not in a: raise ValueError('Missing actual MX metadata')

    def finalize(self):
        if len(self.records) != len(self.targets): raise RuntimeError('Missing trace targets')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = {'schema_version':SCHEMA, 'schema':SCHEMA, 'records':[], 'tensor_bytes':self.used_bytes}
        sets = []
        for idx, record in sorted(self.records.items()):
            a, meta = record['actual'], record['meta']
            self._validate(a, meta)
            rid = f"layer{meta['layer_id']:02d}_query{meta['query_position']}_{meta['stage']}"
            path = self.output_dir/rid
            path.mkdir(exist_ok=False)
            derived = self._derived(a, meta)
            self.used_bytes += sum(v.numel()*v.element_size() for v in derived.values())
            if self.used_bytes > self.budget_bytes: raise RuntimeError('Trace tensor budget exceeded; keys are never truncated')
            torch.save(a, path/'actual.pt')
            torch.save(derived, path/'derived.pt')
            def describe(name, value, diagnostic=False):
                if value is None: return None
                if name.endswith(('_format_exponent','_format_e8m0','_format_zero_blocks')): layout='B,Hq,Q,D/32' if name.startswith('q') else 'B,Hq,Kvisible,D/32'
                elif name == 'sqrt_D_f32': layout='scalar'
                elif name == 'key_positions': layout='Kvisible'
                elif name in ('selected_counts',) or name.endswith('counts_same_input'): layout='B,Hq,Q'
                elif name in ('selected_indices','valid_mask') or name.endswith(('indices_same_input','valid_mask_same_input')): layout='B,Hq,Q,Mmax'
                elif name in ('q_scales','k_scales'): layout='B,Hq,Q' if name.startswith('q') else 'B,Hq,Kvisible'
                elif name == 'o_proj_input': layout='B,Q,Hq*D'
                elif name in ('layer_input','layer_output','attn_input','attn_output'): layout='B,Q,Dmodel'
                elif name == 'allowed_mask': layout='B,1,Q,Kvisible'
                elif name.startswith(('k_', 'v_')): layout='B,Hq,Kvisible,D'
                elif name.startswith('q_') or name == 'pv_head': layout='B,Hq,Q,D'
                else: layout='B,Hq,Q,Kvisible'
                points={'layer_input':'Decoder input before first norm','layer_output':'Decoder output after MLP and residual',
                        'attn_input':'Attention input after Decoder input norm','o_proj_input':'Actual PV after model dtype cast and head reshape before o_proj',
                        'attn_output':'Actual o_proj output before residual','pv_head':'Actual final probabilities times original V before dtype cast',
                        'q_post_rope':'Actual Q after exactly one RoPE','k_visible':'Actual original floating post-RoPE K cache read after GQA expansion',
                        'v_visible':'Actual original floating V cache read after GQA expansion'}
                point = ('CPU diagnostic from captured source/format sidecar; never used by model' if diagnostic else points.get(name, 'Actual '+name+' at production attention observation point'))
                return {'shape':list(value.shape),'dtype':str(value.dtype),'layout':layout,'capture_point':point}
            meta['fields'] = {name:describe(name,value) for name,value in a.items()}
            meta['derived_fields'] = {name:describe(name,value,True) for name,value in derived.items()}
            meta['derived_scope'] = 'targets_only_cpu_diagnostic_not_production_work'
            meta['files'] = {f:sha256(path/f) for f in ('actual.pt','derived.pt')}
            (path/'meta.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
            hashes = dict(meta_sha256=sha256(path/'meta.json'),actual_sha256=meta['files']['actual.pt'],derived_sha256=meta['files']['derived.pt'])
            index['records'].append({'record_id':rid,'path':rid,**hashes})
            for head in range(meta['Hq']):
                indices = a['selected_indices'][0,head,0]
                valid = a['valid_mask'][0,head,0]
                sets.append({**{k:v for k,v in meta.items() if k not in ('fields','derived_fields','files','input_ids','global_query_positions','position_ids','cache_position')},
                             'record_id':rid,'head_id':head,'head':head,'head_to_kv':meta['head_to_kv'],
                             'key_positions':a['key_positions'].tolist(), 'selected_key_positions':indices[valid].tolist(),
                             'num_selected':int(a['selected_counts'][0,head,0]),
                             'selected_mask_ref':{'file':str((path/'actual.pt').resolve()),'field':'selected_mask','index':[0,head,0],'sha256':hashes['actual_sha256']},
                             'selected_mask_field':'selected_mask','selected_mask_file':str((path/'actual.pt').resolve()), **hashes})
        index['tensor_bytes'] = self.used_bytes
        sets_path = self.output_dir.parent/'selection_sets.jsonl'
        sets_path.write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in sets))
        index['selection_sets'] = {'path':str(sets_path.resolve()),'sha256':sha256(sets_path),'rows':len(sets)}
        (self.output_dir/'index.json').write_text(json.dumps(index,indent=2))
        self.close()
        return index
