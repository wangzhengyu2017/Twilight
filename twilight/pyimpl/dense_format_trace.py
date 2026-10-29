"""Authentic dense observations; all TopP sets are offline diagnostics only."""
import json
import math
import torch
from .trace import TraceRecorder,current_context,sha256
from .slice_trace import _reference_selection

SCHEMA='attention-dense-format-trace-v1'
THRESHOLDS={'095':.95,'090':.9,'085':.85}


def diagnostic_sets(logits,positions):
    result={}
    for label,p in THRESHOLDS.items():
        probs,mask,indices,counts,valid=_reference_selection(logits,p)
        prefix='diagnostic_p'+label+'_'
        result.update({prefix+'mask':mask,prefix+'indices':positions[indices.clamp_min(0)].masked_fill(~valid,-1),prefix+'counts':counts,prefix+'valid_mask':valid})
    return result


class DenseFormatTraceRecorder(TraceRecorder):
    def attention_inputs(self,module,hidden,q,k,v,qq,kk,allowed):
        super().attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
        config=module.state.attention_dense_format
        for idx,local,target in self.matching(module.layer_id):
            self.records[idx]['meta'].update(schema_version=SCHEMA,attention_dense_format=config,
                variant_id=config['variant_id'],selection_mode='dense',selection_passes=0,
                final_operand_origin='stored_format_bytes',set_role='actual_all_legal_execution',
                diagnostic_sets=dict(unused_in_forward=True,thresholds=list(THRESHOLDS.values()),semantics='stable_key_position_minimal_crossing'),
                actual_matmul='common_fp32_byte_decoded_operands_divide_sqrt_D',
                cache_gqa='original floating K/V expanded once before cache update',
                format_storage_policy='nibble_in_uint8_container; E8M0 metadata separate from original INT8 source scale',
                negative_zero_policy='decode_to_canonical_positive_zero',
                tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    def attention_dense_chunk(self,layer_id,start,end,payload,allowed):
        for idx,local,target in self.matching(layer_id):
            if not start<=local<end:continue
            if 'execution_mask' in self.records[idx]['actual']:raise RuntimeError('Repeated dense target')
            row=local-start;visible=allowed[0,0,local]
            keys=torch.as_tensor(current_context()['key_positions'],device=visible.device).reshape(-1)
            for name,value in payload.items():
                if name.startswith('q_format_'):value=value[:,:,row:row+1]
                elif name.startswith('k_format_'):value=value[:,:,visible]
                elif name in ('logits_e','probs','execution_mask'):value=value[:,:,row:row+1,visible]
                elif name=='pv_head':value=value[:,:,row:row+1]
                elif name in ('execution_indices','execution_valid_mask','execution_counts'):
                    value=value[:,:,row:row+1]
                    if name=='execution_indices':value=keys[value.clamp_min(0)].masked_fill(value<0,-1)
                else:raise ValueError('Unknown dense observation '+name)
                self._put(idx,name,value)

    def finalize(self):
        if len(self.records)!=len(self.targets):raise RuntimeError('Missing dense trace target')
        self.output_dir.mkdir(parents=True,exist_ok=False)
        index=dict(schema_version=SCHEMA,records=[],diagnostic_sets_unused_in_forward=True)
        sets=[]
        required={'layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','allowed_mask','key_positions','pv_head','logits_e','probs','execution_mask','execution_indices','execution_counts','execution_valid_mask','q_format_full_codes','k_format_full_codes','q_format_full_dq','k_format_full_dq'}
        for idx,record in sorted(self.records.items()):
            a,meta=record['actual'],record['meta'];missing=required-set(a)
            if missing:raise ValueError('Missing dense observation '+str(missing))
            domain=a['allowed_mask'].expand_as(a['execution_mask'])
            if not torch.equal(a['execution_mask'],domain):raise ValueError('Dense execution excluded legal keys')
            if not torch.equal(a['execution_counts'],domain.sum(-1)):raise ValueError('Dense execution count mismatch')
            for name,value in a.items():
                if value is not None and value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite dense observation '+name)
            derived=diagnostic_sets(a['logits_e'],a['key_positions'])
            for label,q,k in [('fp',a['q_post_rope'].float(),a['k_visible'].float()),('source_int',a['q_codes'].float()*a['q_scales'].float()[...,None],a['k_codes'].float()*a['k_scales'].float()[...,None])]:
                logits=(q@k.transpose(-1,-2))/math.sqrt(meta['D'])
                derived['reference_'+label+'_logits_same_input']=logits
                derived['reference_'+label+'_probs_same_input']=logits.softmax(-1)
                derived.update({'reference_'+label+'_'+name+'_same_input':value for name,value in diagnostic_sets(logits,a['key_positions']).items()})
            self.used_bytes+=sum(value.numel()*value.element_size() for value in derived.values())
            if self.used_bytes>self.budget_bytes:raise RuntimeError('Dense trace budget exceeded')
            rid=f"layer{meta['layer_id']:02d}_query{meta['query_position']}_{meta['stage']}";path=self.output_dir/rid;path.mkdir()
            torch.save(a,path/'actual.pt');torch.save(derived,path/'derived.pt')
            storage={}
            for prefix,bits in [('q',meta['attention_dense_format']['q_bits']),('k',meta['attention_dense_format']['k_bits'])]:
                codes=a[prefix+'_format_full_codes'];metadata=a.get(prefix+'_format_e8m0')
                storage[prefix]=dict(logical_payload_bits=codes.numel()*bits,logical_format_metadata_bits=0 if metadata is None else metadata.numel()*8,
                    actual_code_tensor_bytes=codes.numel()*codes.element_size(),actual_format_metadata_tensor_bytes=0 if metadata is None else metadata.numel()*metadata.element_size(),
                    source_scale_tensor_bytes=a[prefix+'_scales'].numel()*a[prefix+'_scales'].element_size(),source_scale_is_separate=True,
                    actual_sidecar_tensor_bytes=sum(v.numel()*v.element_size() for name,v in a.items() if name.startswith(prefix+'_format_')))
            meta.update(storage=storage,fields={name:dict(shape=list(v.shape),dtype=str(v.dtype),capture_point='actual production attention boundary') for name,v in a.items() if v is not None},
                derived_fields={name:dict(shape=list(v.shape),dtype=str(v.dtype),unused_in_forward=True) for name,v in derived.items()},
                files={name:sha256(path/name) for name in ('actual.pt','derived.pt')})
            (path/'meta.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
            hashes=dict(meta_sha256=sha256(path/'meta.json'),actual_sha256=meta['files']['actual.pt'],derived_sha256=meta['files']['derived.pt'])
            index['records'].append(dict(record_id=rid,path=rid,**hashes))
            for label,p in THRESHOLDS.items():
                for head in range(meta['Hq']):
                    prefix='diagnostic_p'+label+'_';valid=derived[prefix+'valid_mask'][0,head,0]
                    sets.append(dict(record_id=rid,variant_id=meta['variant_id'],layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],head_id=head,p=p,unused_in_forward=True,set_role='diagnostic_dense_important_keys',selected_key_positions=derived[prefix+'indices'][0,head,0][valid].tolist(),num_selected=int(derived[prefix+'counts'][0,head,0]),allowed_count=int(domain[0,head,0].sum()),actual_execution_count=int(a['execution_counts'][0,head,0]),mask_ref=dict(file=str((path/'derived.pt').resolve()),field=prefix+'mask',index=[0,head,0],sha256=hashes['derived_sha256']),**hashes))
        sets_path=self.output_dir.parent/'diagnostic_sets.jsonl';sets_path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in sets))
        index.update(tensor_bytes=self.used_bytes,diagnostic_sets=dict(path=str(sets_path.resolve()),sha256=sha256(sets_path),rows=len(sets),unused_in_forward=True))
        (self.output_dir/'index.json').write_text(json.dumps(index,indent=2)+'\n');self.close();return index
