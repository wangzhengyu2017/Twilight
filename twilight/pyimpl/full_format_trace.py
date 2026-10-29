"""Real full-format TopP captures: masks saved here are used in forward."""
import json
import math
import torch
from .trace import TraceRecorder,current_context,sha256
from .dense_format_trace import diagnostic_sets

SCHEMA='attention-full-format-trace-v1'


class FullFormatTraceRecorder(TraceRecorder):
    def attention_inputs(self,module,hidden,q,k,v,qq,kk,allowed):
        super().attention_inputs(module,hidden,q,k,v,qq,kk,allowed)
        config=module.state.attention_full_format
        for idx,local,target in self.matching(module.layer_id):
            self.records[idx]['meta'].update(schema_version=SCHEMA,attention_full_format=config,
                variant_id=config['variant_id'],format=config['format'],p=config['selection']['p'],
                selection_mode='top_p',selection_passes=1,final_operand_profile='decoded_full_format',
                final_operand_origin='stored_format_bytes',set_role='actual_forward_selected_keys',
                selected_mask_unused_in_forward=False,tie_break='key_position_ascending',
                actual_matmul='common_fp32_byte_decoded_operands_divide_sqrt_D',
                actual_pv='dense_zero_masked',cache_gqa='original floating K/V expanded once before cache update',
                negative_zero_policy='decode_to_canonical_positive_zero',tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())

    def attention_full_format_chunk(self,layer_id,start,end,payload,allowed):
        for idx,local,target in self.matching(layer_id):
            if not start<=local<end:continue
            if 'selected_mask' in self.records[idx]['actual']:raise RuntimeError('Repeated full-format target')
            row=local-start;visible=allowed[0,0,local];keys=torch.as_tensor(current_context()['key_positions'],device=visible.device).reshape(-1)
            for name,value in payload.items():
                if name.startswith('q_format_'):value=value[:,:,row:row+1]
                elif name.startswith('k_format_'):value=value[:,:,visible]
                elif name in ('full_format_logits','full_format_probs','selected_mask','final_selected_logits','final_selected_probs'):value=value[:,:,row:row+1,visible]
                elif name=='pv_head':value=value[:,:,row:row+1]
                elif name in ('selected_indices','selected_counts','selected_valid_mask'):
                    value=value[:,:,row:row+1]
                    if name=='selected_indices':value=keys[value.clamp_min(0)].masked_fill(value<0,-1)
                else:raise ValueError('Unknown full-format observation '+name)
                self._put(idx,name,value)

    def finalize(self):
        if len(self.records)!=len(self.targets):raise RuntimeError('Missing full-format target')
        self.output_dir.mkdir(parents=True,exist_ok=False);index=dict(schema_version=SCHEMA,records=[]);sets=[]
        required={'layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','allowed_mask','key_positions','pv_head','full_format_logits','full_format_probs','selected_mask','selected_indices','selected_counts','selected_valid_mask','final_selected_logits','final_selected_probs','q_format_full_codes','k_format_full_codes','q_format_full_dq','k_format_full_dq'}
        for idx,record in sorted(self.records.items()):
            a,meta=record['actual'],record['meta'];missing=required-set(a)
            if missing:raise ValueError('Missing full-format fields '+str(missing))
            mask=a['selected_mask'];domain=a['allowed_mask'].expand_as(mask)
            if mask.dtype!=torch.bool or (mask&~domain).any() or not mask.any(-1).all():raise ValueError('Invalid selected domain')
            if not torch.equal(a['selected_counts'],mask.sum(-1)):raise ValueError('Selected counts mismatch')
            if not torch.equal(a['final_selected_logits'][mask],a['full_format_logits'][mask]):raise ValueError('Final selected scores changed')
            for name,value in a.items():
                if value is None or not value.is_floating_point():continue
                if name=='final_selected_logits':
                    if not torch.equal(torch.isneginf(value),~mask) or not torch.isfinite(value[mask]).all():raise ValueError('Invalid selected logits')
                elif not torch.isfinite(value).all():raise ValueError('Nonfinite full-format '+name)
            derived={}
            for label,q,k in [('fp',a['q_post_rope'].float(),a['k_visible'].float()),('source_int',a['q_dq'],a['k_dq'])]:
                logits=(q@k.transpose(-1,-2))/math.sqrt(meta['D']);probs=logits.softmax(-1)
                derived.update({'reference_'+label+'_logits_same_input':logits,'reference_'+label+'_probs_same_input':probs})
                derived.update({'reference_'+label+'_'+name+'_same_input':value for name,value in diagnostic_sets(logits,a['key_positions']).items()})
            # Dense reference is the actual full-format score distribution before selection.
            derived['full_format_dense_pv_same_input']=a['full_format_probs']@a['v_visible'].float()
            self.used_bytes+=sum(v.numel()*v.element_size() for v in derived.values())
            if self.used_bytes>self.budget_bytes:raise RuntimeError('Full-format trace budget exceeded')
            rid=f"layer{meta['layer_id']:02d}_query{meta['query_position']}_{meta['stage']}";path=self.output_dir/rid;path.mkdir()
            torch.save(a,path/'actual.pt');torch.save(derived,path/'derived.pt')
            storage={}
            for prefix in ('q','k'):
                c=a[prefix+'_format_full_codes'];e=a.get(prefix+'_format_e8m0');storage[prefix]=dict(logical_payload_bits=c.numel()*8,logical_format_metadata_bits=0 if e is None else e.numel()*8,actual_code_tensor_bytes=c.numel()*c.element_size(),actual_format_metadata_tensor_bytes=0 if e is None else e.numel()*e.element_size(),source_scale_tensor_bytes=a[prefix+'_scales'].numel()*a[prefix+'_scales'].element_size(),source_scale_is_separate=True)
            meta.update(storage=storage,fields={name:dict(shape=list(v.shape),dtype=str(v.dtype),capture_point='actual production full-format TopP boundary') for name,v in a.items() if v is not None},derived_fields={name:dict(shape=list(v.shape),dtype=str(v.dtype),unused_in_forward=True) for name,v in derived.items()},files={name:sha256(path/name) for name in ('actual.pt','derived.pt')})
            (path/'meta.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n');hashes=dict(meta_sha256=sha256(path/'meta.json'),actual_sha256=meta['files']['actual.pt'],derived_sha256=meta['files']['derived.pt'])
            index['records'].append(dict(record_id=rid,path=rid,**hashes))
            for head in range(meta['Hq']):
                valid=a['selected_valid_mask'][0,head,0]
                sets.append(dict(record_id=rid,variant_id=meta['variant_id'],layer_id=meta['layer_id'],query_position=meta['query_position'],stage=meta['stage'],head_id=head,p=meta['p'],unused_in_forward=False,set_role='actual_forward_selected_keys',final_operand_profile='decoded_full_format',key_positions=a['key_positions'].tolist(),selected_key_positions=a['selected_indices'][0,head,0][valid].tolist(),num_selected=int(a['selected_counts'][0,head,0]),allowed_count=a['key_positions'].numel(),mask_ref=dict(file=str((path/'actual.pt').resolve()),field='selected_mask',index=[0,head,0],sha256=hashes['actual_sha256']),**hashes))
        listing=self.output_dir.parent/'selection_sets.jsonl';listing.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in sets))
        index.update(tensor_bytes=self.used_bytes,selection_sets=dict(path=str(listing.resolve()),sha256=sha256(listing),rows=len(sets),unused_in_forward=False))
        (self.output_dir/'index.json').write_text(json.dumps(index,indent=2)+'\n');self.close();return index
