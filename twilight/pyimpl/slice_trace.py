"""Read-only observations of the production Top-p attention execution.

Only ``derived.pt`` contains CPU diagnostic calculations. Capturing a trace never
quantizes an operand, reruns attention, or replaces a forward result.
"""
import json
import math
from pathlib import Path
import torch
from .trace import TraceRecorder, current_context, _jsonable, sha256

SCHEMA = 'attention-topp-trace-v2'


def _reference_selection(logits, p):
    """CPU diagnostic selector; production selection remains in slice_backend."""
    probs = torch.softmax(logits.float(), -1)
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, order)
    if p == 1:
        keep = torch.ones_like(order, dtype=torch.bool)
    else:
        sums = sorted_probs.cumsum(-1)
        # Count preceding mass, thereby including the crossing item.
        keep = torch.cat((torch.ones_like(sums[..., :1], dtype=torch.bool), sums[..., :-1] < p), -1)
    mask = torch.zeros_like(keep).scatter(-1, order, keep)
    positions = torch.arange(logits.shape[-1]).expand_as(order)
    counts = mask.sum(-1)
    width = int(counts.max())
    indices = positions.masked_fill(~mask, logits.shape[-1]).sort(-1).values[..., :width]
    valid = torch.arange(width) < counts[..., None]
    indices = indices.masked_fill(~valid, -1)
    return probs, mask, indices, counts, valid


class SliceTraceRecorder(TraceRecorder):
    def attention_inputs(self, module, hidden, q, k, v, qq, kk, allowed):
        super().attention_inputs(module, hidden, q, k, v, qq, kk, allowed)
        config = getattr(module.state, 'attention_slice', None) or self.metadata.get('attention_slice')
        if not config:
            raise ValueError('Slice recorder requires attention_slice configuration')
        for idx, local, target in self.matching(module.layer_id):
            meta = self.records[idx]['meta']
            meta.update(_jsonable(config))
            meta.update(schema_version=SCHEMA, schema=SCHEMA, attention_slice=_jsonable(config),
                        prefix_rounding='signed_floor_divide' if config['estimator']=='tc_msb' else None,
                        p=config['selection']['p'], tie_break=config['selection']['tie_break'],
                        set_role='predicted_k' if config['estimator'] == 'tc_msb' else 'important_k_reference',
                        final_operand_origin='original_float' if qq is None else 'full_codes',
                        selection_passes=1, actual_executor=config['final_execution'],
                        actual_matmul='fp32' if qq is None else 'fp32_after_dq',
                        actual_pv='dense_zero_masked', selected_mask_semantics='True means selected for final execution',
                        final_selected_layout='B,Hq,Q,Kvisible; logits=-inf and probabilities=0 outside selected_mask',
                        selected_indices_layout='B,Hq,Q,Mmax; original key positions, -1 padding, valid_mask explicit',
                        cache_gqa='original floating K/V cache already expanded once to Hq',
                        quantizer_source_sha256=None if qq is None else sha256(Path(__file__).parent/'quant_attention'/'intflash.py'),
                        query_position_domain='model position, independent of expanded query head',
                        key_position_domain='original model key position, independent of expanded query head',
                        q_prefix_multiplier=None if qq is None else 2 ** (8-config['q_prefix_bits']),
                        k_prefix_multiplier=None if qq is None else 2 ** (8-config['k_prefix_bits']))

    def attention_slice_chunk(self, layer_id, start, end, payload, allowed):
        """Capture backend tensors at their real execution observation point."""
        for idx, local, target in self.matching(layer_id):
            if not start <= local < end:
                continue
            a = self.records[idx]['actual']
            if 'selected_mask' in a:
                raise RuntimeError('Repeated slice attention target')
            row = local-start
            visible = allowed[0, 0, local]
            keys = torch.as_tensor(current_context()['key_positions'], device=visible.device).reshape(-1)
            for name in ('estimator_logits', 'estimator_probs', 'selected_mask',
                         'final_selected_logits', 'final_selected_probs', 'dot_hh'):
                value = payload.get(name)
                self._put(idx, name, None if value is None else value[:, :, row:row+1, visible])
            for name in ('q_hi', 'q_lo'):
                value = payload.get(name)
                self._put(idx, name, None if value is None else value[:, :, row:row+1])
            for name in ('k_hi', 'k_lo'):
                value = payload.get(name)
                self._put(idx, name, None if value is None else value[:, :, visible])
            for name in ('selected_counts', 'selected_indices', 'selected_valid_mask'):
                # Backend may use the shorter valid_mask alias.
                value = payload.get(name, payload.get('valid_mask') if name == 'selected_valid_mask' else None)
                if value is None:
                    raise ValueError('Missing actual slice observation: ' + name)
                value = value[:, :, row:row+1]
                if name == 'selected_indices':
                    valid = value >= 0
                    value = keys[value.clamp_min(0)].masked_fill(~valid, -1)
                self._put(idx, 'valid_mask' if name == 'selected_valid_mask' else name, value)
            pv = payload.get('pv_head', payload.get('pv'))
            if pv is None:
                raise ValueError('Missing actual slice PV observation')
            self._put(idx, 'pv_head', pv[:, :, row:row+1])

    def _derived(self, a, meta):
        if meta['estimator'] != 'tc_msb':
            return {}
        qdq = a['q_codes'].float() * a['q_scales'].float().unsqueeze(-1)
        kdq = a['k_codes'].float() * a['k_scales'].float().unsqueeze(-1)
        logits = qdq @ kdq.transpose(-1, -2) / math.sqrt(meta['D'])
        probs, mask, indices, counts, valid = _reference_selection(logits, meta['p'])
        key_indices = a['key_positions'][indices.clamp_min(0)].masked_fill(~valid, -1)
        derived = {
            'reference_int_logits_same_input':logits,
            'reference_int_probs_same_input':probs,
            'reference_int_top_p_mask_same_input':mask,
            'reference_int_top_p_indices_same_input':key_indices,
            'reference_int_top_p_counts_same_input':counts,
            'reference_int_top_p_valid_mask_same_input':valid,
        }
        for prefix in ('q', 'k'):
            if a[prefix+'_lo'] is None:
                multiplier = meta[prefix+'_prefix_multiplier']
                derived[prefix+'_lo'] = a[prefix+'_codes'].to(torch.int16) - multiplier*a[prefix+'_hi'].to(torch.int16)
        return derived

    @staticmethod
    def _validate(a, meta):
        required = {'layer_input','layer_output','attn_input','o_proj_input','attn_output',
                    'q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales',
                    'q_dq','k_dq','allowed_mask','key_positions','pv_head','q_hi','k_hi','q_lo','k_lo',
                    'dot_hh','estimator_logits','estimator_probs','selected_mask','selected_indices',
                    'selected_counts','valid_mask','final_selected_logits','final_selected_probs'}
        if required-set(a): raise RuntimeError('Missing actual fields: ' + str(required-set(a)))
        mask = a['selected_mask']
        if mask.dtype != torch.bool or mask.shape != a['estimator_logits'].shape:
            raise ValueError('Invalid selected_mask dtype or layout')
        if (mask & ~a['allowed_mask']).any(): raise ValueError('Selection outside allowed domain')
        counts = mask.sum(-1)
        if not torch.equal(counts, a['selected_counts']): raise ValueError('Selection counts mismatch')
        valid = a['valid_mask']
        indices = a['selected_indices']
        if valid.dtype != torch.bool or indices.shape != valid.shape: raise ValueError('Invalid selected index layout')
        if not torch.equal(valid.sum(-1), counts) or (indices[~valid] != -1).any(): raise ValueError('Invalid selected index padding')
        keys = a['key_positions']
        reconstructed = ((indices[..., None] == keys) & valid[..., None]).any(-2)
        if not torch.equal(reconstructed, mask): raise ValueError('Selected indices disagree with mask')
        if (counts < 1).any(): raise ValueError('Empty selection')
        for name, value in a.items():
            if value is None or not value.is_floating_point(): continue
            if name == 'final_selected_logits':
                if not torch.equal(torch.isneginf(value), ~mask) or not torch.isfinite(value[mask]).all():
                    raise ValueError('Invalid final logit nonfinite domain')
            elif not torch.isfinite(value).all(): raise ValueError('Nonfinite actual tensor: ' + name)
        if (a['final_selected_probs'][~mask] != 0).any(): raise ValueError('Nonzero final probability outside selection')
        if not torch.allclose(a['final_selected_probs'].sum(-1), torch.ones_like(counts, dtype=torch.float32), atol=1e-6, rtol=1e-6):
            raise ValueError('Final probabilities do not normalize within S')
        if meta['estimator'] == 'fp_full' and any(a[name] is not None for name in ('q_codes','k_codes','q_scales','k_scales')):
            raise ValueError('FP trace must not manufacture quantization operands')
        if meta['estimator'] != 'fp_full' and any(a[name] is None for name in ('q_codes','k_codes','q_scales','k_scales','q_dq','k_dq')):
            raise ValueError('INT trace requires actual full source codes/scales/DQ')
        if meta['estimator'] == 'tc_msb' and any(a[name] is None for name in ('q_hi','k_hi','dot_hh')):
            raise ValueError('MSB actual prefix observations are required')

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
                if name == 'key_positions': layout='Kvisible'
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
                point = ('CPU diagnostic from captured full source codes; never used by model' if diagnostic else points.get(name, 'Actual '+name+' at production attention observation point'))
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
