"""Actual RS observations and bounded layer/head saturation reductions."""
import json
import torch
from .trace import TraceRecorder,CallCounters,current_context,sha256
from .quant_attention.rs_codec import allocation_report,PAPER_SHA256

SCHEMA='attention-rs-trace-v1'
THRESHOLDS={'095':.95,'090':.9,'085':.85}

def diagnostic_sets(logits,positions,p):
    # Offline only; never called by attention_rs or used to form its output.
    probabilities=logits.float().softmax(-1);order=logits.argsort(dim=-1,descending=True,stable=True)
    cumulative=probabilities.gather(-1,order).cumsum(-1)
    keep=torch.cat((torch.ones_like(cumulative[...,:1],dtype=torch.bool),cumulative[...,:-1]<p),-1)
    mask=torch.zeros_like(keep).scatter(-1,order,keep);counts=mask.sum(-1);width=int(counts.max())
    indices=positions.expand_as(mask).masked_fill(~mask,int(positions.max())+1).sort(-1).values[...,:width]
    valid=torch.arange(width)<counts[...,None]
    return dict(mask=mask,counts=counts,indices=indices.masked_fill(~valid,-1),valid_mask=valid)

class RSCallCounters(CallCounters):
    def __init__(self):super().__init__();self.rs={}
    def record_rs(self,layer,operand,source,side):
        # Each field has size O(H*constant); K counts include every cache read.
        c=source.codes.to(torch.int16);scale=source.scales.float()[...,None]
        axes=(0,2,3);data={'calls':1,'denominator':torch.ones_like(c,dtype=torch.int64).sum(axes),'saturation_count':side['saturation_mask'].sum(axes)}
        for value in range(120,128):data['source_code_'+str(value)+'_count']=(c==value).sum(axes)
        for kind,residual in [('conversion',side['conversion_residual']),('prefix',side['prefix_residual']),('total',side['total_prefix_residual'])]:
            real=residual.float()*scale
            for suffix,error in [('integer',residual.float()),('real',real)]:
                data[kind+'_'+suffix+'_abs_sum']=error.abs().sum(axes,dtype=torch.float64);data[kind+'_'+suffix+'_sq_sum']=error.square().sum(axes,dtype=torch.float64)
                data[kind+'_'+suffix+'_abs_max']=error.abs().amax(axes);data[kind+'_'+suffix+'_sq_max']=error.square().amax(axes)
        for delta in range(1,9):data['conversion_delta_'+str(delta)+'_count']=(side['conversion_residual']==delta).sum(axes)
        # Conditional distributions retain exact integer histograms by source sign.
        for sign,condition in [('negative',c<0),('zero',c==0),('positive',c>0)]:
            for name,tensor,lo,hi in [('source',c,-127,127),('native',side['native_int'],-136,119),('prefix',16*side['H'],-128,112)]:
                width=hi-lo+1;heads=tensor.shape[1]
                values=tensor.permute(1,0,2,3).reshape(heads,-1).long()-lo
                flags=condition.permute(1,0,2,3).reshape(heads,-1).long()
                histogram=torch.zeros((heads,width),device=tensor.device,dtype=torch.int64)
                histogram.scatter_add_(1,values,flags)
                data[sign+'_'+name+'_histogram']=histogram
        key=str(layer)+':'+operand;dest=self.rs.setdefault(key,{})
        for name,value in data.items():
            if isinstance(value,torch.Tensor):value=value.detach()
            if name not in dest:dest[name]=value
            elif name.endswith('_max'):dest[name]=torch.maximum(dest[name],value)
            else:dest[name]=dest[name]+value
    def rs_summary(self):
        return dict(count_semantics='per quantizer call after expanded cache read, not unique values',axes='layer,operand,query_head; batch/token/dimension reduced',
            histogram_ranges=dict(source=[-127,127],native=[-136,119],prefix=[-128,112]),
            by_layer_operand={key:{name:value.cpu().tolist() if isinstance(value,torch.Tensor) else value for name,value in data.items()} for key,data in self.rs.items()})

class RSTraceRecorder(TraceRecorder):
    def attention_inputs(self,module,hidden,q,k,v,qq,kk,allowed):
        super().attention_inputs(module,hidden,q,k,v,qq,kk,allowed);config=module.state.attention_rs
        for idx,local,target in self.matching(module.layer_id):
            self.records[idx]['meta'].update(schema_version=SCHEMA,attention_rs=config,variant_id=config['variant_id'],
                rs_mode=config['mode'],selection_passes=config['selection_passes'],final_operand_origin=config['final_operand_origin'],final_scores=config['final_scores'],final_execution=config['final_execution'],
                paper_sha256=PAPER_SHA256,paper_location='HPCA2025 BADA printed823 Eqs5-7;824 Eq8;826 Fig5',
                adaptation='fixed original per-head-token source scales, not paper LSQ/QAT; positive codes120..127 clip119',
                diagnostic_sets_unused_in_forward=True,diagnostic_thresholds=list(THRESHOLDS.values()),
                actual_matmul='DQ-first FP32 actual packed-word decode; separate original-source selected bmm only for MSB',
                actual_pv='dense_zero_masked',cache_gqa='original floating cache expanded once before quantization',
                tf32_allowed=torch.backends.cuda.matmul.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    def attention_rs_chunk(self,layer,start,end,payload,allowed):
        for idx,local,target in self.matching(layer):
            if not start<=local<end:continue
            if 'selected_mask' in self.records[idx]['actual']:raise RuntimeError('Repeated RS target')
            row=local-start;visible=allowed[0,0,local];keys=torch.as_tensor(current_context()['key_positions'],device=visible.device).reshape(-1)
            for name,value in payload.items():
                if name in ('production_allocations','floating_coexistence_bytes'):
                    self.records[idx]['meta'][name]=value
                    continue
                if name.startswith('q_rs_'):value=value[:,:,row:row+1]
                elif name.startswith('k_rs_'):value=value[:,:,visible]
                elif name in ('estimator_logits','estimator_probs','selected_mask','final_selected_logits','final_selected_probs'):value=value[:,:,row:row+1,visible]
                elif name in ('selected_indices','selected_counts','selected_valid_mask'):
                    value=value[:,:,row:row+1]
                    if name=='selected_indices':value=keys[value.clamp_min(0)].masked_fill(value<0,-1)
                elif name.startswith('selected_source_'):value=value[:,:,row:row+1]
                elif name=='pv_head':value=value[:,:,row:row+1]
                elif name!='sqrt_D_f32':raise ValueError('Unknown RS payload '+name)
                self._put(idx,name,value)
    def finalize(self):
        if len(self.records)!=len(self.targets):raise RuntimeError('Missing RS trace targets')
        self.output_dir.mkdir(parents=True,exist_ok=False);index=dict(schema_version=SCHEMA,records=[]);sets=[];diagnostics=[]
        required={'layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','allowed_mask','key_positions','pv_head','estimator_logits','estimator_probs','selected_mask','selected_indices','selected_counts','selected_valid_mask','final_selected_logits','final_selected_probs'}
        for idx,record in sorted(self.records.items()):
            a,m=record['actual'],record['meta'];missing=required-set(a)
            if missing:raise ValueError('Missing RS observations '+str(missing))
            mask=a['selected_mask'];domain=a['allowed_mask'].expand_as(mask);counts=mask.sum(-1);valid=a['selected_valid_mask'];indices=a['selected_indices'];positions=a['key_positions']
            if (mask&~domain).any() or not torch.equal(counts,a['selected_counts']) or not torch.equal(valid.sum(-1),counts) or (indices[~valid]!=-1).any():raise ValueError('Invalid RS actual set')
            if not torch.equal(((indices[...,None]==positions)&valid[...,None]).any(-2),mask):raise ValueError('RS indices/mask mismatch')
            if m['rs_mode']=='dense_full' and not torch.equal(mask,domain):raise ValueError('Dense RS must execute all legal keys')
            if not torch.equal(torch.isneginf(a['final_selected_logits']),~mask) or (a['final_selected_probs'][~mask]!=0).any():raise ValueError('RS final score domain')
            derived={};storage={}
            for prefix,bits in [('q',m['attention_rs']['q_bits']),('k',m['attention_rs']['k_bits'])]:
                side={name.removeprefix(prefix+'_rs_'):value for name,value in a.items() if name.startswith(prefix+'_rs_')}
                from types import SimpleNamespace
                storage[prefix]=allocation_report(SimpleNamespace(codes=a[prefix+'_codes'],scales=a[prefix+'_scales'],dq=a[prefix+'_dq']),side,bits)
                storage[prefix]['allocation_scope']='captured CPU clones only; production allocations and aliases are separate in production_allocations'
            # Every record gets native/source/prefix references; diagnostics never feed forward.
            native=(a['q_rs_full_dq']@a['k_rs_full_dq'].transpose(-1,-2))/m['D']**.5
            source=(a['q_dq']@a['k_dq'].transpose(-1,-2))/m['D']**.5
            floating=(a['q_post_rope'].float()@a['k_visible'].float().transpose(-1,-2))/m['D']**.5
            for label,logits in [('native_full',native),('original_source',source),('original_fp',floating),('actual_predictor',a['estimator_logits'])]:
                derived[label+'_logits']=logits;derived[label+'_probs']=logits.softmax(-1)
                for suffix,p in THRESHOLDS.items():
                    for name,value in diagnostic_sets(logits,positions,p).items():derived[label+'_diagnostic_p'+suffix+'_'+name]=value
            derived['native_final_on_actual_set']=native.masked_fill(~mask,-torch.inf);derived['source_final_on_actual_set']=source.masked_fill(~mask,-torch.inf)
            derived['native_probs_on_actual_set']=derived['native_final_on_actual_set'].softmax(-1);derived['source_probs_on_actual_set']=derived['source_final_on_actual_set'].softmax(-1)
            self.used_bytes+=sum(v.numel()*v.element_size() for v in derived.values())
            if self.used_bytes>self.budget_bytes:raise RuntimeError('RS trace budget exceeded')
            rid=f"layer{m['layer_id']:02d}_query{m['query_position']}_{m['stage']}";path=self.output_dir/rid;path.mkdir()
            torch.save(a,path/'actual.pt');torch.save(derived,path/'derived.pt')
            m.update(storage=storage,fields={name:dict(shape=list(value.shape),dtype=str(value.dtype),capture_point='actual production attention boundary') for name,value in a.items()},
                derived_fields={name:dict(shape=list(value.shape),dtype=str(value.dtype),unused_in_forward=True,estimand='same_input_estimator') for name,value in derived.items()},files={name:sha256(path/name) for name in ('actual.pt','derived.pt')})
            (path/'meta.json').write_text(json.dumps(m,indent=2,allow_nan=False)+'\n');hashes=dict(meta_sha256=sha256(path/'meta.json'),actual_sha256=m['files']['actual.pt'],derived_sha256=m['files']['derived.pt'])
            index['records'].append(dict(record_id=rid,path=rid,**hashes))
            for head in range(m['Hq']):
                common=dict(record_id=rid,variant_id=m['variant_id'],layer_id=m['layer_id'],query_position=m['query_position'],stage=m['stage'],head_id=head,selection_passes=m['selection_passes'],final_operand_origin=m['final_operand_origin'],**hashes)
                sets.append(dict(**common,set_role='actual_all_legal_execution' if m['rs_mode']=='dense_full' else 'actual_used_prediction_set',p=m['attention_rs']['selection']['p'],selected_key_positions=indices[0,head,0][valid[0,head,0]].tolist(),num_selected=int(counts[0,head,0])))
                for suffix,p in THRESHOLDS.items():
                    key='native_full_diagnostic_p'+suffix+'_';dm=derived[key+'mask'][0,head,0]
                    diagnostics.append(dict(**common,p=p,unused_in_forward=True,set_role='offline_native_full_important_keys',selected_key_positions=positions[dm].tolist(),num_selected=int(dm.sum()),mask_ref=dict(file=str((path/'derived.pt').resolve()),field=key+'mask',index=[0,head,0],sha256=hashes['derived_sha256'])))
        for name,rows in [('selection_sets',sets),('diagnostic_sets',diagnostics)]:
            path=self.output_dir.parent/(name+'.jsonl');path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows));index[name]=dict(path=str(path.resolve()),sha256=sha256(path),rows=len(rows))
        index['tensor_bytes']=self.used_bytes;(self.output_dir/'index.json').write_text(json.dumps(index,indent=2)+'\n');self.close();return index
