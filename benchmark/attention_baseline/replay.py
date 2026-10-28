"""Independent, CPU-only verification of recorded execution and integer arithmetic."""
import argparse
import math
from pathlib import Path
import torch
from .common import load_json, write_json, sha256_file, canonical_hash, semantic_config

from twilight.pyimpl.quant_attention.intflash import SOURCE_REVISION, SOURCE_DELTA, INPUT_CAST_POLICY

ATOL=1e-5
RTOL=1e-4
REQUIRED=('layer_input','layer_output','attn_input','o_proj_input','attn_output','q_post_rope','k_visible','v_visible','q_codes','k_codes','q_scales','k_scales','q_dq','k_dq','logits_e','probs','pv_head','allowed_mask','key_positions')


def error(a,b):
    a,b=a.double(),b.double()
    return {'max_abs':float((a-b).abs().max()),'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))}


def replay_run(run_dir):
    root=Path(run_dir); trace=root/'trace'; index=load_json(trace/'index.json')
    assert load_json(root/'status.json')['status']=='complete', 'run incomplete'
    records=index['records']; assert len(records)==6, f'expected six trace records, got {len(records)}'
    seen=set(); identities=set(); reports=[]
    metrics=load_json(root/'metrics.json')
    config=load_json(root/'resolved_config.json')
    assets=load_json(root/'asset_manifest.json')
    code=load_json(root/'code_manifest.json')
    assert code['code_sha256']==metrics['code_sha256']==canonical_hash({'head':code['head'],'files':code['files']}), 'invalid run code identity'
    assert code['patch_sha256']==sha256_file(root/'code.patch'), 'run patch hash mismatch'
    assert canonical_hash(assets)==metrics['asset_manifest_sha256'], 'run asset hash mismatch'
    algorithm=semantic_config(config.get('algorithm',config))
    for entry in records:
        rid=entry['record_id']; assert rid not in seen, 'duplicate record';seen.add(rid)
        path=(trace/entry.get('path',rid)).resolve(); assert path.is_relative_to(trace.resolve()), 'unsafe trace path'
        for name,key in [('meta.json','meta_sha256'),('actual.pt','actual_sha256'),('derived.pt','derived_sha256')]:
            assert entry[key]==sha256_file(path/name), f'{rid}/{name}: index hash mismatch'
        meta=load_json(path/'meta.json')
        identity=(meta['layer_id'],meta['query_position'],meta['stage'])
        assert identity not in identities, 'duplicate target identity'
        identities.add(identity)
        assert meta['sample_id']==0 and meta['window_id'] is None and meta['predicted_target']==meta['query_position']+1, 'invalid trace sample/target'
        assert meta['method']==metrics['method']==config.get('algorithm',config)['attention_quantization']['method'], 'trace method mismatch'
        assert meta['backend']=='torch_qdq' and meta['cache_storage']=='original_float' and meta['quantization_at']=='attention_read', 'invalid execution metadata'
        for key in ('code_sha256','tokens_sha256'):
            assert meta[key]==metrics[key], f'trace provenance mismatch: {key}'
        assert meta['model_file_hashes']==assets['model_file_hashes'], 'trace model identity mismatch'
        assert meta['config_sha256']==canonical_hash(algorithm)==metrics['semantic_config_sha256'], 'trace configuration identity mismatch'
        assert meta['model_dtype']=='torch.bfloat16' and metrics['model_dtype']=='bfloat16', 'trace model dtype mismatch'
        assert meta['scale_broadcast']=='[B,H,L,1]' and meta['rounding']=='torch_round_ties_to_even', 'quantizer convention mismatch'
        assert (meta['Hq'],meta['Hkv'],meta['D'])==(32,8,128) and meta['head_to_kv']==[h//4 for h in range(32)], 'head metadata mismatch'
        if meta['method']=='intflash_qk8':
            assert meta['source_revision']==SOURCE_REVISION and meta['source_delta']==SOURCE_DELTA and meta['input_cast_policy']==INPUT_CAST_POLICY, 'quantizer source mismatch'
            assert meta['zero_point']==0 and meta['statistics_axis']=='D' and meta['quantizer_input_dtype']=='float16' and meta['source_input_cast']=='bf16_to_fp16', 'quantizer metadata mismatch'
        for name in ('actual.pt','derived.pt'):
            assert meta['files'][name]==sha256_file(path/name), f'{rid}/{name}: metadata hash mismatch'
        actual=torch.load(path/'actual.pt',map_location='cpu',weights_only=True); derived=torch.load(path/'derived.pt',map_location='cpu',weights_only=True)
        assert all(k in actual for k in REQUIRED), f'{rid}: missing actual fields'
        for k,t in actual.items():
            assert k in meta['fields'], f'{rid}/{k}: missing field metadata'
            if torch.is_tensor(t):
                assert t.device.type=='cpu'
                if t.is_floating_point(): assert torch.isfinite(t).all(), f'{rid}/{k}: nonfinite'
                info=meta['fields'][k]
                assert info and info['layout'] and info['capture_point'], f'{rid}/{k}: incomplete field metadata'
                assert list(t.shape)==info['shape'], f'{rid}/{k}: shape metadata mismatch'
                assert str(t.dtype).replace('torch.','')==str(info['dtype']).replace('torch.',''), f'{rid}/{k}: dtype metadata mismatch'
            else:
                assert t is None and meta['fields'][k] is None, f'{rid}/{k}: invalid optional field'
        q,k,v=actual['q_post_rope'],actual['k_visible'],actual['v_visible']; dot=None
        assert q.ndim==k.ndim==v.ndim==4 and q.shape[-2]==1 and k.shape==v.shape and q.shape[:2]==k.shape[:2] and q.shape[-1]==k.shape[-1], 'invalid Q/K/V layout'
        assert q.shape==(1,32,1,128) and k.shape==(1,32,meta['query_position']+1,128), 'Q/K/V dimensions mismatch'
        assert q.dtype==k.dtype==v.dtype==torch.bfloat16, 'original Q/K/V dtype mismatch'
        assert (actual['q_codes'] is not None)==(meta['method']=='intflash_qk8'), 'method/codes mismatch'
        if actual['q_codes'] is not None:
            qc,kc=actual['q_codes'],actual['k_codes'];qs,ks=actual['q_scales'],actual['k_scales']
            assert qc.shape==q.shape and kc.shape==k.shape and qs.shape==q.shape[:-1] and ks.shape==k.shape[:-1], 'code/scale dimensions mismatch'
            assert qs.dtype==ks.dtype==torch.float16, 'source scale dtype mismatch'
            assert qc.dtype==kc.dtype==torch.int8 and int(qc.min())>=-127 and int(kc.min())>=-127, 'invalid codes'
            assert torch.isfinite(qs).all() and torch.isfinite(ks).all() and (qs>0).all() and (ks>0).all(), 'invalid scales'
            # Reconstruct the fixed FP16 formula from original real model Q/K,
            # independently of the production quantizer and stored integer codes.
            for original,codes,scales in ((q,qc,qs),(k,kc,ks)):
                source=original.to(torch.float16)
                assert torch.isfinite(source).all(), 'nonfinite source cast'
                maximum=source.abs().max(dim=-1).values
                expected_scales=torch.where(maximum==0,torch.ones_like(maximum),maximum/127)
                assert (expected_scales>0).all(), 'source nonzero scale underflow'
                rounded=torch.round(source/expected_scales.unsqueeze(-1))
                assert torch.isfinite(rounded).all() and (rounded.abs()<=127).all(), 'invalid source codes'
                assert torch.equal(scales,expected_scales) and torch.equal(codes,rounded.to(torch.int8)), 'real tensor source codes/scales mismatch'
            # Deliberately avoid matmul for the independent integer reference.
            dot=torch.zeros(qc.shape[:-1]+(kc.shape[-2],),dtype=torch.int64)
            for d in range(qc.shape[-1]): dot += qc[...,d].long().unsqueeze(-1)*kc[...,d].long().unsqueeze(-2)
            assert derived['int64_dot'].dtype==torch.int64 and torch.equal(dot,derived['int64_dot']), 'integer reference mismatch'
            assert int(dot.abs().max())<=q.shape[-1]*127*127, 'integer bound violated'
            q=qc.float()*qs.float().unsqueeze(-1);k=kc.float()*ks.float().unsqueeze(-1)
            for key,value in [('q_dq',q),('k_dq',k)]:
                assert actual.get(key) is not None and torch.equal(actual[key],value), f'{key} mismatch'
        else:
            assert all(actual[n] is None for n in ('k_codes','q_scales','k_scales')), 'partial quantization fields'
            assert derived.get('int64_dot') is None, 'unexpected FP integer dot'
            q,k=q.float(),k.float()
        assert actual['allowed_mask'].dtype==torch.bool and actual['key_positions'].dtype==torch.int64, 'invalid mask/positions dtype'
        expected_positions=torch.arange(meta['query_position']+1,dtype=torch.int64)
        assert torch.equal(actual['key_positions'],expected_positions), 'visible prefix positions mismatch'
        assert meta['cache_length']==meta['expected_cache_length']==(meta['query_position']+1 if meta['stage']=='decode' else 128), 'cache metadata mismatch'
        assert actual['allowed_mask'].all(), 'visible prefix has prohibited key'
        mask=actual['allowed_mask'];assert mask.shape==(1,1,1,k.shape[-2]) and mask.any(-1).all(), 'invalid allowed domain'
        logits=(q@k.transpose(-1,-2))/math.sqrt(q.shape[-1]);logits=logits.masked_fill(~mask,float('-inf'))
        probs=torch.softmax(logits,dim=-1);pv=probs@v.float()
        refs={'logits_e':logits,'probs':probs,'pv_head':pv}; report={'record_id':rid,'integer_exact':dot is not None,'errors':{}}
        for name,ref in refs.items():
            for source,data in [('actual',actual),('derived',derived)]:
                got=data[name];assert got.shape==ref.shape, f'{name} shape mismatch'
                assert torch.equal(torch.isinf(got),torch.isinf(ref)), f'{name} infinity domain mismatch'
                finite=torch.isfinite(ref)
                assert torch.allclose(got[finite],ref[finite],atol=ATOL,rtol=RTOL), f'{rid}/{source}/{name}: tolerance failure {error(got[finite],ref[finite])}'
                report['errors'][source+'_'+name]=error(got[finite],ref[finite])
        if dot is not None:
            scaled=dot.float()*actual['q_scales'].float().unsqueeze(-1)*actual['k_scales'].float().unsqueeze(-2)/math.sqrt(q.shape[-1])
            finite=mask.expand_as(logits)
            assert torch.allclose(scaled[finite],actual['logits_e'][finite],atol=ATOL,rtol=RTOL), 'integer scaled logits tolerance failure'
            report['errors']['integer_scaled_logits']=error(scaled[finite],actual['logits_e'][finite])
        reports.append(report)
    assert identities=={(layer,q,stage) for layer in (0,16,31) for q,stage in ((127,'prefill'),(128,'decode'))}, 'trace target identities mismatch'
    return {'status':'complete','passed':True,'run_dir':str(root.resolve()),'atol':ATOL,'rtol':RTOL,'num_records':len(reports),'records':reports,'index_sha256':sha256_file(trace/'index.json')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    try: report=replay_run(a.run_dir)
    except Exception as e:
        write_json(a.output,{'status':'failed','passed':False,'reason':str(e),'atol':ATOL,'rtol':RTOL});raise
    write_json(a.output,report)

if __name__=='__main__':main()
