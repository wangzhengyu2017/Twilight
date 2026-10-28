"""Real-model compatibility, full patch coverage and original-float cache checks."""
import argparse
import os
from pathlib import Path
import time
import traceback

from .common import (canonical_hash, code_provenance, configure_runtime, environment, load_config, load_model, validate_model, write_json, sha256_file, validate_assets, load_tokens, tokens_hash)


HF_EAGER_THRESHOLDS = {'relative_l2': 0.02, 'max_abs': 1.0, 'mean_nll_abs_diff': 0.02}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    asset_directory = os.environ.get('ASSET_DIR')
    parser.add_argument('--assets', default=str(Path(asset_directory) / 'asset_manifest.json') if asset_directory else None, help='Frozen assets manifest; defaults to ASSET_DIR/asset_manifest.json')
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    write_json(out / 'status.json', {'status': 'running'})
    stage = 'initialization'
    started = time.perf_counter()
    try:
        import torch
        import torch.nn.functional as F
        from transformers.cache_utils import DynamicCache
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        from twilight.pyimpl.attention import enable_sparse_attention, reset_sparse_config
        from twilight.pyimpl.trace import CallCounters, forward_context
        configure_runtime()
        config = load_config(args.config)
        cfg = validate_model(args.model_path)
        if args.assets is None:
            raise ValueError('Preflight requires --assets or ASSET_DIR for the identical frozen WikiText prefix')
        asset_path = Path(args.assets).resolve()
        token_path = asset_path.parent / 'tokens.pt'
        assets = validate_assets(args.model_path, token_path, asset_path)
        prefix = load_tokens(token_path)[:130]
        if prefix.numel() != 130:
            raise ValueError('Preflight requires at least 130 frozen WikiText tokens')
        input_manifest = {'source_policy': 'frozen_wikitext_first_130', 'num_tokens': 130, 'tokens_sha256': assets['tokens_sha256'], 'subset_sha256': tokens_hash(prefix), 'asset_manifest_sha256': canonical_hash(assets)}
        write_json(out / 'asset_manifest.json', assets)
        write_json(out / 'input_manifest.json', input_manifest)
        model_root = Path(args.model_path).resolve()
        model_hashes = assets['model_file_hashes']
        write_json(out / 'model_manifest.json', {'model_path': str(model_root), 'model_file_hashes': model_hashes, 'model_files_sha256': canonical_hash(model_hashes)})
        write_json(out / 'environment.json', environment())
        code_provenance(out)
        write_json(out / 'resolved_config.json', config)
        stage = 'model_load'
        model = load_model(args.model_path)
        # The frozen dataset prefix is identical across methods; no refitting or token selection.
        ids = prefix.to(device='cuda', dtype=torch.long)[None]
        reference = None
        reference_loss = None
        method = config['attention_quantization']['method']
        stage = 'hf_eager_reference'
        with torch.inference_mode():
            if method == 'fp_common':
                reference = model(input_ids=ids[:, :128], position_ids=torch.arange(128, device='cuda')[None], use_cache=False).logits.detach()
                reference_loss = F.cross_entropy(reference[0,:-1].float(), ids[0,1:128]).item()
                if not torch.isfinite(reference).all():
                    raise ValueError('HF eager produced nonfinite logits')
        stage = 'enable_patch'
        reset_sparse_config()
        enable_sparse_attention(model, sparse_config=config, query_chunk=64)
        if getattr(model, '_twilight_patch_count', None) != cfg['num_hidden_layers']:
            raise ValueError('Attention patch count does not equal model layer count')
        counters = CallCounters()
        def call(tokens, positions, cache=None):
            pos = torch.tensor(positions, device='cuda', dtype=torch.long)
            expected_length = positions[-1]+1 if cache is not None else len(positions)
            with forward_context(run_id=out.name, sample_id=0, window_id=None, stage='decode' if cache is not None and positions[0] > 0 else 'prefill', global_query_positions=positions, position_ids=positions, key_positions=list(range(expected_length)), expected_cache_length=expected_length, input_ids=tokens[0].tolist(), cache_position=positions, counters=counters):
                result = model(input_ids=tokens, position_ids=pos[None], cache_position=pos, past_key_values=cache, use_cache=cache is not None, output_attentions=False, return_dict=True)
            if not torch.isfinite(result.logits).all():
                raise ValueError('Patched model produced nonfinite logits')
            return result.logits
        stage = 'no_cache_prefill'
        comparison = None
        with torch.inference_mode():
            patched = call(ids[:, :128], list(range(128)))
            if reference is not None:
                delta = patched.float()-reference.float()
                relative_l2 = (delta.norm()/reference.float().norm()).item()
                max_abs = delta.abs().max().item()
                loss = F.cross_entropy(patched[0,:-1].float(), ids[0,1:128]).item()
                comparison = {'num_tokens':128, 'relative_l2':relative_l2, 'max_abs':max_abs, 'mean_nll_reference':reference_loss, 'mean_nll_patched':loss, 'mean_nll_abs_diff':abs(loss-reference_loss), 'thresholds':dict(HF_EAGER_THRESHOLDS)}
                if any(comparison[key] > threshold for key, threshold in HF_EAGER_THRESHOLDS.items()):
                    raise ValueError(f'FP common failed HF eager comparison: {comparison}')
            del patched, reference
        # Independently reconstruct each K/V update directly from the actual projections.
        # These originals must appear in cache exactly, after one RoPE and repeat_kv.
        stage = 'cache_invariants'
        expected_updates = {}
        cos_sin = {}
        handles = []
        for layer_id, layer in enumerate(model.model.layers):
            def before(module, arguments, kwargs, layer_id=layer_id):
                positions = kwargs.get('position_embeddings')
                if positions is None:
                    raise ValueError('HF4.45.2 must provide shared position_embeddings')
                cos_sin[layer_id] = positions
            handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
            def capture_k(module, arguments, output, layer_id=layer_id):
                b, length, _ = output.shape
                key = output.view(b,length,cfg['num_key_value_heads'],128).transpose(1,2)
                cos, sin = cos_sin[layer_id]
                _, key = apply_rotary_pos_emb(torch.zeros_like(key), key, cos, sin)
                expected_updates.setdefault(layer_id, {})['k'] = repeat_kv(key,4).clone()
            def capture_v(module, arguments, output, layer_id=layer_id):
                b,length,_ = output.shape
                value = output.view(b,length,cfg['num_key_value_heads'],128).transpose(1,2)
                expected_updates.setdefault(layer_id, {})['v'] = repeat_kv(value,4).clone()
            handles.append(layer.self_attn.k_proj.register_forward_hook(capture_k))
            handles.append(layer.self_attn.v_proj.register_forward_hook(capture_v))
        cache = DynamicCache()
        cache_checks = []
        with torch.inference_mode():
            for step in range(3):
                old_length = 0 if step == 0 else 127+step
                positions = list(range(128)) if step == 0 else [127+step]
                inp = ids[:,:128] if step == 0 else ids[:,127+step:128+step]
                previous = [(k.clone(),v.clone()) for k,v in zip(cache.key_cache,cache.value_cache)]
                logits = call(inp, positions, cache)
                del logits
                length = 128+step
                if cache.get_seq_length() != length or len(cache.key_cache) != cfg['num_hidden_layers']:
                    raise ValueError('DynamicCache length/layer coverage mismatch')
                for layer_id, (key,value) in enumerate(zip(cache.key_cache,cache.value_cache)):
                    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16 or key.shape != (1,32,length,128) or value.shape != key.shape:
                        raise ValueError('Cache must contain original BF16 Hq-layout K/V')
                    if step and (not torch.equal(key[:,:,:old_length],previous[layer_id][0]) or not torch.equal(value[:,:,:old_length],previous[layer_id][1])):
                        raise ValueError('An existing cache prefix was modified')
                    expected = expected_updates[layer_id]
                    if not torch.equal(key[:,:,old_length:], expected['k']) or not torch.equal(value[:,:,old_length:],expected['v']):
                        raise ValueError(f'Layer {layer_id} cache differs from original post-RoPE projected K/V')
                cache_checks.append({'length':length,'all_layers':True,'original_kv_exact':True,'prefix_unchanged':True})
        for handle in handles:
            handle.remove()
        if set(counters.counts) != {str(i) for i in range(cfg['num_hidden_layers'])} or any(v != {method:{'prefill':2,'decode':2}} for v in counters.counts.values()):
            raise ValueError(f'Preflight full layer coverage mismatch: {counters.counts}')
        diagnostics = counters.as_dict()['quantization_diagnostics']
        write_json(out / 'quantization_diagnostics.json', diagnostics)
        result = {'quantization_diagnostics':diagnostics,'status':'complete','method':method,'hf_eager_comparison':comparison,'patch_count':model._twilight_patch_count,'cache_checks':cache_checks,'layer_calls':counters.counts,'cache_storage':'original_float','quantization_at':'attention_read','all_logits_finite':True,'elapsed_seconds':time.perf_counter()-started,'config_sha256':canonical_hash(config),'model_files_sha256':canonical_hash(model_hashes),'input_manifest':input_manifest}
        write_json(out / 'preflight.json', result)
        write_json(out / 'status.json', {'status':'complete'})
        print(result)
    except Exception as error:
        write_json(out / 'status.json', {'status':'failed','stage':stage,'reason':str(error)})
        (out / 'run.log').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
