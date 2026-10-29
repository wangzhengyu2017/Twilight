"""One scorer for frozen WikiText windows and the 128+16 cached decode protocol."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.attention_baseline.common import (canonical_hash, code_provenance, configure_runtime, environment, load_config, load_model, load_tokens, scoring_ranges, sha256_file, tensor_hash, tokens_hash, validate_assets, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', '--model-path', required=True)
    p.add_argument('--tokens', required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--algo-config-path', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--dtype', choices=['bfloat16'], default='bfloat16')
    p.add_argument('--query-chunk', type=int, default=64)
    p.add_argument('--ppl-mode', choices=['windows', 'cached_decode'], required=True)
    p.add_argument('--context-length', type=int, default=2048)
    p.add_argument('--stride', type=int, default=1024)
    p.add_argument('--max-tokens', type=int)
    p.add_argument('--prefill-tokens', type=int, default=128)
    p.add_argument('--decode-targets', type=int, default=16)
    p.add_argument('--trace-targets')
    p.add_argument('--gate-report')
    p.add_argument('--slice-gate-report')
    p.add_argument('--dense-format-gate-report')
    p.add_argument('--full-format-gate-report')
    return p


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache
    from twilight.pyimpl.attention import enable_sparse_attention, reset_sparse_config
    from twilight.pyimpl.trace import CallCounters, TraceRecorder, forward_context
    from twilight.pyimpl.quant_attention.intflash import SOURCE_DELTA, INPUT_CAST_POLICY
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    start = time.perf_counter()
    stage = 'initialization'
    log = (output / 'run.log').open('w', buffering=1)
    metrics = {'sum_nll': None, 'num_targets': None, 'ppl': None}
    write_json(output / 'status.json', {'status': 'running', 'stage': stage})
    try:
        configure_runtime()
        config = load_config(args.algo_config_path)
        method = config['attention_quantization']['method']
        slice_config = config.get('attention_slice')
        dense_config = config.get('attention_dense_format')
        full_config = config.get('attention_full_format')
        if args.query_chunk < 1:
            raise ValueError('query_chunk must be positive')
        if (args.context_length, args.stride) != (2048, 1024):
            raise ValueError('Window protocol is fixed at C2048 S1024')
        if (args.prefill_tokens, args.decode_targets) != (128, 16):
            raise ValueError('Decode protocol is fixed at P128 L16')
        if args.ppl_mode == 'cached_decode' and args.max_tokens is not None:
            raise ValueError('cached_decode does not accept max_tokens')
        if args.trace_targets and args.ppl_mode != 'cached_decode':
            raise ValueError('Trace capture is supported only on the short cached decode')
        stage = 'asset_validation'
        assets = validate_assets(args.model_path, args.tokens, args.assets)
        ids = load_tokens(args.tokens)
        profile = 'smoke' if args.max_tokens is not None or args.ppl_mode == 'cached_decode' else 'full'
        if args.max_tokens is not None:
            if args.max_tokens != 4096 or ids.numel() < 4096:
                raise ValueError('Smoke windows require the first 4096 frozen tokens')
            ids = ids[:4096]
        if args.ppl_mode == 'cached_decode':
            if ids.numel() < 144:
                raise ValueError('Decode requires at least 144 frozen tokens')
            ids = ids[:144]
        ranges = scoring_ranges(ids.numel()) if args.ppl_mode == 'windows' else [{'input_query_position': 127 + i, 'predicted_target': 128 + i} for i in range(16)]
        env = environment()
        code = code_provenance(output)
        write_json(output / 'asset_manifest.json', assets)
        write_json(output / 'environment.json', env)
        resolved = {'algorithm': config, 'dtype': args.dtype, 'query_chunk': args.query_chunk, 'context_length': 2048, 'stride': 1024, 'prefill_tokens': 128, 'decode_targets': 16, 'ppl_mode': args.ppl_mode, 'profile': profile, 'trace_targets': args.trace_targets, 'model_path': str(Path(args.model_path).resolve())}
        write_json(output / 'resolved_config.json', resolved)
        method_manifest = {**config['attention_quantization'], 'baseline_kind': 'bf16_model_fp32_attention', 'model_dtype': 'bfloat16', 'qk_compute': 'fp32', 'quantization_at': 'attention_read', 'actual_matmul': 'fp32_after_dq' if method == 'intflash_qk8' else 'fp32', 'quantized_operands': 'qk' if method == 'intflash_qk8' else None, 'qk_codes_dtype': 'int8' if method == 'intflash_qk8' else None, 'source_delta': list(SOURCE_DELTA) if method == 'intflash_qk8' else [], 'input_cast_policy': INPUT_CAST_POLICY if method == 'intflash_qk8' else None}
        if full_config:
            method_manifest.update(attention_full_format=full_config,variant_id=full_config['variant_id'],selection_mode='top_p',selection_passes=1,final_operand_profile='decoded_full_format',final_operand_origin='stored_format_bytes',actual_matmul='fp32_after_actual_format_byte_decode',actual_pv='dense_zero_masked')
            write_json(output / 'full_format_manifest.json',method_manifest)
        if dense_config:
            method_manifest.update(attention_dense_format=dense_config, variant_id=dense_config['variant_id'], selection_mode='dense', selection_passes=0, final_operand_origin='stored_format_bytes', actual_matmul='fp32_after_actual_format_byte_decode', actual_pv='dense')
            write_json(output / 'dense_format_manifest.json', method_manifest)
        if slice_config:
            method_manifest.update(attention_slice=slice_config, variant_id=slice_config['variant_id'], baseline_name=slice_config['baseline_name'], selection_passes=1, final_operand_origin='original_float' if method == 'fp_common' else 'full_codes', actual_pv='dense_zero_masked')
            write_json(output / 'slice_manifest.json', method_manifest)
        write_json(output / 'method_manifest.json', method_manifest)
        gate_hash = None
        if profile == 'full':
            stage = 'gate_validation'
            if full_config:
                if not args.full_format_gate_report: raise ValueError('Full full-format TopP windows require --full-format-gate-report')
                from benchmark.attention_full_format.gate import validate_gate_report
                validate_gate_report(args.full_format_gate_report,args.model_path,args.assets,config,query_chunk=args.query_chunk)
                gate_hash=sha256_file(args.full_format_gate_report)
            elif dense_config:
                if not args.dense_format_gate_report: raise ValueError('Full dense-format windows require --dense-format-gate-report')
                from benchmark.attention_dense_format.gate import validate_gate_report
                validate_gate_report(args.dense_format_gate_report, args.model_path, args.assets, config, query_chunk=args.query_chunk)
                gate_hash = sha256_file(args.dense_format_gate_report)
            elif slice_config:
                if not args.slice_gate_report:
                    raise ValueError('Full TopP windows require --slice-gate-report')
                if slice_config.get('estimator') == 'format_msb':
                    from benchmark.attention_format.gate import validate_gate_report
                else:
                    from benchmark.attention_slice.gate import validate_gate_report
                validate_gate_report(args.slice_gate_report, args.model_path, args.assets, config, query_chunk=args.query_chunk)
                gate_hash = sha256_file(args.slice_gate_report)
            else:
                if not args.gate_report:
                    raise ValueError('Full windows require --gate-report')
                from benchmark.attention_baseline.gate import validate_gate_report
                validate_gate_report(args.gate_report, args.model_path, args.assets, config, query_chunk=args.query_chunk)
                gate_hash = sha256_file(args.gate_report)
        metrics = {'protocol_id': 'TQ-WT2-v1' if args.ppl_mode == 'windows' else 'TQ-WT2-decode-v1', 'profile': profile, 'mode': args.ppl_mode, 'baseline_kind': 'bf16_model_fp32_attention', 'method': method, 'backend': 'torch_qdq', 'model_dtype': args.dtype, 'qk_compute': 'fp32', 'cache_storage': 'original_float', 'tokens_sha256': assets['tokens_sha256'], 'subset_sha256': tokens_hash(ids), 'scoring_ranges_sha256': canonical_hash(ranges), 'asset_manifest_sha256': canonical_hash(assets), 'semantic_config_sha256': canonical_hash(config), 'capture_config_sha256': sha256_file(args.trace_targets) if args.trace_targets else None, 'code_sha256': code['code_sha256'], 'environment_sha256': env['environment_sha256'], 'gate_report_sha256': gate_hash, 'num_input_tokens': ids.numel(), 'context_length': 2048, 'stride': 1024, 'query_chunk': args.query_chunk, 'prefill_tokens': 128, 'decode_targets': 16, 'sum_nll': None, 'num_targets': None, 'ppl': None, 'timing_valid': not bool(args.trace_targets), 'peak_memory_bytes': None, 'dataset_identity_verified': assets.get('dataset_identity_verified', False), 'quantized_operands': method_manifest['quantized_operands'], 'qk_codes_dtype': method_manifest['qk_codes_dtype'], 'actual_matmul': method_manifest['actual_matmul'], 'source_delta': method_manifest['source_delta'], 'input_cast_policy': method_manifest['input_cast_policy']}
        if slice_config:
            metrics.update(variant_id=slice_config['variant_id'], baseline_name=slice_config['baseline_name'], attention_slice=slice_config, q_prefix_bits=slice_config['q_prefix_bits'], k_prefix_bits=slice_config['k_prefix_bits'], selection_passes=1, actual_pv='dense_zero_masked')
        write_json(output / 'metrics.json', metrics)
        stage = 'model_load'
        torch.cuda.synchronize()
        if full_config:
            metrics.update(variant_id=full_config['variant_id'],attention_full_format=full_config,selection_mode='top_p',selection_passes=1,final_operand_profile='decoded_full_format',final_operand_origin='stored_format_bytes')
        if dense_config:
            metrics.update(variant_id=dense_config['variant_id'], selection_mode='dense', selection_passes=0, final_operand_origin='stored_format_bytes')
        load_start = time.perf_counter()
        model = load_model(args.model_path)
        reset_sparse_config()
        enable_sparse_attention(model, sparse_config=config, query_chunk=args.query_chunk)
        if getattr(model, '_twilight_patch_count', None) != model.config.num_hidden_layers:
            raise ValueError('Not all decoder attention layers were patched')
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        counters = CallCounters()
        run_id = output.name
        if full_config:
            from twilight.pyimpl.full_format_trace import FullFormatTraceRecorder
            recorder_class=FullFormatTraceRecorder
        elif dense_config:
            from twilight.pyimpl.dense_format_trace import DenseFormatTraceRecorder
            recorder_class = DenseFormatTraceRecorder
        elif slice_config:
            if slice_config.get('estimator') == 'format_msb':
                from twilight.pyimpl.format_trace import FormatTraceRecorder
                recorder_class = FormatTraceRecorder
            else:
                from twilight.pyimpl.slice_trace import SliceTraceRecorder
                recorder_class = SliceTraceRecorder
        else:
            recorder_class = TraceRecorder
        recorder = recorder_class(output / 'trace', args.trace_targets, metadata={**metrics, 'run_id': run_id, 'model_file_hashes': assets['model_file_hashes'], 'config_sha256': canonical_hash(config), 'code_sha256': code['code_sha256'], 'algorithm': config}).attach(model) if args.trace_targets else None
        rows = []
        total_nll = 0.0
        num_targets = 0
        stage = 'scoring'
        torch.cuda.reset_peak_memory_stats()
        forward_seconds = 0.0
        trace_write_seconds = 0.0
        def invoke(input_ids, positions, global_positions, key_positions, window_id, stage_name, cache=None):
            nonlocal forward_seconds
            position_tensor = torch.tensor(positions, device='cuda', dtype=torch.long)
            tensor = input_ids[None].to('cuda')
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with forward_context(run_id=run_id, sample_id=0, window_id=window_id, stage=stage_name, global_query_positions=global_positions, position_ids=positions, key_positions=key_positions, expected_cache_length=len(key_positions), input_ids=input_ids.tolist(), cache_position=positions, counters=counters, recorder=recorder):
                result = model(input_ids=tensor, position_ids=position_tensor[None], cache_position=position_tensor, past_key_values=cache, use_cache=cache is not None, output_attentions=False, return_dict=True)
            if not torch.isfinite(result.logits).all():
                raise ValueError('Nonfinite model logits')
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - tick
            return result.logits
        def nll(logits, labels):
            nonlocal forward_seconds
            torch.cuda.synchronize()
            tick = time.perf_counter()
            value = float(F.cross_entropy(logits.float(), labels.to('cuda'), reduction='sum').double().cpu().item())
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - tick
            if not math.isfinite(value):
                raise ValueError('Nonfinite NLL')
            return value
        with torch.inference_mode():
            if args.ppl_mode == 'windows':
                for window_id, r in enumerate(ranges):
                    b, e, lo = r['b'], r['e'], r['lo']
                    logits = invoke(ids[b:e], list(range(e-b)), list(range(b,e)), list(range(e-b)), window_id, 'prefill')
                    loss = nll(logits[0, lo-b-1:e-b-1], ids[lo:e])
                    count = e-lo
                    row = {**r, 'window_id': window_id, 'sum_nll': loss, 'num_targets': count, 'logits_sha256': tensor_hash(logits)}
                    rows.append(row)
                    total_nll += loss
                    num_targets += count
                    log.write(json.dumps(row) + '\n')
                    del logits
            else:
                cache = DynamicCache()
                for i in range(16):
                    query = 127+i
                    inp = ids[:128] if i == 0 else ids[query:query+1]
                    positions = list(range(128)) if i == 0 else [query]
                    logits = invoke(inp, positions, positions, list(range(query+1)), None, 'prefill' if i == 0 else 'decode', cache)
                    loss = nll(logits[0,-1:], ids[query+1:query+2])
                    row = {'input_query_position': query, 'predicted_target': query+1, 'sum_nll': loss, 'num_targets': 1, 'logits_sha256': tensor_hash(logits)}
                    rows.append(row)
                    total_nll += loss
                    num_targets += 1
                    log.write(json.dumps(row) + '\n')
                    del logits
        stage = 'coverage_validation'
        expected_counts = {'prefill': len(ranges)} if args.ppl_mode == 'windows' else {'prefill': 1, 'decode': 15}
        expected_layers = {str(i) for i in range(model.config.num_hidden_layers)}
        if set(counters.counts) != expected_layers:
            raise ValueError(f'Attention coverage is incomplete: {counters.counts}')
        for layer, methods in counters.counts.items():
            if methods != {method: expected_counts}:
                raise ValueError(f'Layer {layer} method/stage coverage mismatch: {methods}')
        expected_targets = ids.numel()-1 if args.ppl_mode == 'windows' else 16
        if num_targets != expected_targets:
            raise ValueError('Scoring target coverage mismatch')
        diagnostics = counters.as_dict()['quantization_diagnostics']
        write_json(output / 'quantization_diagnostics.json', diagnostics)
        metrics['quantization_diagnostics'] = diagnostics
        coverage = {'quantization_diagnostics': diagnostics, 'input_cast_policy': method_manifest['input_cast_policy'], 'scoring_ranges': ranges, 'scoring_ranges_sha256': canonical_hash(ranges), 'num_targets': num_targets, 'layers': counters.counts, 'cache_lengths': counters.cache_lengths, 'expected_layer_count': model.config.num_hidden_layers, 'expected_calls_per_layer': expected_counts, 'all_layers_covered': True}
        write_json(output / 'coverage.json', coverage)
        if slice_config or dense_config or full_config:
            selection_stats = counters.slice_summary()
            write_json(output / 'selection_stats.json', selection_stats)
            write_json(output / 'costs.json', counters.slice_costs())
            metrics['selection_stats'] = selection_stats
        row_file = output / ('window_metrics.jsonl' if args.ppl_mode == 'windows' else 'target_metrics.jsonl')
        row_file.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows))
        if recorder:
            stage = 'trace_finalize'
            tick = time.perf_counter()
            recorder.finalize()
            recorder.close()
            trace_write_seconds = time.perf_counter()-tick
        perplexity = math.exp(total_nll/num_targets)
        if not math.isfinite(perplexity) or perplexity < 1:
            raise ValueError('Invalid perplexity')
        metrics.update(sum_nll=total_nll, num_targets=num_targets, ppl=perplexity, peak_memory_bytes=torch.cuda.max_memory_allocated(), model_load_seconds=load_seconds, warmup_seconds=0.0, forward_nll_seconds=forward_seconds, trace_write_seconds=trace_write_seconds, end_to_end_seconds=time.perf_counter()-start)
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'status.json', {'status': 'complete', 'stage': 'complete'})
        log.write(json.dumps({'status':'complete', **metrics})+'\n')
        print(json.dumps({'output_dir': str(output), 'method': method, 'num_targets':num_targets, 'ppl':perplexity}))
        return metrics
    except Exception as error:
        metrics.update(sum_nll=None, num_targets=None, ppl=None)
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'status.json', {'status': 'failed', 'stage': stage, 'reason': str(error), 'exception': type(error).__name__})
        log.write(traceback.format_exc())
        raise
    finally:
        log.close()


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
