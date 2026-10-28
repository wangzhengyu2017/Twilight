"""Shared immutable evidence and scoring primitives for the quality baseline."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE_SHA = 'f117a7872b2d350beb777fcd3e2f35176fa18b68'


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor):
    """Hash original dtype bytes, including BF16, without a float conversion."""
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def tokens_hash(ids):
    return hashlib.sha256(ids.detach().cpu().numpy().astype('<i8', copy=False).tobytes()).hexdigest()


def semantic_config(config):
    # Capture/output options are observational, never numerical configuration.
    return {k: v for k, v in config.items() if k not in {'trace', 'trace_targets', 'output_dir', 'capture'}}


def load_config(path):
    config = load_json(path)
    allowed = {'compressor', 'selector', 'weight_estimator', 'weight_pruner', 'skip_first_two_layers', 'use_estimated_weights_in_attn', 'attention_quantization'}
    if set(config) - {'attention_slice'} != allowed:
        raise ValueError(f'Unsupported/missing algorithm fields: {set(config) ^ allowed}')
    for field in ('compressor', 'selector', 'weight_estimator', 'weight_pruner'):
        if config[field] != {'type': 'none'}:
            raise ValueError(f'{field} must be disabled')
    if config['skip_first_two_layers'] or config['use_estimated_weights_in_attn']:
        raise ValueError('All layers must use the common complete attention path')
    q = config['attention_quantization']
    if set(q) != {'method', 'backend', 'scope', 'cache_storage', 'parameters', 'source_revision'}:
        raise ValueError('Unknown attention_quantization fields')
    if q['method'] not in ('fp_common', 'intflash_qk8') or q['backend'] != 'torch_qdq' or q['scope'] != 'prefill_and_decode' or q['cache_storage'] != 'original_float':
        raise ValueError('Unsupported attention protocol')
    if q['method'] == 'fp_common':
        if q['parameters'] or q['source_revision'] is not None:
            raise ValueError('FP method cannot have quantizer parameters')
    elif q['parameters'] != {'group_axis': 'head_token', 'quantizer_input_dtype': 'float16', 'source_input_cast': 'bf16_to_fp16', 'scale_dtype': 'source', 'rounding': 'torch_round', 'zero_group': 'scale_one'} or q['source_revision'] != '7f7bfb00bcd26b2cef49e7783f51ef610e05abf7':
        raise ValueError('INT source/quantizer parameters do not match the fixed protocol')
    if 'attention_slice' in config:
        from twilight.pyimpl.quant_attention.slice_backend import validate_slice_config
        validate_slice_config(config['attention_slice'], q['method'])
    return config


def scoring_ranges(length, context_length=2048, stride=1024):
    if length < 2:
        raise ValueError('At least two tokens are required')
    if (context_length, stride) != (2048, 1024):
        raise ValueError('TQ-WT2-v1 requires C=2048 S=1024')
    covered = 1
    result = []
    for b in range(0, length, stride):
        e = min(b + context_length, length)
        lo = max(covered, b + 1)
        if lo < e:
            result.append({'b': b, 'e': e, 'lo': lo, 'target_end': e})
        covered = e
        if e == length:
            break
    assert sum(r['target_end'] - r['lo'] for r in result) == length - 1
    return result


def load_tokens(path):
    import torch
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or set(data) != {'input_ids'}:
        raise ValueError('tokens.pt must contain only input_ids')
    ids = data['input_ids']
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.ndim != 1 or ids.numel() < 2:
        raise ValueError('Frozen input_ids must be one-dimensional CPU int64 with >=2 tokens')
    return ids.contiguous()


def validate_model(model_path):
    root = Path(model_path).resolve()
    cfg = load_json(root / 'config.json')
    if cfg.get('model_type') != 'llama' or cfg.get('quantization_config'):
        raise ValueError('An unquantized Llama HF checkpoint is required')
    if cfg.get('pretraining_tp', 1) != 1 or cfg.get('sliding_window') is not None:
        raise ValueError('Unsupported pretraining_tp/sliding_window')
    if (cfg.get('num_hidden_layers'), cfg.get('num_attention_heads'), cfg.get('num_key_value_heads'), cfg.get('hidden_size')) != (32, 32, 8, 4096):
        raise ValueError('Expected 32-layer Llama-3.1-8B (Hq=32,Hkv=8,D=128); revise targets explicitly for another architecture')
    indices = list(root.glob('*.index.json'))
    weights = list(root.glob('*.safetensors')) or list(root.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError('Model weights are missing')
    for index in indices:
        mapping = load_json(index).get('weight_map')
        if not mapping or any(not (root / p).is_file() for p in set(mapping.values())):
            raise ValueError(f'Incomplete weight shards: {index}')
    return cfg


def validate_assets(model_path, tokens, assets):
    manifest = load_json(assets) if not isinstance(assets, dict) else dict(assets)
    model_path = Path(model_path).resolve()
    cfg = validate_model(model_path)
    ids = load_tokens(tokens)
    required = {'dataset_id': 'Salesforce/wikitext', 'dataset_config': 'wikitext-2-raw-v1', 'split': 'test', 'tokens_hash_encoding': 'contiguous little-endian int64'}
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f'Asset {key} mismatch')
    if manifest.get('tokenizer_settings') != {'add_special_tokens': False, 'chat_template_applied': False, 'truncation': False}:
        raise ValueError('Unsupported frozen tokenizer settings')
    if manifest.get('num_tokens') != ids.numel() or manifest.get('tokens_file_sha256') != sha256_file(tokens) or manifest.get('tokens_sha256') != tokens_hash(ids):
        raise ValueError('Frozen token length/file/content hash mismatch')
    if ids.min().item() < 0 or ids.max().item() >= cfg['vocab_size']:
        raise ValueError('Token outside model vocabulary')
    hashes = manifest.get('model_file_hashes', {})
    actual_files = sorted(p for p in model_path.rglob('*') if p.is_file() and '.cache' not in p.relative_to(model_path).parts and p.suffix in {'.json', '.model', '.safetensors', '.bin', '.txt', '.tiktoken'})
    if set(hashes) != {p.relative_to(model_path).as_posix() for p in actual_files}:
        raise ValueError('Model/tokenizer file inventory changed')
    for relative, expected in hashes.items():
        if sha256_file(model_path / relative) != expected:
            raise ValueError(f'Model/tokenizer hash mismatch: {relative}')
    return manifest


def _command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def code_provenance(output_dir=None):
    """Hash tracked source plus untracked implementation; preserve an applicable patch."""
    import difflib
    head = _command(['git', 'rev-parse', 'HEAD'], ROOT)
    tracked = _command(['git', 'ls-files'], ROOT).splitlines()
    untracked = _command(['git', 'ls-files', '--others', '--exclude-standard'], ROOT).splitlines()
    # Runtime outputs and private/local agent state are not implementation sources.
    def source(name):
        p = Path(name)
        return p.suffix in {'.py', '.sh', '.json', '.toml', '.yaml', '.yml', '.txt'} and not any(x in p.parts for x in ('.codex', '.git', '__pycache__', 'outputs', 'results', '.venv')) and not name.startswith('PADE/')
    files = sorted(n for n in set(tracked + untracked) if source(n) and (ROOT / n).is_file())
    hashes = {name: sha256_file(ROOT / name) for name in files}
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD', '--', *[n for n in tracked if source(n)]], cwd=ROOT).decode()
    for name in sorted(set(untracked) & set(files)):
        content = (ROOT / name).read_text().splitlines(keepends=True)
        patch += f'diff --git a/{name} b/{name}\nnew file mode 100644\n'
        if content:
            patch += ''.join(difflib.unified_diff([], content, fromfile='/dev/null', tofile='b/' + name))
        else:
            patch += 'index 0000000..e69de29\n'
    manifest = {'baseline_sha': BASELINE_SHA, 'head': head, 'dirty': bool(_command(['git', 'status', '--porcelain'], ROOT)), 'files': hashes, 'modified_file_hashes': {n: hashes[n] for n in files if n in untracked or subprocess.call(['git', 'diff', '--quiet', 'HEAD', '--', n], cwd=ROOT) != 0}, 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest()}
    manifest['code_sha256'] = canonical_hash({'head': head, 'files': hashes})
    if output_dir:
        (Path(output_dir) / 'code.patch').write_text(patch)
        write_json(Path(output_dir) / 'code_manifest.json', manifest)
    return manifest


def configure_runtime():
    import random
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def environment():
    import torch
    import transformers
    freeze = _command([sys.executable, '-m', 'pip', 'freeze'])
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_runtime': torch.version.cuda, 'pip_freeze': freeze, 'pip_freeze_sha256': hashlib.sha256(freeze.encode()).hexdigest(), 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'float32_matmul_precision': torch.get_float32_matmul_precision(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'seed': 0}
    if torch.cuda.is_available():
        env['gpu'] = torch.cuda.get_device_name(0)
        env['gpu_capability'] = list(torch.cuda.get_device_capability(0))
        env['driver'] = _command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).splitlines()[0]
    env['environment_sha256'] = canonical_hash(env)
    return env


def load_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM
    validate_model(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the real-model baseline')
    return AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True).to('cuda').eval()
