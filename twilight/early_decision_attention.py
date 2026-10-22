"""Torch-only persistent early selector and selected-KV attention.

No dense QK is computed by select(). RoPE requires explicit heuristic consent.
Loadable by file path without importing Transformers, FlashAttention or Triton.
"""
import importlib.util
from pathlib import Path
import sys
import math
import time
import torch

_path = Path(__file__).resolve().parents[1] / "early decision" / "early_decision.py"
_name = "_twilight_input_early_decision"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
core = sys.modules[_name]


class EarlyDecisionAttention:
    """Single head, single sequence, causal append-only decoding.

    Row-layout weights [D,d]. Final K/V may be supplied post-RoPE. All floating
    tensors must have the same dtype/device as the frozen projection weights.
    Weight QDQ is supported by passing frozen dequantized weights; activation
    QDQ, biases and arbitrary attention masks are deliberately rejected.
    """
    @torch.no_grad()
    def __init__(self, wq, wk, *, rank, budget, mode="estimate_topk",
                 bound_mode="norm", rope_mode="none"):
        start = time.perf_counter()
        if mode not in ("estimate_topk", "topk_candidates"):
            raise ValueError("mode must be estimate_topk or topk_candidates")
        if rope_mode not in ("none", "pre_rope_heuristic"):
            raise ValueError("RoPE must be none or explicit pre_rope_heuristic")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        if wq.dtype != wk.dtype or wq.device != wk.device:
            raise ValueError("weights must share dtype/device")
        self.wq, self.wk = wq.detach().clone(), wk.detach().clone()
        self.mode, self.budget, self.bound_mode = mode, budget, bound_mode
        self.rope_mode = rope_mode
        self.model = core.fit_bilinear(wq, wk, rank)
        self.init_ms = (time.perf_counter() - start) * 1000
        self.reset()

    def reset(self):
        self.cache = self.model.new_cache(self.bound_mode)
        self.keys = self.values = None
        self.append_calls = 0
        self.last = {}

    def _tensor(self, x, width, name):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or
                x.shape[1] != width or x.shape[0] == 0 or
                x.dtype != self.wq.dtype or x.device != self.wq.device or
                not torch.isfinite(x).all()):
            raise ValueError(f"{name} must be finite [N,{width}] with weight dtype/device")

    @torch.no_grad()
    def append(self, x, *, keys=None, values):
        self._tensor(x, self.wq.shape[0], "x")
        if keys is None:
            if self.rope_mode != "none":
                raise ValueError("post-RoPE keys required in heuristic mode")
            keys = x @ self.wk
        self._tensor(keys, self.wq.shape[1], "keys")
        self._tensor(values, self.wq.shape[1], "values")
        if keys.shape[0] != x.shape[0] or values.shape[0] != x.shape[0]:
            raise ValueError("append row counts must agree")
        start = time.perf_counter()
        self.cache.append(x)
        self.keys = keys.clone() if self.keys is None else torch.cat((self.keys, keys))
        self.values = values.clone() if self.values is None else torch.cat((self.values, values))
        self.append_calls += 1
        self.last["append_ms"] = (time.perf_counter() - start) * 1000

    @torch.no_grad()
    def select(self, x_query):
        self._tensor(x_query, self.wq.shape[0], "query")
        if x_query.shape[0] != 1:
            raise ValueError("one causal query at a time")
        start = time.perf_counter()
        bounds = self.cache.bounds(x_query[0])
        k = min(self.budget, bounds.estimate.numel())
        if self.mode == "estimate_topk":
            mask = torch.zeros_like(bounds.estimate, dtype=torch.bool)
            mask[torch.topk(bounds.estimate, k).indices] = True
        else:
            mask = core.topk_candidates(bounds, k).mask
        self.last["selection_ms"] = (time.perf_counter() - start) * 1000
        return mask, bounds

    @torch.no_grad()
    def attend(self, x_query, *, query=None, bias=None):
        if bias is not None:
            raise ValueError("attention bias/mask is unsupported")
        mask, bounds = self.select(x_query)
        start = time.perf_counter()
        if query is None:
            if self.rope_mode != "none":
                raise ValueError("post-RoPE query required in heuristic mode")
            query = x_query @ self.wq
        self._tensor(query, self.wq.shape[1], "projected query")
        if query.shape[0] != 1:
            raise ValueError("one query at a time")
        selected_keys, selected_values = self.keys[mask], self.values[mask]
        logits = query @ selected_keys.T / math.sqrt(self.wq.shape[1])
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        output = weights @ selected_values
        self.last["selected_qk_av_ms"] = (time.perf_counter() - start) * 1000
        self.last["selected"] = int(mask.sum())
        return output, mask, bounds

    def costs(self):
        """Logical operation counts, NOT measured hardware speedups.

        Shared original Q/K/V projections excluded from BOTH incremental paths.
        Includes extra metadata projections, scan, selected QK and AV. QR/SVD
        setup reported separately via init_ms, not treated as free warm work.
        """
        n, d, D, r = len(self.cache.norms), self.wq.shape[1], self.wq.shape[0], self.model.rank
        tail = self.model.left_tail.shape[1] if self.bound_mode == "tail" else 0
        b = int(self.last.get("selected", min(n, self.budget)))
        return {"warm_step_logical_mac": 2 * D * (r + tail) + n * r + 2 * b * d,
                "dense_qk_av_logical_mac": 2 * n * d,
                "metadata_bytes": self.cache.storage_bytes,
                "factor_bytes": self.model.storage_bytes,
                "frozen_weight_clone_bytes": (self.wq.numel()+self.wk.numel())*self.wq.element_size(),
                "actual_kv_cache_bytes": (self.keys.numel()+self.values.numel())*self.keys.element_size(),
                "selected_kv_read_bytes": 2 * b * d * self.wq.element_size(),
                "selector_metadata_read_bytes": self.cache.storage_bytes,
                "dense_kv_read_bytes": 2 * n * d * self.wq.element_size(),
                "cache_append_copy_bytes": 2 * n * d * self.wq.element_size() + self.cache.storage_bytes,
                "copy_accounting": "torch.cat destination write bytes per append (KV+metadata); source reads additional; temporary selection gather excluded from logical bytes but timed",
                "non_mac_work": "norms, validation/conversions, softmax, topk comparisons, allocation and cat/gather copies included in CPU time",
                "selection_comparison_complexity": "topk implementation dependent; excluded from MAC, included in CPU timing"}


def enable_early_decision(module, *, rank, budget, mode="estimate_topk",
                          bound_mode="norm", rope_mode):
    """Opt in on an already Twilight-patched Llama/Mistral attention module.

    Explicit pre_rope_heuristic acknowledges loss of all score-bound guarantees.
    New sequence: call module.early_decision_reset() before the next prefill.
    """
    if rope_mode != "pre_rope_heuristic":
        raise ValueError("Llama/Mistral runtime requires explicit pre_rope_heuristic")
    if (getattr(module, "sliding_window", None) is not None or
            getattr(getattr(module, "config", None), "sliding_window", None) is not None):
        raise ValueError("sliding-window runtime not supported")
    compressor = getattr(getattr(module, "state", None), "compressor", None)
    if compressor is not None and getattr(compressor, "name", "NONE") != "NONE":
        raise ValueError("prefill compression cannot be combined with early runtime")
    for name in ("q_proj", "k_proj", "v_proj"):
        if getattr(getattr(module, name), "bias", None) is not None:
            raise ValueError("projection biases not supported")
    d = module.head_dim
    if module.num_heads != module.num_key_value_heads * module.num_key_value_groups:
        raise ValueError("invalid GQA head mapping")
    heads = []
    for h in range(module.num_heads):
        kh = h // module.num_key_value_groups
        heads.append(EarlyDecisionAttention(
            module.q_proj.weight[h*d:(h+1)*d].T,
            module.k_proj.weight[kh*d:(kh+1)*d].T,
            rank=rank, budget=budget, mode=mode, bound_mode=bound_mode, rope_mode=rope_mode))
    module._early_decision_heads = heads
    module.early_decision_reset = lambda: [head.reset() for head in heads]
    module._early_decision_forward = lambda *args, **kwargs: _runtime_forward(module, *args, **kwargs)


@torch.no_grad()
def _runtime_forward(module, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
    # Import only at runtime: the shared adapter itself is torch-only.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    if attention_mask is not None or output_attentions or kwargs.get("position_embeddings") is not None:
        raise ValueError("early runtime supports no mask, no returned attention map, legacy rotary API only")
    if hidden_states.shape[0] != 1 or position_ids is None:
        raise ValueError("early runtime requires batch=1 and explicit position_ids")
    heads = module._early_decision_heads
    previous = len(heads[0].cache.norms)
    length = hidden_states.shape[1]
    expected = torch.arange(previous, previous + length, device=position_ids.device)[None]
    if not torch.equal(position_ids, expected):
        raise ValueError("positions must be contiguous from zero; reset early cache for a new sequence")
    if previous and not use_cache:
        raise ValueError("continuation requires use_cache=True")
    if use_cache and (past_key_value is None or past_key_value.get_usable_length(length, module.layer_idx) != previous):
        raise ValueError("external KV cache must match persistent early cache; reset both for a new sequence")
    d = module.head_dim
    q = module.q_proj(hidden_states).view(1, length, module.num_heads, d).transpose(1, 2)
    k = module.k_proj(hidden_states).view(1, length, module.num_key_value_heads, d).transpose(1, 2)
    v = module.v_proj(hidden_states).view(1, length, module.num_key_value_heads, d).transpose(1, 2)
    cos, sin = module.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
    if use_cache:
        past_key_value.update(k, v, module.layer_idx, {"sin": sin, "cos": cos})
    outputs = []
    for t in range(length):
        row = hidden_states[0, t:t+1]
        per_head = []
        for h, head in enumerate(heads):
            head.append(row, keys=k[0, h, t:t+1], values=v[0, h, t:t+1])
            out, mask, _ = head.attend(row, query=q[0, h, t:t+1])
            head.last["selected"] = int(mask.sum())
            per_head.append(out)
        outputs.append(torch.cat(per_head, dim=-1))
    result = module.o_proj(torch.cat(outputs, dim=0)[None])
    if not use_cache:
        module.early_decision_reset()
    return result, None, past_key_value
