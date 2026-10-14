### attention_forward

> [!IMPORTANT]
>
> `attention_forward()` 是 Twilight 用来替换 Hugging Face 原始 `LlamaAttention.forward()` / `MistralAttention.forward()` 的函数。
>
> - **Prefill 阶段**：用 Flash Attention 做完整 Attention；如果启用 SnapKV，只压缩写入 KV Cache 的 K/V。
> - **Decode 阶段**：先算完整 Attention score，再按照 `Selector → Weight Estimator → Weight Pruner` 得到稀疏 mask，最后计算输出。
> - 这个文件是用于验证算法精度的 Python 实现，并不是真正高效的 sparse kernel，因为 Decode 时仍然先计算了完整 Attention score。

源码位置：[attention.py #60-339](./twilight/pyimpl/attention.py)

整体流程：

```python
hidden_states
      ↓
Q/K/V projection + RoPE + GQA
      ↓
根据 use_cache 更新 KV Cache
      ↓
判断 q_len
      │
      ├── q_len > 1：Prefill
      │       ├── 可选：SnapKV 压缩写入 Cache 的 K/V
      │       ├── Flash Attention 计算完整输出
      │       └── return，不执行下面的 token selector / pruner / 统计
      │
      └── q_len == 1：Decode
              ├── 计算完整 attn score
              ├── Index Selector：得到候选 mask（B0）
              ├── Weight Estimator：估算候选 token 的 score
              ├── Weight Pruner：进一步剪枝（B1）
              ├── 记录 budget / score
              ├── mask 掉未选择 token
              ├── softmax
              ├── attn_weights @ V
              └── o_proj + return
```

#### 1. 函数参数和返回值（#60-69）

```python
def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor]],
]:
```

| 参数 | 作用 | 在这个函数中的使用方式 |
| --- | --- | --- |
| `self` | 当前层的 `LlamaAttention` / `MistralAttention` 对象 | 读取 Q/K/V/O projection、head 数量以及 Twilight 动态添加的 `state` |
| `hidden_states` | 当前层输入 | 形状为 `[bs, q_len, hidden_size]` |
| `attention_mask` | padding / causal mask | Prefill 传入 Flash Attention；Decode 当前直接要求它必须为 `None` |
| `position_ids` | token 的位置编号 | 用于生成 RoPE 的 `cos`、`sin` |
| `past_key_value` | 每一层历史 token 的 KV Cache | `use_cache=True` 时调用 `update()`，Decode 时返回包含全部历史的 K/V |
| `output_attentions` | 是否返回 Attention 权重 | Decode 支持；Prefill 分支存在未定义变量问题，见后面的“当前实现注意点” |
| `use_cache` | 是否使用 KV Cache | 推理通常为 `True` |
| `**kwargs` | 兼容额外参数 | 当前函数没有主动读取 |

返回：

```python
return attn_output, attn_weights, past_key_value
```

- `attn_output`：当前 Attention 层输出，形状 `[bs, q_len, hidden_size]`；
- `attn_weights`：如果 `output_attentions=False`，返回 `None`；
- `past_key_value`：更新后的 Cache 对象。`update()` 是原地更新，因此这里返回的通常还是传进来的那个对象引用。

#### 2. 清理 CUDA allocator 的空闲缓存（#70）

```python
torch.cuda.empty_cache()
```

- 它释放的是 PyTorch caching allocator 当前没有被 Tensor 使用的显存块；
- 它**不会删除仍然存活的 Tensor**，也不会清空 KV Cache；
- 该函数是每一层 Attention 都要调用的，因此这里可能带来比较明显的同步/分配开销；
- 从功能上看，这一行和 sparse attention 算法没有直接关系，更像是开发阶段为了缓解显存压力添加的操作。

#### 3. 读取输入形状（#72、#78）

```python
bsz, q_len, _ = hidden_states.size()

# 后面又读取了一次
bs, q_len, _ = hidden_states.size()
```

`hidden_states` 的形状：

```python
[bs, q_len, hidden_size]
```

- `bs` / `bsz`：batch size；
- `q_len`：本次 forward 输入多少个 token；
- `_`：`hidden_size`，此处暂时不使用，所以用 `_` 接收。

这里的 `bs` 和 `bsz` 完全相同，属于重复赋值：

```python
bs == bsz
```

`q_len` 还是本函数区分 Prefill 和 Decode 的关键：

```python
q_len > 1   # Prefill
q_len == 1  # Decode
```

#### 4. Q/K/V projection 和形状变换（#82-96）

```python
query_states = (
    self.q_proj(hidden_states)
    .view(bs, q_len, self.num_heads, self.head_dim)
    .transpose(1, 2)
)

key_states = (
    self.k_proj(hidden_states)
    .view(bs, q_len, self.num_key_value_heads, self.head_dim)
    .transpose(1, 2)
)

value_states = (
    self.v_proj(hidden_states)
    .view(bs, q_len, self.num_key_value_heads, self.head_dim)
    .transpose(1, 2)
)
```

以 Q 为例：

```python
hidden_states
# [bs, q_len, hidden_size]

self.q_proj(hidden_states)
# [bs, q_len, num_heads * head_dim]

.view(bs, q_len, self.num_heads, self.head_dim)
# [bs, q_len, num_heads, head_dim]

.transpose(1, 2)
# [bs, num_heads, q_len, head_dim]
```

K/V 同理，但是它们首先使用 `num_key_value_heads`：

```python
Q: [bs, num_heads,           q_len, head_dim]
K: [bs, num_key_value_heads, q_len, head_dim]
V: [bs, num_key_value_heads, q_len, head_dim]
```

如果是普通 MHA：

```python
num_heads == num_key_value_heads
```

如果是 GQA：

```python
num_heads > num_key_value_heads
```

#### 5. 计算理论 KV 序列长度（#98-116）

```python
kv_seq_len = key_states.shape[-2]

if past_key_value is not None:
    if self.layer_idx is None:
        raise ValueError(...)

    if hasattr(self, "kv_seq_len"):
        if self.kv_seq_len != 0:
            kv_seq_len += self.kv_seq_len
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len,
                self.layer_idx,
            )
    else:
        kv_seq_len += past_key_value.get_usable_length(
            kv_seq_len,
            self.layer_idx,
        )
```

刚计算完 projection 时：

```python
kv_seq_len = 当前输入 token 数
```

Decode 时还存在历史 Cache，因此理论总长度应该是：

```python
总 KV 长度 = 历史可用 KV 长度 + 当前输入长度
```

`self.layer_idx` 用于告诉 Cache 当前正在更新第几层。如果使用 Cache 却没有 `layer_idx`，代码主动抛出错误。

```python
past_key_value.get_usable_length(
    当前输入长度,
    当前层编号,
)
```

返回当前层可以使用的历史 Cache 长度。

`hasattr(self, "kv_seq_len")` 是为 SnapKV 兼容代码留下的分支：

- 如果模块有额外的 `kv_seq_len`，优先使用这个属性记录压缩前/逻辑上的历史长度；
- 否则直接向 Cache 查询可用长度。

不过在当前 `attention_forward()` 后续代码中，局部变量 `kv_seq_len` 并没有再参与计算，RoPE 直接使用 `position_ids`。因此这部分更接近从 Hugging Face / SnapKV 代码保留下来的兼容逻辑。

#### 6. 给 Q/K 应用 RoPE（#118-122）

```python
cos, sin = self.rotary_emb(
    value_states,
    position_ids,
)

query_states, key_states = apply_rotary_pos_emb(
    query_states,
    key_states,
    cos,
    sin,
    position_ids,
)
```

RoPE 只作用于 Q 和 K，不作用于 V：

```python
Q ──RoPE──> Q_rotated
K ──RoPE──> K_rotated
V ─────────> V
```

位置编码的作用体现在 Q/K 点积中：

```python
Q_rotated @ K_rotated.T
```

`self.rotary_emb(...)` 负责生成对应位置的 `cos` 和 `sin`，`apply_rotary_pos_emb(...)` 再将旋转作用到 Q/K。

#### 7. GQA：把 K/V head 扩展到 Q head 数（#124-126）

```python
key_states = repeat_kv(
    key_states,
    self.num_key_value_groups,
)

value_states = repeat_kv(
    value_states,
    self.num_key_value_groups,
)
```

其中：

```python
num_key_value_groups = num_heads // num_key_value_heads
```

例如：

```python
num_heads = 32
num_key_value_heads = 8
num_key_value_groups = 4
```

每一个 KV head 会重复 4 次：

```python
K/V: [bs, 8,  q_len, head_dim]
              ↓ repeat_kv
K/V: [bs, 32, q_len, head_dim]
```

这样才能和 Q 做逐 head 的矩阵乘法：

```python
Q: [bs, 32, q_len, head_dim]
K: [bs, 32, kv_len, head_dim]
```

#### 8. 根据 `use_cache` 更新 KV Cache（#128-147）

```python
if use_cache:
    cache_kwargs = {
        "sin": sin,
        "cos": cos,
    }

    if q_len == 1:
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )
    else:
        cache_key, cache_value = key_states, value_states

        if self.state.compressor == CompressorType.SNAP_KV:
            cache_key, cache_value = snap_kv_compression(
                query_states,
                key_states,
                value_states,
                **self.state.compressor_args,
            )

        past_key_value.update(
            cache_key,
            cache_value,
            self.layer_idx,
            cache_kwargs,
        )
```

首先构造额外 Cache 参数：

```python
cache_kwargs = {"sin": sin, "cos": cos}
```

这主要用于兼容可能需要 RoPE 信息的 Cache 实现。

##### Decode：追加当前 token，取回完整 K/V（#133-136）

```python
key_states, value_states = past_key_value.update(
    key_states,
    value_states,
    self.layer_idx,
    cache_kwargs,
)
```

Decode 每次只有一个新 token：

```python
当前 K/V: [bs, num_heads, 1, head_dim]
历史 K/V: [bs, num_heads, past_len, head_dim]
```

`update()` 的脑内翻译：

```python
new_cache_K = concat(history_K, current_K, dim=token_dim)
new_cache_V = concat(history_V, current_V, dim=token_dim)

return new_cache_K, new_cache_V
```

更新之后：

```python
key_states:   [bs, num_heads, past_len + 1, head_dim]
value_states: [bs, num_heads, past_len + 1, head_dim]
```

所以后面的当前 Q 可以和全部历史 K 计算 Attention。

##### Prefill：可选压缩“写入 Cache”的 K/V（#137-147）

```python
cache_key, cache_value = key_states, value_states

if self.state.compressor == CompressorType.SNAP_KV:
    cache_key, cache_value = snap_kv_compression(
        query_states,
        key_states,
        value_states,
        **self.state.compressor_args,
    )

past_key_value.update(
    cache_key,
    cache_value,
    self.layer_idx,
    cache_kwargs,
)
```

这里特意创建了两组变量：

```python
key_states / value_states    # 当前 Prefill 计算 Attention 使用
cache_key / cache_value      # 写入 KV Cache 使用
```

如果启用 SnapKV：

```python
完整 K/V ───────────────> Prefill Attention
   │
   └── snap_kv_compression
              ↓
         压缩 K/V ──────> KV Cache
```

因此：

- 当前 Prompt 的 Prefill 输出仍然由完整 K/V 计算；
- 压缩只影响后续 Decode 能从 Cache 中看到哪些历史 token；
- `past_key_value.update()` 的返回值在 Prefill 分支没有接收，因为 Prefill 本次 Attention 不需要使用压缩后的 Cache。

#### 9. Prefill：Flash Attention 完整计算并提前返回（#156-182）

```python
if q_len > 1:
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=0.0,
        sliding_window=getattr(
            self,
            "sliding_window",
            None,
        ),
        use_top_left_mask=False,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(
        bsz,
        q_len,
        -1,
    ).contiguous()

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
```

Flash Attention 要求的 layout 是：

```python
[bs, seq_len, num_heads, head_dim]
```

当前 Q/K/V 是：

```python
[bs, num_heads, seq_len, head_dim]
```

所以需要：

```python
.transpose(1, 2)
```

形状变化：

```python
[bs, num_heads, q_len, head_dim]
                  ↓ transpose(1, 2)
[bs, q_len, num_heads, head_dim]
```

Flash Attention 直接返回各个 head 的输出：

```python
attn_output: [bs, q_len, num_heads, head_dim]
```

随后把后两维合并回 `hidden_size`：

```python
attn_output.reshape(bsz, q_len, -1)
# [bs, q_len, num_heads * head_dim]
# [bs, q_len, hidden_size]
```

最后经过输出投影：

```python
attn_output = self.o_proj(attn_output)
```

最重要的是 #182：

```python
return attn_output, attn_weights, past_key_value
```

Prefill 在这里已经结束，不会继续执行 Decode 下方的：

```python
Index Selector
Weight Estimator
Weight Pruner
budget_info.update()
score_info.update()
```

#### 10. Decode 限制：不支持非空 `attention_mask`（#184-185）

```python
assert attention_mask is None
```

也就是说，当前 Python Decode 实现只接受：

```python
attention_mask = None
```

如果传入 padding mask 等非空 Tensor，会直接触发 `AssertionError`。

单 token Decode 中，当前 Q 位于序列最末尾，天然可以看到全部历史 token，所以通常不需要额外的 causal mask；但是 batch padding 等情况仍可能需要 mask，当前实现没有处理这些情况。

#### 11. Decode 先计算完整 Attention score（#187-190）

```python
attn_weights = torch.matmul(
    query_states,
    key_states.transpose(2, 3),
) / math.sqrt(self.head_dim)
```

Decode 时形状是：

```python
query_states: [bs, num_heads, 1,      head_dim]
key_states:   [bs, num_heads, kv_len, head_dim]
```

K 转置最后两维：

```python
key_states.transpose(2, 3)
# [bs, num_heads, head_dim, kv_len]
```

矩阵乘法：

```python
[bs, num_heads, 1, head_dim]
                @
[bs, num_heads, head_dim, kv_len]
                ↓
[bs, num_heads, 1, kv_len]
```

所以：

```python
attn_weights.shape == [bs, num_heads, 1, kv_len]
```

严格地说，此时还没有 softmax，它应当叫作：

```python
attention logits / attention scores
```

代码变量名虽然叫 `attn_weights`，但 #301 以后才是真正归一化的 Attention weights。

除以：

```python
sqrt(head_dim)
```

用于控制 QK 点积随维度增大而产生的数值幅度。

> [!NOTE]
>
> 到这里已经计算了每个历史 token 的完整 score，所以当前 `pyimpl` 并没有通过 sparse attention 节省这部分计算量。后面的 Selector/Pruner 是在完整 score 已经存在的基础上模拟稀疏结果，主要用于比较准确率。

#### 12. 决定从第几层开始稀疏化（#195-197）

```python
start_layer = (
    2
    if self.state.skip_first_two_layers
    else 0
)

if self.layer_id >= start_layer:
    ...
```

如果 config 中：

```json
"skip_first_two_layers": true
```

那么：

```python
start_layer = 2
```

| Attention 层 | 是否进入 Selector / Estimator / Pruner | 是否更新 budget / score |
| --- | --- | --- |
| Layer 0 | 否，使用完整 Attention | 否 |
| Layer 1 | 否，使用完整 Attention | 否 |
| Layer 2 及以后 | 是 | 开启统计对象时会更新 |

如果：

```json
"skip_first_two_layers": false
```

则：

```python
start_layer = 0
```

所有层都会进入该代码块。

注意：所有层无论是否跳过稀疏化，都会执行 `attention_forward()`；这里跳过的只是 `if self.layer_id >= start_layer` 内部的稀疏流水线。

#### 13. Index Selector：选出候选 token（#198-242）

Selector 的统一输出应当是：

```python
mask: torch.BoolTensor
mask.shape == attn_weights.shape
# [bs, num_heads, 1, kv_len]
```

语义：

```python
mask[..., token_id] == True   # 保留该 token
mask[..., token_id] == False  # 删除该 token
```

这一阶段先得到一个候选 token 集合，可以理解成 Twilight 流水线中的 `B0`。

##### `NONE`：所有 token 都保留（#199-200）

```python
if self.state.selector == IndexSelectorType.NONE:
    mask = torch.ones_like(
        attn_weights,
        dtype=torch.bool,
    )
```

得到全 `True` mask：

```python
[True, True, True, ..., True]
```

##### `STREAMING`：attention sink + 最近 token（#201-206）

```python
elif self.state.selector == IndexSelectorType.STREAMING:
    mask = streaming_selector(
        attn_weights,
        self.state.selector_args["token_budget"],
        self.state.selector_args["num_sinks"],
    )
```

`streaming_selector()` 内部：

```python
mask[..., :num_sinks] = True
mask[..., -token_budget:] = True
```

保留：

```text
[最前面的 sink token] + [最近的 token]
```

如果两段发生重叠，最终保留 token 数会小于 `num_sinks + token_budget`。

##### `QUEST`：按 chunk 估分并选择候选 chunk（#207-213）

```python
elif self.state.selector == IndexSelectorType.QUEST:
    mask = quest_selector(
        self,
        query_states,
        key_states,
        **self.state.selector_args,
    )
```

例如：

```python
self.state.selector_args = {
    "token_budget": 8192,
    "chunk_size": 16,
}
```

`**` 展开后等价于：

```python
mask = quest_selector(
    self,
    query_states,
    key_states,
    token_budget=8192,
    chunk_size=16,
)
```

Quest 将连续 token 分成 chunk，估算每个 chunk 的重要性，再选择 Top-K chunk。因为最后以完整 chunk 为单位展开，所以实际保留 token 数可能向上取整到 `chunk_size` 的倍数。

##### `ORACLE_TOPK`：用真实 score 直接选 Top-K（#214-218）

```python
elif self.state.selector == IndexSelectorType.ORACLE_TOPK:
    mask = oracle_topk_selector(
        attn_weights,
        self.state.selector_args["token_budget"],
    )
```

它直接从完整真实 score 中选择最大的 K 个 token：

```python
_, indices = attn_weights.topk(k=k, dim=-1)
mask.scatter_(dim=-1, index=indices, value=True)
```

因此它更像一个精度上界 / Oracle baseline，而不是可直接获得加速的近似 selector。

##### `TIDAL_DECODE`：部分层重选，其他层复用 mask（#219-226）

```python
elif self.state.selector == IndexSelectorType.TIDAL_DECODE:
    mask = tidal_decode_selector(
        self.state,
        attn_weights,
        self.state.selector_args["token_budget"],
        self.layer_id,
        self.state.selector_args["reselection_layers"],
    )
```

设计意图：

```text
reselection layer
      ↓
根据当前真实 score 重新 Top-K
      ↓
把 mask 保存到 local_state.td_mask_buffer
      ↓
后续层复用同一个 mask
```

它是 `LocalState` 存储跨层临时状态的例子。当前实现存在返回值接口问题，见后面的“当前实现注意点”。

##### `SPARQ`：根据当前 Q 的重要通道估分（#227-232）

```python
elif self.state.selector == IndexSelectorType.SPARQ:
    mask = sparq_selector(
        query_states,
        key_states,
        **self.state.selector_args,
    )
```

SparQ 的大致流程：

```text
当前 Q 取绝对值
      ↓
选择绝对值最大的 r 个 channel
      ↓
从 Q/K 中抽取相同 channel
      ↓
用局部 channel 计算 estimated score
      ↓
Top-K token → mask
```

##### `DS`：根据提前计算的通道排序估分（#233-242）

```python
elif self.state.selector == IndexSelectorType.DS:
    selector_args = self.state.selector_args.copy()
    selector_args.pop("config_path")
    selector_args.pop("selected_channel")

    mask = double_sparse_selector(
        query_states,
        key_states,
        self.sorted_channel,
        **selector_args,
    )
```

`config_path` 和 `selected_channel` 只用于 `_enable_sparse_attention()` 初始化每层的 `self.sorted_channel`，执行 selector 时不再需要，所以先从参数副本中删除。

```python
self.state.selector_args.copy()
```

保证 `pop()` 不会改变共享状态中的原始字典。

DS 的大致流程：

```text
self.sorted_channel：每个 head 预计算的重要 channel
      ↓
从 Q/K 抽取 r 个 channel
      ↓
可选：量化抽取后的 K
      ↓
计算 estimated score
      ↓
Top-K token → mask
```

#### 14. Weight Estimator：估算 Attention score（#244-265）

Selector 解决的是：

```text
哪些 token 值得进入候选集合？
```

Weight Estimator 解决的是：

```text
能否用更低精度、更低成本的 K 估算这些候选 token 的 score？
```

##### `NONE`：直接使用真实 score（#245-246）

```python
if self.state.weight_estimator == WeightEstimatorType.NONE:
    estimated_weights = attn_weights
```

这里不是复制 Tensor，而是让两个变量指向同一个对象：

```python
estimated_weights is attn_weights
```

##### `MIN_MAX_QUANT`：对 K 做逐 token 非对称 fake quant（#247-255）

```python
elif (
    self.state.weight_estimator
    == WeightEstimatorType.MIN_MAX_QUANT
):
    quantized_key = min_max_per_token_quant_kv(
        key_states,
        self.state.weight_estimator_args["quant_bit"],
    )

    estimated_weights = torch.matmul(
        query_states,
        quantized_key.transpose(2, 3),
    ) / math.sqrt(self.head_dim)

    estimated_weights[~mask] = torch.tensor(
        torch.finfo(attn_weights.dtype).min
    )
```

流程：

```text
K（浮点）
   ↓ 每个 token 在 channel 维度做 min-max Q/DQ
quantized_key（仍是浮点 Tensor，只是数值落在量化格点）
   ↓
Q @ quantized_key.T
   ↓
estimated_weights
```

随后把 Selector 没选中的位置设为最小值，避免它们在 Pruner 中再次被选中。

##### `MAX_QUANT`：对 K 做对称 fake quant（#256-265）

```python
elif self.state.weight_estimator == WeightEstimatorType.MAX_QUANT:
    quantized_key = max_per_token_quant_kv(
        key_states,
        self.state.weight_estimator_args["quant_bit"],
        self.state.weight_estimator_args["smooth_k"],
    )

    estimated_weights = torch.matmul(
        query_states,
        quantized_key.transpose(2, 3),
    ) / math.sqrt(self.head_dim)

    estimated_weights[~mask] = torch.tensor(
        torch.finfo(attn_weights.dtype).min
    )
```

`smooth_k=True` 时，会先沿 token 维减去最小值；随后用最大绝对值计算 scale。

这两个 Quant 函数返回的都是反量化后的浮点 Tensor，因此属于 Fake Quant：

```text
FP K → Quantize → Dequantize → FP quantized_key
```

#### 15. Weight Pruner：进一步动态剪枝（#267-281）

Weight Pruner 在 Selector 的候选集合上进一步过滤 token，可以理解成：

```text
Selector 得到候选预算 B0
            ↓
Weight Estimator 估分
            ↓
Weight Pruner 得到实际预算 B1
```

##### `THRESHOLD`（#269-273）

```python
if self.state.weight_pruner == WeightPrunerType.THRESHOLD:
    pruned_mask = elementwise_threshold(
        estimated_weights,
        self.state.weight_pruner_args["threshold"],
    )
```

对应函数的真实判断条件是：

```python
first_weight = estimated_weights[:, :, :, 0:1]
pruned_mask = estimated_weights > first_weight - threshold
```

所以这里的 Threshold 不是：

```python
estimated_weights > 一个固定绝对阈值
```

而是对每个 batch/head，以第 0 个 token 的 score 为参照：

```python
当前 token score > token-0 score - threshold
```

##### `TOP_P`（#274-278）

```python
elif self.state.weight_pruner == WeightPrunerType.TOP_P:
    pruned_mask = top_p_unnormalized(
        estimated_weights,
        self.state.weight_pruner_args["threshold"],
    )
```

`top_p_unnormalized()` 先对 score 做 FP32 softmax：

```python
normalized_weights = nn.functional.softmax(
    estimated_weights,
    dim=-1,
    dtype=torch.float32,
)
```

再调用 CUDA kernel，保留累计概率达到阈值所需的高概率 token。

##### Selector mask 与 Pruner mask 求交集（#280-281）

```python
if self.state.weight_pruner != WeightPrunerType.NONE:
    mask = mask & pruned_mask
```

布尔与：

```python
最终保留 = Selector 选中 AND Pruner 选中
```

例子：

```python
mask        = [True,  True,  False, True]
pruned_mask = [False, True,  True,  True]

mask & pruned_mask
#             ↓
#          [False, True, False, True]
```

因此最终集合不会超出 Selector 的候选集合。

> [!NOTE]
>
> Quant Estimator 分支在 Pruner 前已经把 `~mask` 位置设成最小值；但是 `WeightEstimatorType.NONE` 分支没有提前 mask，而是直接令 `estimated_weights = attn_weights`。所以 `NONE + TOP_P` 会先在完整 token 分布上计算 Top-P，再与 Selector mask 求交。这和“仅在候选 token 内重新归一化后做 Top-P”并不完全等价。

#### 16. 更新 budget / score 统计（#283-293）

```python
if self.state.budget_info is not None:
    self.state.budget_info.update(
        self.layer_id,
        mask,
    )

if self.state.score_info is not None:
    self.state.score_info.update(
        self.layer_id,
        attn_weights,
        mask,
    )
```

注意更新发生在：

```python
mask = mask & pruned_mask
```

之后，所以统计的是最终 `B1 mask`，不是 Selector 刚产生的 `B0 mask`。

`budget_info.update()`：

```python
head_budgets = torch.sum(mask, dim=-1).flatten()
```

统计每个 batch / head 最终有多少个 `True`，即保留多少 token。

`score_info.update()`：

```python
norm_weights = softmax(attn_weights)
norm_weights[~mask] = 0
score_sum = norm_weights.sum(dim=-1)
```

使用的 `attn_weights` 是 #187-190 计算的**真实完整 score**，因此它衡量：

> 最终 mask 覆盖了原始完整 Attention 的多少概率质量。

统计更新的精确条件：

```text
q_len == 1
   AND
self.layer_id >= start_layer
   AND
budget_info / score_info 不是 None
```

Prefill 在 #182 已经 return，因此不会更新统计；默认 `skip_first_two_layers=True` 时，第 0、1 层也不会更新统计。

#### 17. 决定最终使用真实 score 还是估算 score（#295-296）

```python
if self.state.use_estimated_weights_in_attn:
    attn_weights = estimated_weights
```

如果 config：

```json
"use_estimated_weights_in_attn": false
```

那么 Estimator 只用来决定“哪些 token 被保留”，最终保留 token 之间仍使用真实 score：

```text
estimated_weights → Pruner → mask
真实 attn_weights + mask → 最终 Attention
```

如果设置为 `true`：

```text
estimated_weights → Pruner → mask
estimated_weights + mask → 最终 Attention
```

也就是说，量化误差不仅影响 token 选择，还会直接影响最终 Attention 概率。

`score_info.update()` 在替换前执行，所以即使该值为 `true`，评测时仍然用真实完整 score 计算被保留的概率质量。

#### 18. mask 掉未选择 token（#298-299）

```python
attn_weights[~mask] = torch.tensor(
    torch.finfo(attn_weights.dtype).min
)
```

其中：

```python
~mask
```

表示布尔取反：

```python
mask  = [True,  False, True]
~mask = [False, True,  False]
```

`torch.finfo(dtype).min` 是该浮点类型能够表示的最小有限值。例如它不是数学上的 `-∞`，但是一个绝对值很大的负数。

经过 softmax：

```python
exp(非常大的负数) ≈ 0
```

所以被 mask 的 token 最终 Attention 概率近似为 0。

#### 19. FP32 softmax，再转回 Q 的 dtype（#301-304）

```python
attn_weights = nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
).to(query_states.dtype)
```

形状保持不变：

```python
[bs, num_heads, 1, kv_len]
```

先用 FP32 计算 softmax：

```python
dtype=torch.float32
```

可以减小 FP16/BF16 softmax 的数值误差；随后：

```python
.to(query_states.dtype)
```

再转回模型原来的数据类型，以便和 V 做矩阵乘法。

此时变量 `attn_weights` 才真正是归一化概率：

```python
attn_weights.sum(dim=-1) ≈ 1
```

#### 20. Attention weights 与 V 相乘（#306）

```python
attn_output = torch.matmul(
    attn_weights,
    value_states,
)
```

形状：

```python
attn_weights: [bs, num_heads, 1,      kv_len]
value_states: [bs, num_heads, kv_len, head_dim]
                              ↓
attn_output:  [bs, num_heads, 1,      head_dim]
```

数学意义：

```python
当前 head 输出 = Σ(每个历史 token 的 Attention 概率 × 对应 V)
```

被 mask 的 token 权重接近 0，因此不会对输出产生明显贡献。

#### 21. 检查输出形状（#325-329）

```python
if attn_output.size() != (
    bsz,
    self.num_heads,
    q_len,
    self.head_dim,
):
    raise ValueError(...)
```

期望形状：

```python
[bs, num_heads, q_len, head_dim]
```

Decode 时 `q_len=1`：

```python
[bs, num_heads, 1, head_dim]
```

如果矩阵乘法或 head 配置产生了错误形状，这里提前给出明确异常。

#### 22. 合并所有 Attention Head 并做输出投影（#331-334）

```python
attn_output = attn_output.transpose(1, 2)
attn_output = attn_output.reshape(
    bsz,
    q_len,
    self.hidden_size,
)

attn_output = self.o_proj(attn_output)
```

形状变化：

```python
[bs, num_heads, q_len, head_dim]
                ↓ transpose(1, 2)
[bs, q_len, num_heads, head_dim]
                ↓ reshape
[bs, q_len, num_heads * head_dim]
                ↓ num_heads * head_dim = hidden_size
[bs, q_len, hidden_size]
                ↓ o_proj
[bs, q_len, hidden_size]
```

`o_proj` 用于混合不同 Attention Head 的结果，并投影回模型隐藏维度。

#### 23. 按需返回 Attention 权重（#336-339）

```python
if not output_attentions:
    attn_weights = None

return attn_output, attn_weights, past_key_value
```

一般生成时不需要保存完整 Attention matrix：

```python
output_attentions = False
```

因此返回：

```python
(
    attn_output,
    None,
    past_key_value,
)
```

如果 Decode 设置：

```python
output_attentions = True
```

则返回经过 sparse mask 和 softmax 后的 Attention 概率。

### 重要：`attention_forward()` 在完整 Decoder Layer 中的位置

`attention_forward()` 只替换 `self_attn.forward()`，不会替换 Decoder Layer，也不会删除 MLP。

```python
# 等价结构，仅用于帮助理解，不是仓库源码逐字复制

hidden_states
      ↓
input_layernorm
      ↓
self.self_attn(...)       # 被替换成 attention_forward()
      ↓
residual add
      ↓
post_attention_layernorm
      ↓
self.mlp(...)             # 仍然执行 Hugging Face 原始 MLP
      ↓
residual add
```

模型中同时存在多级 `forward()`：

```python
model(...)
  ↓
LlamaForCausalLM.forward()
  ↓
LlamaModel.forward()
  ↓
for decoder_layer in layers:
    LlamaDecoderLayer.forward()
      ├── self_attn.forward()  # Twilight 替换这一层级
      └── mlp.forward()        # 没有修改
  ↓
lm_head
```

`pred.py` 中的一次：

```python
output = model(...)
```

会让所有 Decoder Layer 顺序执行一次，每个 Decoder Layer 内部都会先执行自己的 `self_attn`，再执行 MLP。

### 重要：Prefill、问题模拟 Decode、答案生成 Decode

`benchmark/LongBench/pred.py` 把推理拆成三段：

```text
1. 长上下文 prompt：一次性 Prefill
2. question：已知 token，但是故意逐 token 输入，模拟 Decode
3. answer：未知 token，自回归逐 token 生成
```

第一段：

```python
output = model(
    input_ids=input.input_ids,
    past_key_values=None,
    use_cache=True,
)
```

通常 `q_len > 1`，进入 Flash Attention Prefill，并建立 KV Cache。

第二段：

```python
for input_id in q_input.input_ids[0]:
    output = model(
        input_ids=input_id.unsqueeze(0).unsqueeze(0),
        past_key_values=past_key_values,
        use_cache=True,
    )
```

Question 本来是已知文本，理论上可以一起 Prefill；这里故意一次只输入一个 token，使 `q_len == 1`，从而让读取问题的过程也进入 Twilight Decode selector。

第三段：

```python
for _ in range(max_gen - 1):
    outputs = model(
        input_ids=pred_token_idx,
        past_key_values=past_key_values,
        use_cache=True,
    )
```

答案 token 在生成前未知，只能根据上一个 token 自回归预测下一个 token，因此逐 token Decode 是必需的。

### 重要：`view()`、`reshape()`、`transpose()` 和 `contiguous()`

#### `transpose(dim0, dim1)`

交换两个维度：

```python
x.shape
# [bs, q_len, num_heads, head_dim]

y = x.transpose(1, 2)
y.shape
# [bs, num_heads, q_len, head_dim]
```

`transpose()` 通常只改变 Tensor 的 stride/view，不一定复制底层数据，所以结果可能不连续。

#### `view(...)`

按新形状解释同一块内存，一般要求内存布局与目标形状兼容：

```python
x = x.view(bs, q_len, num_heads, head_dim)
```

如果 Tensor 不连续，直接 `view()` 可能失败。

#### `reshape(...)`

目标与 `view()` 类似，但遇到不连续 Tensor 时，可以在必要时创建连续副本：

```python
x = x.reshape(bs, q_len, hidden_size)
```

#### `contiguous()`

让 Tensor 按当前维度顺序重新存为连续内存：

```python
x = x.transpose(1, 2).contiguous()
```

当前代码在 Flash Attention 输出 reshape 后调用：

```python
attn_output = attn_output.reshape(
    bsz,
    q_len,
    -1,
).contiguous()
```

### 重要：`**self.state.xxx_args` 的参数解包

例如：

```python
self.state.selector_args = {
    "chunk_size": 16,
    "token_budget": 8192,
}
```

调用：

```python
quest_selector(
    self,
    query_states,
    key_states,
    **self.state.selector_args,
)
```

等价于：

```python
quest_selector(
    self,
    query_states,
    key_states,
    chunk_size=16,
    token_budget=8192,
)
```

前提是字典 key 必须与函数参数名一致。否则可能出现：

```text
TypeError: got an unexpected keyword argument
```

### 重要：GQA、MHA 和 KV Cache

#### MHA

```python
num_heads == num_key_value_heads
```

每个 Q head 都有独立的 K/V head。

#### GQA

```python
num_heads > num_key_value_heads
```

多个 Q head 共享一个 K/V head，可以减少 K/V projection 和 KV Cache 占用。

但是当前 `attention_forward()` 的执行顺序是：

```python
K/V projection
      ↓
repeat_kv()              # 先扩展到 num_heads
      ↓
past_key_value.update()  # 再写 Cache
```

这意味着写入 Cache 的已经是重复后的 K/V，而不是原始 `num_key_value_heads` 形状。在标准 GQA 实现中，通常会先缓存较少的 KV heads，再在计算 Attention 前 `repeat_kv()`。

因此当前 Python 实现可能失去 GQA 原本节省 KV Cache 空间的优势。这也符合文件开头“用于准确率验证，而不是效率”的定位。

### 重要：Attention score、Attention weight 和 mask

这三个概念不要混在一起：

```text
Attention score/logit
    = QK^T / sqrt(d)
    = softmax 之前的任意实数

Attention weight/probability
    = softmax(score)
    = 非负，并且 token 维求和约等于 1

mask
    = bool Tensor
    = 决定哪些 token 可以进入 softmax
```

当前代码中的变量名变化：

```python
attn_weights = Q @ K.T / sqrt(d)
# 此时其实是 score/logit

attn_weights[~mask] = dtype_min
# 未选位置变成非常大的负数

attn_weights = softmax(attn_weights)
# 到这里才是真正的 weight/probability
```

### 重要：`mask & pruned_mask` 和 `~mask`

两个 mask 都是 Bool Tensor：

```python
mask & pruned_mask
```

是逐元素逻辑与，只有两个位置都为 `True` 才保留。

```python
~mask
```

是逐元素逻辑非，用于找到所有未选择位置。

不能把它和 Python 标量布尔运算：

```python
mask and pruned_mask
not mask
```

混为一谈。`and/not` 会尝试判断整个 Tensor 的真假，多元素 Tensor 通常会报“truth value is ambiguous”。

### 重要：当前实现中的边界条件和潜在问题

#### 1. `TIDAL_DECODE` 的返回值和调用端不一致

调用端期望：

```python
mask = tidal_decode_selector(...)
# mask 应当是 Bool Tensor
```

但 `tidal_decode.py` 当前实际返回：

```python
return None, torch.ones_like(...)
# 或
return None, getattr(state, "td_mask_buffer")
```

也就是：

```python
mask == (None, BoolTensor)
```

后面的：

```python
~mask
mask & pruned_mask
```

都要求 `mask` 是 Tensor，因此该分支当前存在接口错误。

#### 2. 枚举值不一定有执行分支

`state.py` 中定义了：

```python
IndexSelectorType.PQCACHE
WeightEstimatorType.THRESHOLD
WeightEstimatorType.H_THRESHOLD
```

但是当前 `attention_forward()` 没有对应的 Selector / Estimator 分支。配置成这些值时，`mask` 或 `estimated_weights` 可能没有赋值就被使用。

#### 3. Prefill 中 `output_attentions=True` 可能使用未定义变量

Prefill 只在下面条件成立时赋值：

```python
if not output_attentions:
    attn_weights = None
```

如果：

```python
output_attentions = True
```

Flash Attention 分支并没有创建 `attn_weights`，但是 return 仍然引用它，可能触发 `UnboundLocalError`。

#### 4. `use_cache=True` 默认假设 `past_key_value` 已经存在

代码直接调用：

```python
past_key_value.update(...)
```

所以如果调用者传入：

```python
use_cache=True
past_key_value=None
```

且模型上层没有提前创建 Cache 对象，这里会触发 `AttributeError`。正常 Hugging Face 模型流程通常会在进入 Attention 层前准备 Cache。

#### 5. Decode 不支持非空 Attention mask

```python
assert attention_mask is None
```

因此 padding batch、某些自定义 mask 或部分 Transformers 调用方式可能不兼容。

#### 6. `torch.cuda.empty_cache()` 每层、每次 forward 都执行

它不会清理仍被引用的 KV Cache，却可能增加 allocator 和同步开销。若目标是性能测试，需要单独评估这一行的影响。

#### 7. GQA 在写入 Cache 前已经 `repeat_kv()`

缓存重复后的 KV heads 会增加 Cache 体积，可能抵消 GQA 的内存优势。

#### 8. 当前实现仍然计算完整 score

```python
attn_weights = query_states @ key_states.transpose(2, 3)
```

发生在 Selector 之前。因此 selector 再稀疏，也无法省掉当前 Python 实现已经完成的完整 QK 计算。它适合验证输出精度、budget 和 score coverage，不适合用来代表真正 sparse kernel 的速度。

#### 9. `torch.tensor(dtype_min)` 没有显式指定 device

当前写法：

```python
torch.tensor(torch.finfo(attn_weights.dtype).min)
```

默认先创建 CPU 标量 Tensor，再用于 CUDA Tensor 的索引赋值。标量赋值在不少情况下可以工作，但会带来不必要的 device/同步隐患；直接赋 Python 标量或显式指定同一 device 更清楚。

#### 10. `q_len` 是执行路径判断，不等同于完整生成阶段名称

```python
q_len > 1 → Prefill 分支
q_len == 1 → Decode 分支
```

这只是当前实现的判断规则。例如只有一个 token 的初始输入也会进入“Decode”分支，即使它在业务语义上是第一次模型调用。

### 重要：用 `config_quest_twi.json` 串起整个 Decode 流程

配置：

```json
{
    "compressor": {
        "type": "none"
    },
    "selector": {
        "type": "quest",
        "token_budget": 8192,
        "chunk_size": 16
    },
    "weight_estimator": {
        "type": "min_max_quant",
        "quant_bit": 4
    },
    "weight_pruner": {
        "type": "top_p",
        "threshold": 0.85
    },
    "skip_first_two_layers": true,
    "use_estimated_weights_in_attn": false
}
```

脑内翻译：

```python
Decode，且 layer_id >= 2
      ↓
Quest 按 chunk 估分
      ↓
最多选择约 8192 个候选 token，得到 B0 mask
      ↓
对完整 K 做逐 token 4-bit min-max Fake Quant
      ↓
Q @ quantized_K.T，得到 estimated_weights
      ↓
Selector 外部位置设为 dtype_min
      ↓
Top-P(0.85) 在候选集合中进一步剪枝
      ↓
最终 mask = Quest mask & Top-P mask
      ↓
记录最终 budget 和真实概率覆盖率
      ↓
use_estimated_weights_in_attn=False
所以继续使用真实 attn score
      ↓
用最终 mask 屏蔽真实 score
      ↓
FP32 softmax
      ↓
attn_weights @ V
      ↓
o_proj
```

这里可以把 Twilight 的核心目标概括成：

```text
先用 Selector 给出相对宽松的候选集合 B0
                    ↓
再用低比特 Weight Estimator 判断候选 token 的重要性
                    ↓
最后用 Weight Pruner 得到更小、随 query 动态变化的 B1
                    ↓
在 B1 上用真实权重完成 Attention
```
