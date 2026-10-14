# Top-p 代码解读

> [!TIP]
>
> CheckList
>
> - [x] 是否有 TL;DR
> - [x] 是否有整体流程
> - [x] 是否能清晰拆分步骤
> - [x] 每一步是否能加入代码块和注释
> - [x] 是否有“重要”的语法 / 模型推理过程

## TL;DR

> [!IMPORTANT]
>
> `twilight/pyimpl/top_p.py` 实现的是**注意力 token 剪枝**，不是语言模型生成阶段从词表中抽样下一个 token。
>
> 1. 输入是尚未归一化的 attention logits，典型形状为 `[bs, num_heads, 1, kv_seq_len]`。
> 2. 三个 Python 接口都会先沿最后一维执行 FP32 Softmax。
> 3. CUDA kernel 不先排序，而是反复扫描每一行概率，通过概率值域上的二分搜索找到分界值 `low`。
> 4. `mask` 接口返回与输入同形状的布尔掩码；`indices` 接口返回紧凑索引和每行真实数量；`indices_out` 把结果写入预分配 Tensor。
> 5. `attention.py` 当前只使用 `top_p_unnormalized()` 的 mask 路径。两个 indices 路径主要出现在性能实验和稀疏索引准备代码中。
> 6. 当前 CUDA 源码有两个重要前提：mask 路径只适用于 `q_len == 1`；搜索上界取第 0 个 token 的概率，因此要求 token 0 是该行最大概率。源码虽然算出了真实最大值，但没有把它赋给 `high`。

## 涉及文件

| 层级 | 文件 | 本文分析的内容 |
| --- | --- | --- |
| Python 算法入口 | `twilight/pyimpl/top_p.py:1-48` | Softmax、mask / indices / 预分配 indices 三个接口 |
| Python CUDA 包装 | `twilight/kernel/cuda/sampling.py:37-273` | 模块加载、custom op、Tensor 分配、阈值参数转换 |
| 路径配置 | `twilight/kernel/cuda/env.py:1-5` | JIT 编译时定位 `csrc/src` 与 `csrc/include` |
| Python 导出 | `twilight/kernel/__init__.py:1-8` | 把 CUDA 包装函数暴露给 `top_p.py` |
| PyTorch/CUDA 桥接 | `csrc/src/sampling.cu:22-150` | 检查 Tensor、取得指针和 stream、调用 CUDA launcher、PyBind 导出 |
| C++ 检查宏 | `csrc/src/pytorch_extension_utils.h:23-56, 220-233` | FP16/BF16 dtype 分派以及 CUDA、连续内存、维数检查 |
| CUDA 实现 | `csrc/include/sampling.cuh:138-435` | Top-p 分界搜索、mask 写回、indices 压紧、kernel launch |
| 模型调用处 | `twilight/pyimpl/attention.py:267-304` | Decode 阶段以 Top-p mask 剪枝，再重新 Softmax |

## 整体流程

```text
Decode 阶段 estimated attention logits
形状：[bs, num_heads, 1, kv_seq_len]
                    │
                    ▼
top_p_unnormalized(...)
                    │
                    ├── Softmax(dim=-1, dtype=float32)
                    │       logits → probabilities
                    ▼
top_p_fp32_return_mask(...)
                    │
                    ▼
sampling.py：加载 / 缓存 CUDA 扩展，进入 registered custom op
                    │
                    ▼
csrc/src/sampling.cu：检查参数，以最后一维为 row（mask 路径要求 q_len=1）
                    │
                    ▼
csrc/include/sampling.cuh
                    │
                    ├── 每个 row 启动 1 个 CUDA block
                    ├── 反复求 sum(probability > pivot)
                    ├── 在概率值域中搜索分界 low
                    └── mask[i] = probability[i] > low
                    │
                    ▼
返回 bool mask
                    │
                    ▼
attention.py：mask = selector_mask & top_p_mask
                    │
                    ├── 未保留 logits 写成该 dtype 的最小有限值
                    ├── 再做一次 Softmax
                    └── attention_probability @ V
```

如果调用 indices 路径，前半段相同，最后一步改为：

```text
probability > low
       │
       ├── BlockScan 计算每个线程的写入偏移
       ├── 把选中列号紧凑写入 output_indices
       └── 把真实选中数量写入 output_counts
```

---

# pyimpl

## top_p.py

### Top-p 到底保留什么

设某一行 Softmax 后的概率为：

$$
\mathbf{a}=(a_0,a_1,\ldots,a_{K-1}),\qquad
a_i\ge 0,\qquad \sum_i a_i=1
$$

标准 Top-p 的目标是：按概率从大到小排列后，找到累计概率第一次达到阈值 $p$ 的最短前缀。

```python
# 仅用于解释数学含义；源码没有真的执行 torch.sort
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)

# 理想语义：保留累计概率达到 p 所需的最少若干项
```

例如：

```text
probs = [0.644, 0.237, 0.087, 0.032]
p = 0.85

0.644                 < 0.85
0.644 + 0.237 = 0.881 >= 0.85

最终保留前两个 token：mask = [True, True, False, False]
```

Twilight kernel 用一个概率分界值 `low` 表示结果：

$$
\text{mask}_i=[a_i>low]
$$

它通过反复计算下面这个函数寻找 `low`，不显式排序：

$$
f(x)=\sum_{i:a_i>x}a_i
$$

`x` 越大，被累加的 token 越少，所以 $f(x)$ 单调不增，能够在概率值域上做二分搜索。

> [!NOTE]
>
> 当分界处存在相同概率时，单一数值阈值无法只选择其中一部分。源码使用严格大于 `>`，最终可能整体保留或排除同值项；为保证累计质量时，实际结果可能比“排序后选最短前缀”保留更多并列 token。

### 三个函数对比

| 函数 | 返回值 | 是否在函数内分配输出 | 主要使用位置 |
| --- | --- | --- | --- |
| `top_p_unnormalized` | `bool mask` | 是 | `attention.py` 的真实 Python 稀疏注意力路径 |
| `top_p_unnormalized_return_indices` | `(int32 indices, int32 counts)` | 是 | benchmark / 首次探测实际容量 |
| `top_p_unnormalized_return_indices_out` | `(output_indices, output_counts)` | 否，调用者预分配 | 循环 benchmark / 复用 buffer |

三个函数共同要求最后一维是待剪枝的 KV token 维度。当前 C++ 桥接还要求输入为 4 维：

| 张量 | 典型 shape | dtype | 含义 |
| --- | --- | --- | --- |
| `attn_weights` | `[bs, h, q_len, kv_len]` | 常见 FP16/BF16/FP32 | 未归一化 attention logits |
| `normalized_weights` | 同输入 | FP32 | 沿 `kv_len` 做 Softmax 后的概率 |
| `mask` | 同输入 | `torch.bool` | 每个 KV token 是否保留 |
| `indices` | `[bs, h, q_len, output_len]` | `torch.int32` | 每行压紧后的保留 token 索引 |
| `counts` | `[bs, h, q_len]` | `torch.int32` | 每行实际选中数量 |

### top_p_unnormalized

源码位置：`twilight/pyimpl/top_p.py:14-19`。

输入和输出：

| 参数 | type | 说明 |
| --- | --- | --- |
| `attn_weights` | `torch.Tensor` | 未归一化的 attention logits，最后一维是 KV token |
| `threshold` | `float` | Top-p 的累计概率阈值，越大通常保留越多 token |
| 返回值 | `torch.Tensor` | 与输入同 shape 的布尔 mask |

整体只有两步：

1. 把 logits 转换成 FP32 概率；
2. 调用 FP32 CUDA kernel 返回 mask。

```python
def top_p_unnormalized(
    attn_weights: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    # Step 1：最后一维 kv_len 上做 Softmax。
    # 即使 attn_weights 是 FP16 / BF16，输出也强制为 FP32。
    normalized_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    )

    # Step 2：输入已经是 probability，所以调用 fp32 kernel。
    # 返回值 mask.shape == normalized_weights.shape。
    mask = top_p_fp32_return_mask(
        normalized_weights,
        threshold,
    )
    return mask
```

为什么函数名带 `unnormalized`：因为它接收的是 logits，而底层 kernel 接收的是 probabilities。

```text
top_p_unnormalized(logits)
           │
           └── softmax ──▶ top_p_fp32_return_mask(probabilities)
```

### top_p_unnormalized_return_indices

源码位置：`twilight/pyimpl/top_p.py:22-33`。

输入和输出：

| 参数 | type | 说明 |
| --- | --- | --- |
| `attn_weights` | `torch.Tensor` | 未归一化 logits |
| `threshold` | `float` | Top-p 累计概率阈值 |
| `input_indices` | `torch.Tensor \| None` | 可选的原始索引映射，需为 int32 CUDA Tensor |
| `max_output_len` | `int \| None` | `indices` 最后一维的容量；默认等于 `kv_len` |
| 返回值 1 | `torch.Tensor` | int32 紧凑索引，shape 为 `[..., max_output_len]` |
| 返回值 2 | `torch.Tensor` | int32 真实选中数，shape 为 `attn_weights.shape[:-1]` |

```python
def top_p_unnormalized_return_indices(
    attn_weights,
    threshold,
    input_indices=None,
    max_output_len=None,
):
    # Step 1：logits → FP32 probabilities。
    normalized_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    )

    # Step 2：底层先分配 indices / counts，再调用 indices_out 路径。
    return top_p_fp32_return_indices(
        normalized_weights,
        threshold,
        input_indices,
        max_output_len,
    )
```

`input_indices` 用于“候选集合已经是原始 KV 的一个子集”的情况：

```text
当前概率行的本地列号：       [0, 1, 2, 3]
input_indices 保存的原始位置：[7, 9, 20, 31]
Top-p 在本地选中：           [0, 2]

input_indices is None  → output_indices = [0, 2]
input_indices 已提供   → output_indices = [7, 20]
```

这使第二阶段 Top-p 可以把“候选数组中的局部下标”重新映射成“完整 KV Cache 中的原始 token 下标”。

> [!WARNING]
>
> `max_output_len` 只是 `output_indices` 的容量，不会截断 `output_counts`。如果真实选择 100 个 token、容量只有 64，kernel 只写入 64 个索引，但 `counts` 仍返回 100。要取得完整索引，必须保证：
>
> `max_output_len >= counts.max()`。

### top_p_unnormalized_return_indices_out

源码位置：`twilight/pyimpl/top_p.py:36-48`。

这个函数与上一函数的筛选语义完全相同，区别是输出 Tensor 由调用者提前创建。

```python
def top_p_unnormalized_return_indices_out(
    attn_weights,
    output_indices,
    output_counts,
    threshold,
    input_indices=None,
):
    # Step 1：每次调用仍会新建 Softmax 的 FP32 输出。
    normalized_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    )

    # Step 2：indices 和 counts 不再分配，直接写入调用者的 buffer。
    return top_p_fp32_return_indices_out(
        normalized_weights,
        output_indices,
        output_counts,
        threshold,
        input_indices,
    )
```

典型使用方式见 `benchmark/efficiency/bench_breakdown.py:231-275`：

```python
# 第一次运行：先得到每行真实 counts，据此确定所需最大容量。
indices, counts = top_p_unnormalized_return_indices(logits, p)
max_output_len = max(1, int(counts.max().item()))

# 只分配一次，后续 decode / benchmark 循环反复复用。
output_indices = torch.empty(
    (*logits.shape[:-1], max_output_len),
    dtype=torch.int32,
    device="cuda",
)
output_counts = torch.empty(
    logits.shape[:-1],
    dtype=torch.int32,
    device="cuda",
)

top_p_unnormalized_return_indices_out(
    logits,
    output_indices,
    output_counts,
    p,
)
```

> [!NOTE]
>
> 复用的只是 `output_indices` 和 `output_counts`；`normalized_weights` 仍由每次 Softmax 产生，因而这个接口不是“整个调用零分配”。

### 重要：`dtype=torch.float32` 为什么写在 Softmax 中

```python
normalized_weights = nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
)
```

这里同时完成两件事：

1. 计算前把输入转换为 FP32；
2. Softmax 的输出也是 FP32。

长序列 attention 中，大量较小概率需要参与求和。使用 FP32 能减小 FP16/BF16 下的舍入和下溢影响，同时与随后调用的 `top_p_fp32_*` 接口类型一致。

这不代表原模型后续一直使用 FP32。`attention.py:301-304` 在最终 mask 后 Softmax 时，会再把概率转换回 `query_states.dtype`：

```python
# 最终真正参与 attention @ V 的概率
attn_weights = nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
).to(query_states.dtype)
```

因此 Top-p 内部第一次 Softmax 的主要用途是**确定保留集合**；最终 attention 会在应用 mask 后重新归一化。

### 重要：这里的 Top-p 不是生成阶段的 nucleus sampling

二者数学操作相似，但作用对象和输出完全不同：

| 对比项 | Twilight attention Top-p | 文本生成 nucleus sampling |
| --- | --- | --- |
| 概率来自 | 一个 query 对历史 KV token 的 attention logits | LM head 对整个词表的 logits |
| 最后一维 | `kv_seq_len` | `vocab_size` |
| 目的 | 选择要参与 attention 的历史 token | 选择下一个生成 token 的候选词 |
| 结果 | mask 或 KV 索引 | 抽样出的词表 token id |
| 是否随机采样 | 否 | 通常会从截断后的分布随机采样 |

当前真实 Python attention 调用链位于 `twilight/pyimpl/attention.py:267-304`：

```python
# estimated_weights 此处还是未归一化 attention logits。
pruned_mask = top_p_unnormalized(
    estimated_weights,
    self.state.weight_pruner_args["threshold"],
)

# Selector mask 与 Top-p mask 取交集，得到最终 mask。
mask = mask & pruned_mask

# 未选择 token 的真实 / 估计 logits 被写成极小值。
attn_weights[~mask] = torch.tensor(
    torch.finfo(attn_weights.dtype).min
)

# 对保留 token 重新归一化，再与 V 相乘。
attn_weights = nn.functional.softmax(
    attn_weights,
    dim=-1,
    dtype=torch.float32,
).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states)
```

这里还需要区分两种 estimator 情况：

```text
WeightEstimator != NONE
    ├── 先把 Selector 未选中的 estimated logits 设为极小值
    ├── Top-p 的概率分布实际限制在 Selector 候选集合内
    └── 再与 Selector mask 取交集

WeightEstimator == NONE
    ├── estimated_weights 直接等于完整 attn_weights
    ├── Top-p 先基于完整 KV 行计算 pruned_mask
    └── 最后才与 Selector mask 取交集
```

因此不能无条件表述成“Top-p 总是在 B0 内重新归一化”。只有量化 estimator 分支在调用 Top-p 前明确把候选外 logits 设成极小值；`NONE` 分支的 Top-p 阈值仍受完整 KV 行影响。

### 重要：Python 类型语法 `Tensor | None` 和 `tuple[...]`

```python
input_indices: torch.Tensor | None = None
max_output_len: int | None = None

) -> tuple[torch.Tensor, torch.Tensor]:
```

- `A | None` 表示参数可以是 `A`，也可以是 `None`，等价于 `Optional[A]`。
- `tuple[Tensor, Tensor]` 表示返回一个二元组，两个元素都是 Tensor。
- 这些是类型提示，不会自动做运行时检查；真正的 dtype / shape 检查主要发生在 C++ 桥接层。

---

# kernel

## Python 导出：kernel/__init__.py

`top_p.py:7-11` 从 `twilight.kernel` 导入函数，而 `twilight/kernel/__init__.py:1-8` 又从 `cuda/sampling.py` 重导出这些函数：

```text
twilight.pyimpl.top_p
        │ from twilight.kernel import ...
        ▼
twilight.kernel.__init__
        │ from .cuda.sampling import ...
        ▼
twilight.kernel.cuda.sampling
```

因此 `top_p_fp32_return_mask` 不是纯 Python 算法，它最终进入 CUDA 扩展。

## cuda/sampling.py

### get_sampling_module

源码位置：`twilight/kernel/cuda/sampling.py:37-153`。

作用：第一次调用时取得 CUDA 扩展、注册 PyTorch custom op，并把结果缓存在模块级变量 `_sampling_module` 中。

```python
_sampling_module = None

def get_sampling_module():
    global _sampling_module

    # 只在第一次调用时加载 / 注册。
    if _sampling_module is None:
        if has_prebuilt_ops:
            # FlashInfer 环境已有预编译扩展。
            from . import _kernels
            module = _kernels
        else:
            # 没有预编译扩展时，运行时编译 csrc/src/sampling.cu。
            module = load_cuda_ops(
                "twilight_sampling",
                sources=[TWILIGHT_CSRC_DIR / "sampling.cu"],
                extra_include_paths=[
                    FLASHINFER_INCLUDE_DIR,
                    FLASHINFER_CSRC_DIR,
                    TWILIGHT_INCLUDE_DIR,
                ],
            )

        # 中间省略 custom op 注册……

        # SimpleNamespace 把底层 module 和包装函数放入同一个对象。
        _sampling_module = SimpleNamespace(
            _raw_module=module,
            top_p_fp32_return_mask=top_p_fp32_return_mask,
            top_p_fp32_return_indices=top_p_fp32_return_indices,
            # fp16 接口同理
        )

    return _sampling_module
```

JIT 路径中的源码目录来自 `twilight/kernel/cuda/env.py:1-5`：

```python
TWILIGHT_HOME = pathlib.Path(__file__).resolve().parents[3]
TWILIGHT_CSRC_DIR = TWILIGHT_HOME / "csrc" / "src"
TWILIGHT_INCLUDE_DIR = TWILIGHT_HOME / "csrc" / "include"
```

第一次 JIT 调用可能包含编译开销；后续调用复用 `_sampling_module`，不会每次重新执行加载逻辑。

### registered `top_p_fp32_return_mask`

源码位置：`twilight/kernel/cuda/sampling.py:82-100`。

```python
@register_custom_op(
    "flashinfer::top_p_fp32_return_mask",
    mutates_args=(),
)
def top_p_fp32_return_mask(probs, maybe_top_p_arr, top_p_val):
    with probs.device as device:
        # 新建与 probs 同 shape 的 bool Tensor，并初始化为 False。
        mask = torch.zeros_like(probs, dtype=torch.bool)

        # 进入 PyBind 暴露的 C++ 函数；C++ / CUDA 原地写 mask。
        module.top_p_fp32_return_mask(
            probs,
            mask,
            maybe_top_p_arr,
            top_p_val,
            get_cuda_stream(device),
        )
        return mask
```

`mutates_args=()` 描述的是这个 registered op 对外没有修改其输入参数；`mask` 是函数内部创建的新输出。底层 CUDA 确实会原地填写这个新建的 `mask`。

### registered `top_p_fp32_return_indices`

源码位置：`twilight/kernel/cuda/sampling.py:123-142`。

```python
@register_custom_op(
    "flashinfer::top_p_fp32_return_indices",
    mutates_args=(),
)
def top_p_fp32_return_indices(
    probs,
    maybe_input_indices,
    maybe_top_p_arr,
    top_p_val,
):
    # indices 暂按 probs 的完整 shape 分配。
    indices = torch.empty_like(probs, dtype=torch.int32)

    # counts 去掉最后一个 kv_len 维度。
    counts = torch.empty(
        probs.shape[:-1],
        dtype=torch.int32,
        device=probs.device,
    )

    module.top_p_fp32_return_indices(
        probs,
        indices,
        counts,
        maybe_input_indices,
        maybe_top_p_arr,
        top_p_val,
        get_cuda_stream(probs.device),
    )
    return indices, counts
```

需要注意：当前公开的 `top_p_fp32_return_indices()` 最后会转到 `_out` 接口并直接调用 `_raw_module`，没有走上面这个 registered indices op；这个注册函数仍被保存在 `SimpleNamespace` 中。

### _to_tensor_scalar_tuple

源码位置：`twilight/kernel/cuda/sampling.py:156-160`。

底层 C++ 同时支持“所有 row 共用一个 p”和“每个 row 有自己的 p”。这个辅助函数把一个 Python 参数拆成固定的两个槽位：

```python
def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        # 使用 top_p_arr；标量槽位填 0，不会被 kernel 读取。
        return (x, 0)
    else:
        # 数组槽位为空；使用标量 top_p_val。
        return (None, x)
```

调用处使用 `*` 解包：

```python
get_sampling_module().top_p_fp32_return_mask(
    probs,
    *_to_tensor_scalar_tuple(top_p),
)

# top_p = 0.85 时，等价于：
top_p_fp32_return_mask(probs, None, 0.85)

# top_p = tensor([...]) 时，等价于：
top_p_fp32_return_mask(probs, top_p, 0)
```

### top_p_fp32_return_mask

源码位置：`twilight/kernel/cuda/sampling.py:172-178`。

```python
def top_p_fp32_return_mask(probs, top_p):
    # 取得缓存后的 registered custom op；
    # 再把 top_p 拆成 optional Tensor + scalar 两种表示。
    return get_sampling_module().top_p_fp32_return_mask(
        probs,
        *_to_tensor_scalar_tuple(top_p),
    )
```

这是 `top_p_unnormalized()` 的直接下游函数。

### top_p_fp32_return_indices

源码位置：`twilight/kernel/cuda/sampling.py:200-216`。

```python
def top_p_fp32_return_indices(
    probs,
    top_p,
    input_indices=None,
    max_output_len=None,
):
    # 未给容量时，按最坏情况“所有 KV token 都保留”分配。
    output_len = (
        max_output_len
        if max_output_len is not None
        else probs.shape[-1]
    )

    indices = torch.empty(
        (*probs.shape[:-1], output_len),
        dtype=torch.int32,
        device=probs.device,
    )
    counts = torch.empty(
        probs.shape[:-1],
        dtype=torch.int32,
        device=probs.device,
    )

    # 分配版最终复用预分配版的实现。
    return top_p_fp32_return_indices_out(
        probs,
        indices,
        counts,
        top_p,
        input_indices,
    )
```

### top_p_fp32_return_indices_out

源码位置：`twilight/kernel/cuda/sampling.py:248-273`。

```python
def top_p_fp32_return_indices_out(
    probs,
    output_indices,
    output_counts,
    top_p,
    input_indices=None,
):
    module = get_sampling_module()
    maybe_top_p_arr, top_p_val = _to_tensor_scalar_tuple(top_p)

    # 取出实际 PyBind/JIT extension，而不是 registered wrapper。
    raw_module = getattr(module, "_raw_module", None)
    if raw_module is None:
        raise RuntimeError("raw twilight sampling module is unavailable")

    # CUDA 直接修改调用者提供的两个输出 buffer。
    raw_module.top_p_fp32_return_indices(
        probs,
        output_indices,
        output_counts,
        input_indices,
        maybe_top_p_arr,
        top_p_val,
        get_cuda_stream(probs.device),
    )
    return output_indices, output_counts
```

之所以直接使用 `_raw_module`，源码注释已经给出原因：registered custom op 自己分配输出，不适合复用调用者预先分配的 buffer。

### FP16 接口与 FP32 接口的关系

`sampling.py` 还定义了同构的 FP16 接口：

```text
top_p_fp16_return_mask
top_p_fp16_return_indices
top_p_fp16_return_indices_out
```

它们与 FP32 版本的控制流基本相同，只是 C++ 使用 `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` 分派 FP16；编译时定义 `FLASHINFER_ENABLE_BF16` 后也分派 BF16。由于 `top_p.py` 强制用 FP32 Softmax，其三个函数实际调用的都是 `top_p_fp32_*`，不是这些 FP16 接口。

### 重要：`torch.empty` 的未写区域不能读取

indices 接口使用 `torch.empty` 分配：

```python
indices = torch.empty(..., dtype=torch.int32, device=probs.device)
```

`torch.empty` 只申请内存，不初始化内容。kernel 只保证每行前 `min(counts[row], output_stride)` 个槽位被写入。因此读取时必须结合 `counts`：

```python
valid_count = min(
    int(output_counts[row].item()),
    output_indices.shape[-1],
)
valid_indices = output_indices[row, :valid_count]
```

不能把一整行 `output_indices` 都当成有效结果。

### 重要：CUDA stream 与异步执行

```python
get_cuda_stream(probs.device)
```

包装层把当前 PyTorch CUDA stream 的句柄传给 C++，kernel 会排入同一条 stream。函数内没有调用 `torch.cuda.synchronize()`；GPU 工作仍遵守当前 stream 的异步执行语义。

当随后由 PyTorch GPU 算子消费结果时，同 stream 顺序会保证依赖；只有要在 CPU 上精确计时或立即读取 `.item()` 时，才通常需要显式同步。benchmark 在读取 `counts.max().item()` 或计时时处理了相应同步逻辑。

---

## csrc/src/sampling.cu

这个文件是 PyTorch Tensor 与模板 CUDA kernel 之间的桥梁。

`CHECK_INPUT` 和 `CHECK_DIM` 定义在 `csrc/src/pytorch_extension_utils.h:220-233`：

```cpp
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
```

所以本文件中的 `CHECK_INPUT(probs)` 不只是泛泛的“检查输入”，而是明确要求 Tensor 位于 CUDA 且内存连续；`CHECK_DIM(4, probs)` 再要求它是 4 维。

### top_p_fp32_return_mask

源码位置：`csrc/src/sampling.cu:45-63`。

```cpp
void top_p_fp32_return_mask(
    at::Tensor probs,
    at::Tensor mask,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    int64_t cuda_stream) {

  CHECK_INPUT(probs);
  CHECK_DIM(4, probs);

  // probs: [batch_size, num_heads, qo_len, kv_seq_len]
  unsigned int batch_size = probs.size(0);
  unsigned int num_heads = probs.size(1);
  unsigned int kv_seq_len = probs.size(3);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);

  // Python Softmax 已确保 probs 是 float32。
  cudaError_t status = sampling::TopPReturnMask<float>(
      static_cast<float*>(probs.data_ptr()),
      static_cast<uint8_t*>(mask.data_ptr()),
      maybe_top_p_arr.has_value()
          ? static_cast<float*>(maybe_top_p_arr->data_ptr())
          : nullptr,
      batch_size * num_heads,
      top_p_val,
      kv_seq_len,
      stream);

  TORCH_CHECK(status == cudaSuccess, ...);
}
```

这里把 `[batch_size, num_heads, q_len, kv_len]` 看成多行概率，每个 CUDA block 负责一行。然而 launcher 收到的 `num_blocks` 是 `batch_size * num_heads`，没有乘 `q_len`。所以这个 mask 路径只有在 `q_len == 1` 时覆盖所有行；这正好符合当前 `attention.py` 只在单 token Decode 阶段进入剪枝的用法。

### top_p_fp32_return_indices

源码位置：`csrc/src/sampling.cu:104-139`。

```cpp
void top_p_fp32_return_indices(
    at::Tensor probs,
    at::Tensor output_indices,
    at::Tensor output_counts,
    std::optional<at::Tensor> maybe_input_indices,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    int64_t cuda_stream) {

  CHECK_DIM(4, probs);
  CHECK_DIM(4, output_indices);
  TORCH_CHECK(probs.scalar_type() == at::ScalarType::Float, ...);
  TORCH_CHECK(output_indices.scalar_type() == at::ScalarType::Int, ...);
  TORCH_CHECK(output_counts.scalar_type() == at::ScalarType::Int, ...);

  unsigned int kv_seq_len = probs.size(3);
  unsigned int output_stride = output_indices.size(3);

  // 与 mask 路径不同，这里把前三维全部计入 row 数。
  unsigned int num_rows = probs.numel() / kv_seq_len;

  cudaError_t status = sampling::TopPReturnIndices<float>(
      static_cast<float*>(probs.data_ptr()),
      static_cast<int32_t*>(output_indices.data_ptr()),
      static_cast<int32_t*>(output_counts.data_ptr()),
      maybe_input_indices.has_value()
          ? static_cast<int32_t*>(maybe_input_indices->data_ptr())
          : nullptr,
      maybe_top_p_arr.has_value()
          ? static_cast<float*>(maybe_top_p_arr->data_ptr())
          : nullptr,
      num_rows,
      top_p_val,
      kv_seq_len,
      output_stride,
      stream);
}
```

此处 `num_rows = bs × heads × q_len`，所以 indices 路径会覆盖全部 query 行。源码还检查 `probs` 为 float32、索引和计数为 int32；`input_indices` 存在时也必须是 int32。

> [!WARNING]
>
> 该桥接层没有在本文件中显式检查 `output_counts` 的维数、各输出 shape 是否与 `probs` 匹配、`top_p_arr` 的长度和 dtype 是否正确。调用者应遵守 Python 包装函数创建出的标准契约，不要随意传入不匹配的预分配 Tensor。

### PYBIND11_MODULE

源码位置：`csrc/src/sampling.cu:141-150`。

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_p_fp32_return_mask",
        &top_p_fp32_return_mask,
        "Using top-p sampling to mask attention weights");

  m.def("top_p_fp32_return_indices",
        &top_p_fp32_return_indices,
        "Using top-p sampling to compact attention weight indices");
}
```

它把 C++ 函数暴露为 Python extension module 的属性，所以 `sampling.py` 才能调用：

```python
module.top_p_fp32_return_mask(...)
raw_module.top_p_fp32_return_indices(...)
```

### 重要：`std::optional`、裸指针与 `reinterpret_cast`

```cpp
std::optional<at::Tensor> maybe_top_p_arr
```

表示 Python 传入的可能是 Tensor，也可能是 `None`。C++ 再将其转换成 CUDA kernel 使用的空 / 非空指针：

```cpp
has_top_p_arr
    ? static_cast<float*>(maybe_top_p_arr->data_ptr())
    : nullptr
```

Tensor 真正的数据地址通过 `data_ptr()` 取得。`static_cast<float*>` 告诉 C++ 按 FP32 数组解释这段显存。

stream 句柄则来自 Python 整数：

```cpp
cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
```

`reinterpret_cast` 不复制 stream，只把句柄的位模式重新解释成 CUDA stream 类型。

---

## csrc/include/sampling.cuh

### 线程与数据映射

两个 CUDA kernel 都采用：

```cpp
const uint32_t bx = blockIdx.x;
const uint32_t tx = threadIdx.x;
const uint32_t row_idx = bx;
```

对应关系是：

```text
1 个 CUDA block  ←→  1 行 probability
1024 个线程      ←→  合作扫描这一行的 kv_len 个元素
VEC_SIZE 个元素  ←→  每个线程一次向量化处理的元素数
```

当 `kv_len > 1024 × VEC_SIZE` 时，每个线程通过外层 `i` 循环继续处理后面的片段。

### 临时共享内存

mask kernel 使用 `RenormTempStorage`，indices kernel 使用多一个 BlockScan 缓冲区的 `RenormIndicesTempStorage`：

```cpp
template <typename T, uint32_t BLOCK_THREADS, ...>
struct RenormIndicesTempStorage {
  union {
    BlockReduce<T, ...>::TempStorage reduce;
    BlockReduce<int, ...>::TempStorage reduce_int;
    BlockReduce<Pair<T>, ...>::TempStorage reduce_pair;
    BlockScan<int, ...>::TempStorage scan;
  } block_prim;

  struct {
    T max_val;
    T min_val;
    union {
      T value;
      int count;
      Pair<T> pair;
    } block_aggregate;
  };
};
```

这里的 `union` 表示不同阶段复用同一块 shared memory；reduce 和 scan 不会同时需要，所以不必把所有临时空间累加分配。

### TopPReturnMaskKernel

源码位置：`csrc/include/sampling.cuh:138-250`。

#### Step 1：确定当前 row 和阈值

```cpp
const uint32_t row_idx = blockIdx.x;

// top_p_arr 非空：每行使用自己的阈值。
// top_p_arr 为空：所有行使用同一个标量 top_p_val。
DType p = DType(
    top_p_arr == nullptr
        ? top_p_val
        : top_p_arr[row_idx]
);
```

#### Step 2：并行扫描行最大概率

```cpp
DType threadlocal_max_val = DType(0);

for (uint32_t i = 0;
     i < ceil_div(d, BLOCK_THREADS * VEC_SIZE);
     ++i) {
  probs_vec.fill(DType(0));

  if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
    probs_vec.load(
        probs + row_idx * d
              + i * BLOCK_THREADS * VEC_SIZE
              + tx * VEC_SIZE);
  }

  threadlocal_max_val = max(
      threadlocal_max_val,
      BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(...)
          .Reduce<VEC_SIZE>(probs_greater_than_pivot, cub::Max()));
  __syncthreads();
}
```

`BlockReduce` 让一个 block 的线程共同求最大值。计算结果经 shared memory 广播给整个 block。

但是随后源码没有用这个最大值初始化 `high`，而是重新读取行首：

```cpp
threadlocal_max_val = temp_storage.max_val;  // 已算出的真实最大值
probs_vec.load(probs + row_idx * d);         // 重新读取行首

DType low = 0.0;
DType high = probs_vec[0];                   // 实际使用 token 0
```

这个差异是当前实现最重要的源码约束，详见后面的[重要：源码约束与风险](#重要源码约束与风险)。

#### Step 3：定义二分搜索状态

```cpp
DType low = 0.0;
DType high = probs_vec[0];
DType min_gt_low;
DType max_le_high;
DType sum_low(1);
DType eps = DType(1e-5);
```

核心函数为：

$$
f(x)=\sum_{a_i>x}a_i
$$

源码希望维持：

```text
f(low)  >= p
f(high) <  p
```

其中：

- `min_gt_low`：这一行中严格大于 `low` 的最小概率；
- `max_le_high`：这一行中小于等于 `high` 的最大概率；
- `eps = 1e-5`：停止搜索的数值容差；
- `sum_low`：源码会更新它，但后续没有再读取，当前不影响输出。

#### Step 4：每轮计算 `f(mid)`

```cpp
DType mid = (low + high) / DType(2);
DType threadlocal_sum(0);

for (uint32_t i = 0; i < ...; ++i) {
  // 每个线程读取 VEC_SIZE 个 probability。
  probs_vec.load(...);

  #pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    // 只累加严格大于 pivot 的概率。
    probs_greater_than_pivot[j] =
        probs_vec[j] > mid ? probs_vec[j] : DType(0);

    // 同时寻找 low / high 附近真实存在的离散概率值。
    if (probs_vec[j] > low && col < d) {
      min_gt_low = min(min_gt_low, probs_vec[j]);
    }
    if (probs_vec[j] <= high && col < d) {
      max_le_high = max(max_le_high, probs_vec[j]);
    }
  }

  threadlocal_sum += BlockReduce(...).Sum<VEC_SIZE>(
      probs_greater_than_pivot
  );
}
```

#### Step 5：根据累计概率移动边界

```cpp
if (threadlocal_sum >= p) {
  // 当前保留的概率质量仍足够，可以提高门槛、尝试少留 token。
  low = mid;
  sum_low = threadlocal_sum;
} else {
  // 当前保留质量不足，需要降低门槛、多留 token。
  high = min(mid, max_le_high);
}

// 当相邻离散概率值之间已经小于 eps 时结束。
} while (max_le_high - min_gt_low > eps);
```

这里不是在 token 数组下标上二分，而是在概率数值区间 `[low, high]` 上二分。每一轮仍需扫描整行并做一次 block reduction。

#### Step 6：生成 bool mask

```cpp
for (uint32_t i = 0; i < ...; ++i) {
  probs_vec.load(...);

  #pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    mask[row_idx * d + col] =
        probs_vec[j] > low ? 1 : 0;
  }
}
```

Python 创建的是 `torch.bool`，C++ 侧通过 `uint8_t*` 写入 `0/1`。最终每个位置表示对应 KV token 是否保留。

### TopPReturnIndicesKernel

源码位置：`csrc/include/sampling.cuh:252-394`。

前半部分与 mask kernel 相同：取得 row 和 p、扫描概率、搜索 `low`。区别在最后不写同形状 mask，而是压紧选中索引。

#### Step 1：每个线程统计自己选中了几个元素

```cpp
int thread_count = 0;
bool selected[VEC_SIZE];

#pragma unroll
for (uint32_t j = 0; j < VEC_SIZE; ++j) {
  const uint32_t col = base_col + j;
  selected[j] = col < d && probs_vec[j] > low;
  thread_count += selected[j] ? 1 : 0;
}
```

#### Step 2：BlockScan 得到紧凑写入位置

```cpp
int thread_offset;
int block_count;

BlockScan<int, BLOCK_THREADS, SCAN_ALGO>(...)
    .ExclusiveSum(
        thread_count,
        thread_offset,
        block_count
    );
```

- `thread_offset`：当前线程之前一共有多少个选中项；
- `block_count`：这一轮整个 block 一共选中了多少项；
- `row_count`：之前循环片段已经选中的累计数量。

所以当前线程从下面的位置开始写：

```cpp
uint32_t write_offset = row_count + thread_offset;
```

#### Step 3：写本地索引或映射后的原始索引

```cpp
if (selected[j]) {
  const uint32_t col = base_col + j;

  // output_stride 就是 output_indices 最后一维的容量。
  if (write_offset < output_stride) {
    output_indices[row_idx * output_stride + write_offset] =
        input_indices == nullptr
            ? static_cast<int32_t>(col)
            : input_indices[row_idx * d + col];
  }

  // 即使 buffer 已满，真实 count 仍继续增加。
  ++write_offset;
}
```

写入顺序来自对原始列的顺序扫描，不是按 probability 从大到小排序。输出表达“哪些 token 被选中”，不表达 token 的概率排名。

#### Step 4：保存真实 count

```cpp
if (threadIdx.x == 0) {
  output_counts[row_idx] = row_count;
}
```

`row_count` 不受 `output_stride` 截断，所以可能出现：

```text
output_indices capacity = 64
output_counts[row]      = 100

实际写入索引数          = 64
真实被 Top-p 选中的数量  = 100
```

### TopPReturnMask launcher

源码位置：`csrc/include/sampling.cuh:396-413`。

```cpp
template <typename DType>
cudaError_t TopPReturnMask(..., uint32_t num_blocks, ..., uint32_t d, ...) {
  constexpr uint32_t BLOCK_THREADS = 1024;

  // 尝试让一次内存操作搬运 16 bytes，同时保证向量长度整除 d。
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  dim3 nblks(num_blocks);       // 每个 row 一个 block
  dim3 nthrs(BLOCK_THREADS);    // 每个 block 1024 threads

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = TopPReturnMaskKernel<
        BLOCK_THREADS,
        REDUCE_ALGO,
        VEC_SIZE,
        DType>;

    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    cudaLaunchKernel(
        (void*)kernel,
        nblks,
        nthrs,
        args,
        smem_size,
        stream);
  });

  return cudaSuccess;
}
```

FP32 时 `16 / sizeof(float) = 4`，所以 `VEC_SIZE` 会根据 `kv_len` 是否能整除 4，在合法对齐值中选择。

### TopPReturnIndices launcher

源码位置：`csrc/include/sampling.cuh:415-435`。

它与 mask launcher 的 launch 配置相同，只是：

1. shared memory 类型换成包含 BlockScan 空间的 `RenormIndicesTempStorage`；
2. 参数多了 `output_indices`、`output_counts`、`input_indices` 和 `output_stride`；
3. 启动的是 `TopPReturnIndicesKernel`。

```cpp
void* args[] = {
    &probs,
    &output_indices,
    &output_counts,
    &input_indices,
    &top_p_arr,
    &top_p_val,
    &d,
    &output_stride,
};
```

### 重要：CUDA 中的 `__global__`、`__shared__`、`__syncthreads()`

| 语法 | 含义 | 本文件中的用途 |
| --- | --- | --- |
| `__global__` | 函数在 GPU 上执行、由 host 启动 | 定义两个 Top-p kernel |
| `blockIdx.x` | 当前 block 编号 | 选择当前 probability row |
| `threadIdx.x` | block 内线程编号 | 决定当前线程处理的列 |
| `extern __shared__` | launch 时动态指定大小的 block 共享内存 | 保存 CUB reduce / scan 临时数据和 block 聚合结果 |
| `__syncthreads()` | 等待同一 block 的所有线程到达 | 保证 shared memory 写完后再读 |
| `#pragma unroll` | 提示编译器展开固定次数循环 | 展开每线程 `VEC_SIZE` 个元素的处理 |

一次 block reduction / scan 后，如果没有同步就复用 shared memory，其他线程可能还没有完成写入。因此这里的 `__syncthreads()` 是正确读取聚合结果的必要条件，不是普通的性能装饰。

### 重要：为什么不用完整排序

传统写法通常是：

```text
sort descending → cumulative sum → 找到首次达到 p 的位置
```

当前 kernel 改为：

```text
对概率值做二分 → 每轮扫描并 reduction → 得到数值分界 → 写 mask / indices
```

两者的主要区别：

| 方案 | 需要的核心操作 | 结果顺序 | 边界并列值 |
| --- | --- | --- | --- |
| 排序 Top-p | 排序 + prefix sum | 概率降序 | 可以借助排序次序只取一部分 |
| 当前 kernel | 多轮全行扫描 + reduction | mask 保持原位置；indices 按原列位置 | 单一 `>` 阈值无法拆分并列项 |

源码没有给出理论或实测复杂度结论，因此这里只能确认实现策略，不能仅凭代码断言它在所有序列长度和概率分布上都比排序更快。性能应以 `benchmark/efficiency/bench_top_p.py` 的实测为准。

---

## 重要：源码约束与风险

### 1. mask 路径依赖 `q_len == 1`

`csrc/src/sampling.cu:27-38` 注释输入为四维，但 block 数只取：

```cpp
num_blocks = batch_size * num_heads;
```

正确覆盖全部四维 row 理论上应包含 `q_len`。indices 路径在 `csrc/src/sampling.cu:118-136` 使用：

```cpp
num_rows = probs.numel() / kv_seq_len;
```

它等价于 `batch_size × num_heads × q_len`。

当前 `attention.py` 的 Top-p 发生在单 token Decode，`q_len == 1`，所以与现有主调用路径对齐；若单独把 mask 接口用于 Prefill 或多 query 输入，后面的 query row 不会被 kernel 覆盖。

### 2. 搜索上界依赖 token 0 是最大概率

两个 kernel 都先用 BlockReduce 算出了真实行最大值：

```cpp
threadlocal_max_val = temp_storage.max_val;
```

但实际初始化却是：

```cpp
DType high = probs_vec[0];
```

源码注释把它称为 `attention sink`。这意味着二分不变量 `f(high) < p` 只有在第 0 个 token 是最大概率（或至少 `f(probs[0]) < p`）时才可靠。

一个最简单的反例：

```text
probs = [0.1, 0.9]
p = 0.5

标准 Top-p：只需保留概率 0.9 的 token
当前上界：high = probs[0] = 0.1
结果可能保留 [0.1, 0.9]，发生过度保留
```

仓库 benchmark 有意构造 token 0 较大的 logits：

```python
# benchmark/efficiency/bench_top_p.py:37-38
logits[:, :, :, 0].uniform_(1.5, 2.2)
logits[:, :, :, 1:].uniform_(-9.0, -1.0)
```

因此 benchmark 覆盖的是 attention-sink 前提成立的分布，并不能证明任意概率排列都正确。

### 3. `threshold` 没有在当前调用链中做范围检查

数学上的 Top-p 通常要求：

$$
0 < p \le 1
$$

`top_p.py`、`sampling.py` 和本仓库的 C++ 桥接没有显式拒绝 `p <= 0` 或 `p > 1`。调用者应自行保证范围有效。

### 4. `counts` 与 buffer 容量是两个概念

```text
counts[row] = 真实选中数量
output_indices.shape[-1] = 最多能写入的索引数量
```

`counts > capacity` 不是 kernel 报错，而是索引被静默截断。构造后续 ragged attention 元数据前，必须确保容量不小于 count；否则按照 count 读取会把未初始化区域误当成有效索引，甚至造成 shape / 内存语义错误。

### 5. indices 不是概率降序

kernel 按原始列遍历并通过 BlockScan 压紧，所以 indices 表示选中集合的原位置顺序。后续若需要“概率最高的 token 在最前”，还需要另外排序，不能把该输出当成 `torch.topk` 的降序结果。

### 6. `llama_patch.py` 中的调用结果当前没有被接收

`twilight/model/llama_patch.py:122-133` 调用了：

```python
top_p_unnormalized(estimated_weights, 0.85)
```

但没有保存返回的 mask / indices，随后直接把 `kv_seq_len` 设成固定的 `self.B1`。因此仅从当前源码看，这个调用没有把 Top-p 选择结果接入后续 page indices；它不能作为“Top-p 已完成端到端稀疏 KV 选择”的证据。

`twilight/pyimpl/attention.py` 才真正接收 `pruned_mask` 并用于 attention 剪枝。

### 7. 旧 `model/wrapper.py` 的 TopPWrapper 会再次 Softmax

`twilight/model/wrapper.py:178-196` 的 `random_weights()` 已先执行 Softmax，但 `forward()` 又把结果传给名为 `top_p_unnormalized()` 的接口，因此会发生第二次 Softmax。新的 `benchmark/efficiency/bench_top_p.py:53-64` 则正确地把 logits 交给 `top_p_unnormalized()`，把 probabilities 直接交给 kernel 接口。

---

## 输出接口选择建议

```text
只需在 Python attention 中屏蔽 token
        └── top_p_unnormalized → bool mask

需要把稀疏 token 位置交给后续 kernel
        └── top_p_unnormalized_return_indices → indices + counts

同 shape、同容量反复执行，希望复用输出 buffer
        └── top_p_unnormalized_return_indices_out
```

选择 indices 接口时，应始终一起携带 `counts`；单独的 `indices` Tensor 无法说明每一行有多少槽位有效。

## 最终总结

```text
top_p.py 的职责
├── 接收未归一化 logits
├── 用 FP32 Softmax 得到概率
└── 选择 mask / indices / 预分配 indices 接口

sampling.py 的职责
├── 选择预编译或 JIT CUDA module
├── 注册 PyTorch custom op
├── 分配或接收输出 Tensor
├── scalar p / per-row p 参数适配
└── 传递当前 CUDA stream

sampling.cu 的职责
├── 检查 PyTorch Tensor 基本契约
├── 提取 shape、dtype、data_ptr
├── indices 路径把前三维展平成 row
├── mask 路径按 bs × heads 启动（依赖 q_len = 1）
└── 进入模板 CUDA launcher

sampling.cuh 的职责
├── 每个 row 启动一个 block
├── 通过多轮 reduction 搜索概率分界 low
├── probability > low → bool mask
└── probability > low → BlockScan → compact indices + true counts
```

> [!TIP]
>
> CheckList 最终复核
>
> - [x] TL;DR：文档开头已给出入口、算法、输出和关键限制。
> - [x] 整体流程：已覆盖模型调用到 CUDA kernel 再返回 mask / indices 的全链路。
> - [x] 清晰拆分步骤：三个 Python API、包装层、C++ 桥接、两个 CUDA kernel 分别拆解。
> - [x] 每步代码块和注释：各函数与 kernel 阶段均附有带注释代码块。
> - [x] 重要语法 / 模型推理：已单列 Softmax 精度、attention Top-p、类型提示、custom op、stream、CUDA 语法和源码风险。
