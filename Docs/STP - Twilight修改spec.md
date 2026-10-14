**目标：在 Twilight 的 Top-p 前增加 Q/K MSB-slice 先验，以“高 4-bit × 高 4-bit”筛出高召回候选，再仅对候选执行完整“8-bit × 8-bit”打分和最终 Top-p。**

本文约定每个需要新增或修改的文件都是一级标题。代码块中的行末标记遵循：

- `# 增`：新增行；
- `# 删`：删除行；
- C++/JSONC 中用合法注释包住同一标记，即 `// # 增`、`// # 删`；
- 功能说明注释放在它所解释的代码上一行；
- 行末标记按代码块尽量纵向对齐；正式落地 JSON 时必须移除 JSONC 注释和行末标记。

## 已固定的算法语义

| 项目 | 规格 |
|---|---|
| 量化 | Q、K 分别做逐 token、逐 head、沿 `head_dim` 的对称 INT8 量化，范围 `[-127, 127]`，scale 为 `amax(abs(x))/127`。 |
| MSB-slice | 对 INT8 二补码做算术右移 `8-msb_bits`；默认 `msb_bits=4`，范围为 `[-8, 7]`。 |
| MSB 重构 | 高 4-bit 点积乘 `2^(2*(8-msb_bits))`，再乘 Q/K scale并除以 `sqrt(head_dim)`。 |
| B0 | 现有 Selector（Quest、DS、Streaming 等）产生的候选 mask。`selector=none` 时 B0 为全序列。 |
| Bpre | 在 B0 内，对 MSB logits 做 `prior_threshold` Top-p；默认建议从 `0.95` 开始搜索。 |
| B1 | 在 Bpre 内，对完整 INT8×INT8 logits 重新归一化并执行最终 `threshold` Top-p；默认保持 Twilight 的 `0.85`。 |
| 最终 attention 权重 | 默认仍使用原始浮点 QK logits，只用 STP 的 B1 mask裁剪；`use_estimated_weights_in_attn=true` 时才改用完整 INT8×INT8 logits。 |
| 正确性参考 | “Full-INT8 Top-p”：在同一个 B0 上直接计算完整 INT8×INT8 logits并做最终 Top-p。 |
| 必须满足 | `0 < threshold <= prior_threshold <= 1`、`quant_bits=8`、`1 <= msb_bits <= 8`、`min_tokens >= 1`。 |

## 端到端数据流

```text
Q_fp / K_fp
  -> per-token symmetric INT8 quantization
  -> Q_int8 / K_int8 + q_scale / k_scale
  -> arithmetic MSB slice (default high 4 bit)
  -> MSB QK over B0
  -> prior Top-p(p0) -> Bpre
  -> full INT8 QK only over Bpre
  -> final Top-p(p1) -> B1
  -> original FP attention logits masked by B1
  -> softmax -> V aggregation
```

## 分阶段落地要求

| 阶段 | 目的 | 完成标准 |
|---|---|---|
| P0：Python correctness | 验证量化定义、候选关系、LongBench/Passkey 精度和预算。完整 INT8 打分允许暂时 dense 计算。 | 单元测试通过；`B1 ⊆ Bpre ⊆ B0`；能输出 B0/Bpre/B1；LongBench 可用新配置运行。 |
| P1：Triton prototype | K 的高 4 bit packed 存储；MSB 阶段扫 B0，完整 INT8 阶段只扫 Bpre。 | 与 Python reference mask/indices一致；报告候选召回、预算和各阶段延迟。 |
| P2：端到端性能集成 | 把真实量化 K/K-MSB cache和 STP indices 接到 paged attention，而不是使用合成 cache。 | prefill/decode cache正确更新；STP indices真实驱动 attention；端到端延迟优于 Full-INT8 Top-p。 |

P0/P1 是本 spec 给出具体代码的主范围。现有 `twilight/model/llama_patch.py` 的 prefill、cache和 selector 数据流尚未接通，因此 P2 不应在该原型上直接堆补丁；应在 P0/P1 验证完成后单独重构，本文末尾给出它的替换边界。

## 与当前仓库实现的差异

当前源码并不存在一条已经接通的“Q8×K8 Top-p”端到端路径：`pyimpl/attention.py` 的 `MIN_MAX_QUANT/MAX_QUANT` 只伪量化 K 后又转回浮点，Q 仍是浮点；`model/llama_patch.py`/FTKA 路径主要是浮点 Q × INT4/INT8 K，且选择结果尚未接到真实 attention。因此本 spec 会先建立一个定义明确的 **Full-INT8 Top-p reference/baseline**，再比较 STP。实验报告不得把旧的 K-only 伪量化路径直接称为 8-bit×8-bit baseline。

# `twilight/pyimpl/state.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| `typing` 导入 | 只有 `List`。 | 增加 `Optional`。 | STP 统计为空时返回 `None`。 | 静态类型无未导入名字。 |
| `WeightPrunerType` | 只有 NONE/THRESHOLD/TOP_P。 | 新增 `STP=3`。 | 让 JSON 中的 `"type":"stp"` 可被统一配置系统解析。 | `WeightPrunerType.from_str("stp")` 成功。 |
| `HistoryBudgetInfo.reset` | 只清 B1 与 B0。 | 增加 STP selector/prior 分阶段预算容器。 | LongBench 必须能看到 B0、Bpre、B1，而非只有最终预算。 | reset 后无跨样本残留。 |
| `HistoryBudgetInfo` 方法 | 无 STP 专项统计。 | 新增 `update_stp`、query 平均和总平均方法。 | 量化 STP 先验的压缩率与开销收益。 | 非 STP 返回 `None`，STP 返回 Python float。 |

```python
# 扩展类型标注以支持“无 STP 数据”的返回值。                                      # 增
from typing import List, Optional                                                  # 增
from typing import List                                                            # 删


class WeightPrunerType(Enum):
    """The type of pruner applied to estimated weights."""

    NONE = 0
    THRESHOLD = 1
    TOP_P = 2
    # 注册两阶段 MSB-slice Top-p pruner。                                          # 增
    STP = 3                                                                        # 增

    @classmethod
    def from_str(cls, algo_str: str) -> "WeightPrunerType":
        return cls[algo_str.upper()]


class HistoryBudgetInfo:
    def reset(self) -> None:
        self._budget_num = []
        self._B0 = []
        self._last_layer_id = -999
        # 按 query/layer 保存 STP 进入先验前的 selector 预算和先验后的预算。          # 增
        self._stp_budget_num = []                                                   # 增
        # STP 使用独立层游标，避免与最终 B1 的 update() 相互干扰。                  # 增
        self._stp_last_layer_id = -999                                              # 增

    # 记录同一层进入 STP 的 B0 和 MSB 先验产生的 Bpre。                            # 增
    def update_stp(                                                                # 增
        self,                                                                      # 增
        layer_id: int,                                                             # 增
        selector_mask: torch.Tensor,                                               # 增
        prior_mask: torch.Tensor,                                                  # 增
    ) -> None:                                                                     # 增
        # 将 token 维压缩成逐 head budget。                                        # 增
        selector_budget = torch.sum(selector_mask, dim=-1).flatten()                # 增
        prior_budget = torch.sum(prior_mask, dim=-1).flatten()                      # 增
        # 层号不连续表示开始了一个新 query。                                       # 增
        if layer_id != self._stp_last_layer_id + 1:                                 # 增
            self._stp_budget_num.append({})                                         # 增
        # 使用显式阶段名，避免把 B0/Bpre 的语义写反。                              # 增
        self._stp_budget_num[-1][layer_id] = {                                      # 增
            "selector": selector_budget,                                           # 增
            "prior": prior_budget,                                                 # 增
        }                                                                           # 增
        self._stp_last_layer_id = layer_id                                          # 增

    # 返回每个 query 在指定 STP 阶段上的层/head 平均预算。                         # 增
    def get_avg_stp_budget_per_query(self, stage: str) -> List[float]:               # 增
        # 只接受稳定的外部字段名。                                                  # 增
        if stage not in ("selector", "prior"):                                   # 增
            raise ValueError(f"Unknown STP budget stage: {stage}")                 # 增
        ret = []                                                                    # 增
        for query_budget_info in self._stp_budget_num:                              # 增
            layer_avg = []                                                          # 增
            for layer_id, stage_budget_info in query_budget_info.items():            # 增
                # 与现有统计保持一致：跳过前两层。                                 # 增
                if layer_id > 1:                                                    # 增
                    budget = stage_budget_info[stage]                               # 增
                    layer_avg.append(torch.mean(budget.float()).item())              # 增
            # 空 query 不写入 NaN。                                                 # 增
            if layer_avg:                                                           # 增
                ret.append(float(np.mean(layer_avg)))                               # 增
        return ret                                                                  # 增

    # 汇总指定阶段；非 STP 配置返回 None。                                         # 增
    def get_total_avg_stp_budget(self, stage: str) -> Optional[float]:               # 增
        per_query = self.get_avg_stp_budget_per_query(stage)                         # 增
        if not per_query:                                                           # 增
            return None                                                             # 增
        return float(np.mean(per_query))                                            # 增
```

# `twilight/pyimpl/quantize.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 文件末尾、示例入口之前 | 只有量化后立即反量化的模拟函数。 | 新增“返回整数和 scale”的对称 INT8 量化。 | STP 必须真正访问整数 bit slice。 | 反量化误差满足半个量化步长附近的容差。 |
| 同上 | 无 bit-slice。 | 新增算术右移的 `signed_msb_slice_int8`。 | 明确定义负数的二补码高位语义。 | 输出范围在 4-bit 时为 `[-8,7]`。 |
| 同上 | 无 packed INT4。 | 新增 pack/unpack reference。 | P1 的 K-MSB cache 需要每字节保存两个有符号 4-bit。 | `unpack(pack(x)) == x`。 |

保留现有 `min_max_per_token_quant_kv` 和 `max_per_token_quant_kv`，在其后、`if __name__ == "__main__"` 之前加入：

```python
# 新增：逐 token、逐 head 的对称 INT8 量化，返回整数值与 FP32 scale。              # 增
def symmetric_per_token_quant_int8(                                                 # 增
    tensor: torch.Tensor,                                                           # 增
    eps: float = 1e-8,                                                              # 增
) -> tuple[torch.Tensor, torch.Tensor]:                                              # 增
    # 禁止静默接受非浮点输入。                                                      # 增
    if not tensor.is_floating_point():                                               # 增
        raise TypeError("symmetric_per_token_quant_int8 expects a float tensor")    # 增
    # 沿 head_dim 计算独立 scale，并用 FP32 保证量化稳定。                          # 增
    abs_max = tensor.detach().abs().amax(dim=-1, keepdim=True).float()               # 增
    scale = abs_max.clamp_min(eps) / 127.0                                           # 增
    # 使用 [-127,127] 对称范围，避免 -128 的绝对值不对称。                         # 增
    quantized = torch.round(tensor.float() / scale).clamp(-127, 127)                 # 增
    return quantized.to(torch.int8), scale                                           # 增


# 新增：从 INT8 二补码中取得有符号 MSB-slice；右移必须是算术右移。                 # 增
def signed_msb_slice_int8(                                                          # 增
    tensor: torch.Tensor,                                                           # 增
    msb_bits: int = 4,                                                              # 增
) -> torch.Tensor:                                                                 # 增
    # 只接受真实 INT8，避免对浮点 tensor 误做 bit 运算。                           # 增
    if tensor.dtype != torch.int8:                                                   # 增
        raise TypeError("signed_msb_slice_int8 expects torch.int8")                 # 增
    if not 1 <= msb_bits <= 8:                                                       # 增
        raise ValueError("msb_bits must be in [1, 8]")                              # 增
    shift = 8 - msb_bits                                                             # 增
    # 先升到 int16，确保负数右移后仍保留符号。                                     # 增
    return torch.bitwise_right_shift(tensor.to(torch.int16), shift).to(torch.int8)   # 增


# 新增：把最后一维相邻两个 signed-int4 打包到一个 uint8。                          # 增
def pack_signed_int4(tensor: torch.Tensor) -> torch.Tensor:                          # 增
    if tensor.dtype != torch.int8:                                                   # 增
        raise TypeError("pack_signed_int4 expects torch.int8")                     # 增
    if tensor.shape[-1] % 2 != 0:                                                    # 增
        raise ValueError("the packed dimension must be even")                      # 增
    if torch.any((tensor < -8) | (tensor > 7)):                                      # 增
        raise ValueError("signed int4 values must be in [-8, 7]")                  # 增
    # 低 channel 放低 nibble，高 channel 放高 nibble。                             # 增
    low = tensor[..., 0::2].to(torch.int16) & 0x0F                                  # 增
    high = (tensor[..., 1::2].to(torch.int16) & 0x0F) << 4                           # 增
    return (low | high).to(torch.uint8).contiguous()                                 # 增


# 新增：CPU/PyTorch correctness 使用的 signed-int4 解包 reference。                # 增
def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:                        # 增
    if packed.dtype != torch.uint8:                                                  # 增
        raise TypeError("unpack_signed_int4 expects torch.uint8")                  # 增
    low = (packed.to(torch.int16) & 0x0F)                                            # 增
    high = ((packed.to(torch.int16) >> 4) & 0x0F)                                    # 增
    # 对 nibble 做显式符号扩展。                                                     # 增
    low = torch.where(low >= 8, low - 16, low)                                       # 增
    high = torch.where(high >= 8, high - 16, high)                                   # 增
    output = torch.empty(                                                           # 增
        (*packed.shape[:-1], packed.shape[-1] * 2),                                  # 增
        dtype=torch.int8,                                                           # 增
        device=packed.device,                                                       # 增
    )                                                                               # 增
    output[..., 0::2] = low.to(torch.int8)                                           # 增
    output[..., 1::2] = high.to(torch.int8)                                          # 增
    return output                                                                    # 增
```

# `twilight/pyimpl/stp.py`（新增）

| 组成 | 修改意见 | 原因 | 验收 |
|---|---|---|---|
| `STPResult` | 同时返回最终 mask、先验 mask、MSB logits、完整 INT8 logits。 | attention 需要 mask/可选 estimated weights；测试需要中间结果。 | shape 与 QK logits一致。 |
| `_quantized_qk_logits` | 统一整数点积、scale恢复和 `sqrt(D)`。 | 避免 MSB/full 两条公式漂移。 | 与手工 dequant-matmul 对齐。 |
| `_ensure_min_tokens` | 在允许集合中保证最少 token。 | Top-p 严格阈值、相同分数或极端输入不能返回空 mask。 | 不会选到 B0/Bpre 外。 |
| `stp_pruner` | 完整实现 B0→Bpre→B1。 | P0 正确性主实现。 | `final ⊆ prior ⊆ selector`。 |

```python
# 新增：STP Python correctness/reference 实现。                                    # 增
import math                                                                         # 增
from dataclasses import dataclass                                                    # 增

import torch                                                                        # 增

# 复用统一 INT8 量化与 MSB 定义。                                                   # 增
from .quantize import signed_msb_slice_int8, symmetric_per_token_quant_int8          # 增
# 复用 Twilight 当前的 Top-p 语义和 CUDA sampling kernel。                          # 增
from .top_p import top_p_unnormalized                                                # 增


# 保存两阶段选择结果，便于 attention 接入和实验诊断。                              # 增
@dataclass(frozen=True)                                                              # 增
class STPResult:                                                                     # 增
    final_mask: torch.Tensor                                                         # 增
    prior_mask: torch.Tensor                                                         # 增
    msb_logits: torch.Tensor                                                         # 增
    full_logits: torch.Tensor                                                        # 增


# 把不允许参与本阶段 Top-p 的位置写成当前 dtype 最小有限值。                       # 增
def _masked_logits(logits: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor: # 增
    return logits.masked_fill(~allowed_mask, torch.finfo(logits.dtype).min)           # 增


# 在 allowed_mask 内保证每行至少选择 min_tokens 个 token。                         # 增
def _ensure_min_tokens(                                                             # 增
    logits: torch.Tensor,                                                           # 增
    selected_mask: torch.Tensor,                                                    # 增
    allowed_mask: torch.Tensor,                                                     # 增
    min_tokens: int,                                                                # 增
) -> torch.Tensor:                                                                 # 增
    if min_tokens <= 0:                                                             # 增
        return selected_mask & allowed_mask                                          # 增
    seq_len = logits.shape[-1]                                                       # 增
    k = min(min_tokens, seq_len)                                                     # 增
    # 先屏蔽不允许位置，再寻找 fallback token。                                     # 增
    fallback_logits = _masked_logits(logits, allowed_mask)                           # 增
    fallback_indices = fallback_logits.topk(k=k, dim=-1).indices                     # 增
    fallback_mask = torch.zeros_like(allowed_mask)                                   # 增
    fallback_mask.scatter_(dim=-1, index=fallback_indices, value=True)               # 增
    fallback_mask &= allowed_mask                                                    # 增
    # allowed token 少于 min_tokens 时，目标自动收缩为 allowed 数量。               # 增
    target = torch.minimum(                                                          # 增
        allowed_mask.sum(dim=-1, keepdim=True),                                      # 增
        torch.full_like(allowed_mask.sum(dim=-1, keepdim=True), min_tokens),          # 增
    )                                                                               # 增
    need_fallback = selected_mask.sum(dim=-1, keepdim=True) < target                  # 增
    return (selected_mask | (fallback_mask & need_fallback)) & allowed_mask          # 增


# 统一计算整数 QK、scale 恢复和 attention 温度。                                   # 增
def _quantized_qk_logits(                                                           # 增
    query_int: torch.Tensor,                                                        # 增
    key_int: torch.Tensor,                                                          # 增
    query_scale: torch.Tensor,                                                      # 增
    key_scale: torch.Tensor,                                                        # 增
    reconstruction_shift: int = 0,                                                  # 增
) -> torch.Tensor:                                                                 # 增
    # P0 使用 FP32 matmul承载整数 reference；P1 会替换成只读候选的 Triton kernel。   # 增
    integer_dot = torch.matmul(                                                      # 增
        query_int.float(),                                                          # 增
        key_int.float().transpose(2, 3),                                             # 增
    )                                                                               # 增
    # 两个操作数都由 MSB 值左移 reconstruction_shift 重构。                        # 增
    reconstruction = float(1 << (2 * reconstruction_shift))                         # 增
    dequant_scale = query_scale * key_scale.transpose(2, 3)                         # 增
    return (                                                                         # 增
        integer_dot                                                                 # 增
        * reconstruction                                                            # 增
        * dequant_scale                                                             # 增
        / math.sqrt(query_int.shape[-1])                                             # 增
    )                                                                               # 增


# 实现 B0 -> MSB Top-p(Bpre) -> full INT8 Top-p(B1)。                               # 增
def stp_pruner(                                                                     # 增
    query_states: torch.Tensor,                                                     # 增
    key_states: torch.Tensor,                                                       # 增
    selector_mask: torch.Tensor,                                                    # 增
    prior_threshold: float,                                                         # 增
    threshold: float,                                                               # 增
    quant_bits: int = 8,                                                            # 增
    msb_bits: int = 4,                                                              # 增
    min_tokens: int = 1,                                                            # 增
) -> STPResult:                                                                    # 增
    # 第一版只定义 8-bit baseline，防止配置名与实现语义不一致。                    # 增
    if quant_bits != 8:                                                             # 增
        raise ValueError("STP P0 requires quant_bits == 8")                         # 增
    if not 1 <= msb_bits <= quant_bits:                                              # 增
        raise ValueError("msb_bits must be in [1, quant_bits]")                    # 增
    if not 0.0 < threshold <= prior_threshold <= 1.0:                                # 增
        raise ValueError("require 0 < threshold <= prior_threshold <= 1")           # 增
    if min_tokens < 1:                                                              # 增
        raise ValueError("min_tokens must be >= 1")                                # 增
    if query_states.ndim != 4 or key_states.ndim != 4:                               # 增
        raise ValueError("STP expects Q/K in [batch, heads, length, dim]")           # 增
    expected_mask_shape = (                                                         # 增
        query_states.shape[0],                                                      # 增
        query_states.shape[1],                                                      # 增
        query_states.shape[2],                                                      # 增
        key_states.shape[2],                                                        # 增
    )                                                                               # 增
    if selector_mask.shape != expected_mask_shape:                                  # 增
        raise ValueError(                                                           # 增
            f"selector_mask must have shape {expected_mask_shape}, "                # 增
            f"got {tuple(selector_mask.shape)}"                                    # 增
        )                                                                           # 增
    # B0 每行必须至少有一个 token，否则 Top-p 的归一化没有定义。                   # 增
    if not torch.all(selector_mask.any(dim=-1)):                                     # 增
        raise ValueError("selector_mask contains an empty row")                    # 增

    # Q/K 独立做逐 token、逐 head 对称 INT8 量化。                                 # 增
    query_int8, query_scale = symmetric_per_token_quant_int8(query_states)           # 增
    key_int8, key_scale = symmetric_per_token_quant_int8(key_states)                 # 增
    # 算术右移取得高位切片。                                                        # 增
    query_msb = signed_msb_slice_int8(query_int8, msb_bits)                          # 增
    key_msb = signed_msb_slice_int8(key_int8, msb_bits)                              # 增
    shift = quant_bits - msb_bits                                                    # 增

    # 第一阶段：在 B0 内用 MSB×MSB 形成先验分布。                                  # 增
    msb_logits = _quantized_qk_logits(                                               # 增
        query_msb,                                                                  # 增
        key_msb,                                                                    # 增
        query_scale,                                                                # 增
        key_scale,                                                                  # 增
        reconstruction_shift=shift,                                                 # 增
    )                                                                               # 增
    prior_mask = top_p_unnormalized(                                                 # 增
        _masked_logits(msb_logits, selector_mask),                                   # 增
        prior_threshold,                                                            # 增
    ) & selector_mask                                                               # 增
    prior_mask = _ensure_min_tokens(                                                 # 增
        msb_logits, prior_mask, selector_mask, min_tokens                            # 增
    )                                                                               # 增

    # 第二阶段：P0 为 correctness 暂时 dense 计算 full INT8 logits。                # 增
    full_logits = _quantized_qk_logits(                                              # 增
        query_int8,                                                                 # 增
        key_int8,                                                                   # 增
        query_scale,                                                                # 增
        key_scale,                                                                  # 增
    )                                                                               # 增
    # 最终 Top-p 只在 Bpre 内重新归一化。                                           # 增
    final_mask = top_p_unnormalized(                                                 # 增
        _masked_logits(full_logits, prior_mask),                                     # 增
        threshold,                                                                  # 增
    ) & prior_mask                                                                  # 增
    final_mask = _ensure_min_tokens(                                                 # 增
        full_logits, final_mask, prior_mask, min_tokens                              # 增
    )                                                                               # 增

    return STPResult(                                                               # 增
        final_mask=final_mask,                                                       # 增
        prior_mask=prior_mask,                                                       # 增
        msb_logits=msb_logits,                                                       # 增
        full_logits=full_logits,                                                     # 增
    )                                                                               # 增
```

# `twilight/pyimpl/attention.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 策略导入区 | 不认识 STP。 | 导入 `stp_pruner`。 | 接入 P0 reference。 | 模块可导入。 |
| Pruner 分发 | 只有 threshold/top-p。 | 新增 STP 分支，传入 Q/K、B0 mask和配置。 | STP 同时需要 bit-sliced Q/K，不能只接收已有 logits。 | `pruned_mask` 是 Tensor，shape 与 attention logits一致。 |
| STP 分支统计 | 只记录最终 mask。 | 在最终 `mask &= pruned_mask` 之前记录 selector mask与 prior mask。 | 形成 B0/Bpre/B1 统计闭环。 | LongBench 每样本可读取三阶段预算。 |
| 配置解析 | 不校验策略组合。 | STP 强制 estimator=none、检查必需参数。 | STP 内部已经定义 Full-INT8 estimator；禁止重复估计。 | 错误配置在 patch 模型前失败。 |

在策略导入区加入：

```python
# 导入 STP 两阶段 pruner。                                                         # 增
from .stp import stp_pruner                                                         # 增
```

在现有 Weight Pruner 分发中，把 STP 分支放在 `TOP_P` 分支之后：

```python
        if self.state.weight_pruner == WeightPrunerType.THRESHOLD:
            pruned_mask = elementwise_threshold(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )
        elif self.state.weight_pruner == WeightPrunerType.TOP_P:
            pruned_mask = top_p_unnormalized(
                estimated_weights,
                self.state.weight_pruner_args["threshold"],
            )
        # STP 自己生成 MSB 先验和完整 INT8 logits。                                 # 增
        elif self.state.weight_pruner == WeightPrunerType.STP:                       # 增
            stp_result = stp_pruner(                                                 # 增
                query_states,                                                       # 增
                key_states,                                                         # 增
                selector_mask=mask,                                                 # 增
                **self.state.weight_pruner_args,                                    # 增
            )                                                                       # 增
            # B1 是 STP 最终输出；后续仍与 B0 做防御性逻辑与。                     # 增
            pruned_mask = stp_result.final_mask                                     # 增
            # 让 use_estimated_weights_in_attn=true 时使用 Full-INT8 logits。       # 增
            estimated_weights = stp_result.full_logits                              # 增
            # 在覆盖 mask 前记录 B0 和 Bpre。                                       # 增
            if self.state.budget_info is not None:                                  # 增
                self.state.budget_info.update_stp(                                  # 增
                    self.layer_id,                                                  # 增
                    selector_mask=mask,                                             # 增
                    prior_mask=stp_result.prior_mask,                               # 增
                )                                                                   # 增

        if self.state.weight_pruner != WeightPrunerType.NONE:
            mask = mask & pruned_mask
```

在 `enable_sparse_attention` 拆出四组 type/args 后、创建统计器之前加入：

```python
    # STP 内部固定包含 MSB 和 Full-INT8 两级 estimator，禁止叠加旧 estimator。      # 增
    if weight_pruner_type.lower() == "stp":                                        # 增
        if weight_estimator_type.lower() != "none":                                # 增
            raise ValueError(                                                       # 增
                "STP requires weight_estimator.type == 'none'"                      # 增
            )                                                                       # 增
        # 配置文件必须显式写出研究变量，避免默认值污染实验可复现性。                # 增
        required_stp_args = {                                                       # 增
            "quant_bits",                                                         # 增
            "msb_bits",                                                           # 增
            "prior_threshold",                                                    # 增
            "threshold",                                                          # 增
            "min_tokens",                                                         # 增
        }                                                                           # 增
        missing_stp_args = required_stp_args - weight_pruner_args.keys()             # 增
        if missing_stp_args:                                                        # 增
            raise ValueError(                                                       # 增
                f"Missing STP args: {sorted(missing_stp_args)}"                    # 增
            )                                                                       # 增
```

实现时不要改变下面两点：

1. `mask = mask & pruned_mask` 仍由公共路径执行，维持所有 pruner 的一致语义；
2. 默认 `use_estimated_weights_in_attn=false`，最终 Value 聚合仍使用原始浮点 logits。

# `twilight/pyimpl/__init__.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 公共导出 | 只导出 enable/reset。 | 导出 `STPResult` 和 `stp_pruner`。 | 单元测试、研究脚本和外部调用不应依赖私有路径。 | `from twilight.pyimpl import stp_pruner` 成功。 |

```python
"""Python implementation of sparse attention algorithm."""

from .attention import enable_sparse_attention, reset_sparse_config
# 暴露 STP reference 和结构化中间结果。                                           # 增
from .stp import STPResult, stp_pruner                                               # 增
```

# `benchmark/LongBench/pred.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 每样本统计读取 | 只读最终平均 budget 和 score。 | reset 前读取 STP selector/prior预算。 | 研究 STP 必须同时报告 B0、Bpre、B1。 | 非 STP 输出 null，STP 输出数值。 |
| JSONL 字段 | 只有 `budget`。 | 保留兼容字段，并新增 `stp_selector_budget`、`stp_prior_budget`、`stp_final_budget`。 | 旧 eval 不受影响，新分析脚本可直接读取。 | `eval.py` 仍可原样评分。 |

替换 `get_pred` 中“Record budget”块；旧的两行读取保留为上下文，新增字段如下：

```python
        # Record budget
        avg_budget = budget_info.get_total_avg_budget()
        avg_score = score_info.get_total_avg_score()
        # 在 reset 前取得 STP 的 B0/Bpre；非 STP 返回 None。                       # 增
        avg_stp_selector_budget = budget_info.get_total_avg_stp_budget("selector")  # 增
        avg_stp_prior_budget = budget_info.get_total_avg_stp_budget("prior")        # 增
        # 只有 STP 样本才把最终 budget重复命名为 B1。                               # 增
        avg_stp_final_budget = (                                                     # 增
            avg_budget if avg_stp_prior_budget is not None else None                 # 增
        )                                                                           # 增
        budget_info.reset()
        score_info.reset()

        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "budget": avg_budget,
                "score_sum": avg_score,
                # 保存进入 MSB 先验前的 B0。                                       # 增
                "stp_selector_budget": avg_stp_selector_budget,                    # 增
                # 保存 MSB 先验后的 Bpre。                                          # 增
                "stp_prior_budget": avg_stp_prior_budget,                          # 增
                # 显式命名最终 B1，同时保留旧 budget 字段兼容现有分析。             # 增
                "stp_final_budget": avg_stp_final_budget,                          # 增
            }
        )
```

无需修改 `benchmark/LongBench/eval.py` 和 `metrics.py`：它们只读取 `pred/answers/all_classes/length`，会忽略新增预算字段。

# `benchmark/configs/config_stp.json`（新增）

| 字段 | 建议值 | 说明 |
|---|---:|---|
| selector | `none` | 隔离测量 STP 自身，不叠加 Quest/DS。 |
| quant_bits / msb_bits | `8 / 4` | 对应 Full-INT8 baseline和高 4-bit先验。 |
| prior_threshold | `0.95` | 起始搜索点，不是最终固定结论。 |
| threshold | `0.85` | 与当前 Twilight Top-p 一致。 |
| min_tokens | `1` | 防止空 mask。 |

```jsonc
{                                                                                   // # 增
    "compressor": {                                                                // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "selector": {                                                                  // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "weight_estimator": {                                                          // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "weight_pruner": {                                                             // # 增
        "type": "stp",                                                            // # 增
        "quant_bits": 8,                                                           // # 增
        "msb_bits": 4,                                                             // # 增
        "prior_threshold": 0.95,                                                   // # 增
        "threshold": 0.85,                                                         // # 增
        "min_tokens": 1                                                            // # 增
    },                                                                              // # 增
    "skip_first_two_layers": true,                                                 // # 增
    "use_estimated_weights_in_attn": false                                         // # 增
}                                                                                   // # 增
```

# `benchmark/configs/config_quest_stp.json`（新增）

| 字段 | 建议值 | 说明 |
|---|---:|---|
| Quest token budget | `8192` | B0，保持现有 Quest-Twilight 对比口径。 |
| Quest chunk size | `16` | 保持现有实现。 |
| STP prior/final p | `0.95 / 0.85` | Bpre/B1 两级动态预算。 |

```jsonc
{                                                                                   // # 增
    "compressor": {                                                                // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "selector": {                                                                  // # 增
        "type": "quest",                                                          // # 增
        "token_budget": 8192,                                                      // # 增
        "chunk_size": 16                                                           // # 增
    },                                                                              // # 增
    "weight_estimator": {                                                          // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "weight_pruner": {                                                             // # 增
        "type": "stp",                                                            // # 增
        "quant_bits": 8,                                                           // # 增
        "msb_bits": 4,                                                             // # 增
        "prior_threshold": 0.95,                                                   // # 增
        "threshold": 0.85,                                                         // # 增
        "min_tokens": 1                                                            // # 增
    },                                                                              // # 增
    "skip_first_two_layers": true,                                                 // # 增
    "use_estimated_weights_in_attn": false                                         // # 增
}                                                                                   // # 增
```

# `benchmark/configs/config_ds_stp.json`（新增）

| 字段 | 建议值 | 说明 |
|---|---:|---|
| DS `r/quant_bit` | `64 / 2` | 保持现有 DS selector。 |
| DS token budget | `8192` | B0。 |
| STP | 与 Quest-STP 相同 | 便于比较 selector 与 STP 的正交收益。 |

`config_path` 必须改成实验机器上的真实 channel profile；不要复制仓库现有的个人绝对路径。

```jsonc
{                                                                                   // # 增
    "compressor": {                                                                // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "selector": {                                                                  // # 增
        "type": "ds",                                                             // # 增
        "quant_bit": 2,                                                            // # 增
        "r": 64,                                                                   // # 增
        "config_path": "/ABSOLUTE/PATH/TO/channel_config.json",                   // # 增
        "token_budget": 8192,                                                      // # 增
        "selected_channel": "q"                                                   // # 增
    },                                                                              // # 增
    "weight_estimator": {                                                          // # 增
        "type": "none"                                                            // # 增
    },                                                                              // # 增
    "weight_pruner": {                                                             // # 增
        "type": "stp",                                                            // # 增
        "quant_bits": 8,                                                           // # 增
        "msb_bits": 4,                                                             // # 增
        "prior_threshold": 0.95,                                                   // # 增
        "threshold": 0.85,                                                         // # 增
        "min_tokens": 1                                                            // # 增
    },                                                                              // # 增
    "skip_first_two_layers": true,                                                 // # 增
    "use_estimated_weights_in_attn": false                                         // # 增
}                                                                                   // # 增
```

# `benchmark/scripts/run_passkey.sh`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| `algo_cfg_path` | 硬编码 Quest-1024。 | 默认改为 Quest-STP，并允许第一个 CLI 参数覆盖。 | 无需每次编辑脚本即可切换 baseline/STP。 | 默认运行 STP；传旧配置仍可复现实验。 |

```bash
# 原脚本固定使用 Quest-1024。                                                       # 删
algo_cfg_path=./configs/config_quest_1024.json                                       # 删
# 默认运行 Quest-STP；$1 非空时用调用方配置覆盖。                                  # 增
algo_cfg_path=${1:-./configs/config_quest_stp.json}                                  # 增
```

建议调用方式：`bash scripts/run_passkey.sh configs/config_stp.json` 或 `bash scripts/run_passkey.sh configs/config_quest_stp.json`。

# `csrc/include/sampling.cuh`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| `TopPReturnMaskKernel` 的二分上界 | 虽然前面已经归约出真实最大概率，随后却用 `probs[0]` 当上界。 | 使用 `threadlocal_max_val`。 | MSB 近似可能改变最大 token；STP 不能依赖 attention sink永远是最大值。 | 最大概率不在 token 0 时仍与 reference一致。 |
| `TopPReturnIndicesKernel` 的二分上界 | 同上。 | 同上。 | mask 与 indices必须具有相同语义。 | counts 等于 reference mask之和。 |

在两个 kernel 中分别替换相同代码。第一处位于 `TopPReturnMaskKernel`：

```cpp
  threadlocal_max_val = temp_storage.max_val;
  probs_vec.fill(DType(0));                                                        // # 删
  probs_vec.load(probs + row_idx * d);  // load attention sink                    // # 删

  DType low = 0.0, high = probs_vec[0];                                            // # 删
  // 使用整行真实最大概率作为二分上界，不再要求 token 0 是最大值。                 // # 增
  DType low = DType(0), high = threadlocal_max_val;                                // # 增
```

第二处位于 `TopPReturnIndicesKernel`：

```cpp
  threadlocal_max_val = temp_storage.max_val;
  probs_vec.fill(DType(0));                                                        // # 删
  probs_vec.load(probs + row_idx * d);                                             // # 删

  DType low = 0.0, high = probs_vec[0];                                            // # 删
  // indices kernel必须和 mask kernel使用同一真实最大值上界。                     // # 增
  DType low = DType(0), high = threadlocal_max_val;                                // # 增
```

注意：这不是 STP 专属优化，而是解除现有 Top-p kernel 的 sink 最大值假设；必须先单独跑 Top-p regression，再跑 STP。

# `twilight/kernel/cuda/sampling.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| JIT module name | 固定为 `twilight_sampling`。 | 修改为带 STP cachebuster 的新名字。 | 只改 header 时不能依赖所有 FlashInfer 版本都能发现依赖变化；新名字强制重新构建。 | 首次运行重新编译且 non-sink Top-p测试命中新实现。 |

```python
            module = load_cuda_ops(
                # 旧 JIT 名可能继续命中修改 sampling.cuh 之前的缓存。               # 删
                "twilight_sampling",                                               # 删
                # 修改 header 后使用新 module key 强制 cache miss/rebuild。         # 增
                "twilight_sampling_stp_v1",                                        # 增
                sources=[
                    TWILIGHT_CSRC_DIR / "sampling.cu",
                ],
```

如果后续继续修改 `sampling.cuh` 的 ABI/语义，应递增 `_v1`，并在 benchmark日志中记录该版本。

# `twilight/kernel/triton/stp.py`（新增）

| 组成 | 修改意见 | 原因 | 验收 |
|---|---|---|---|
| `_stp_msb_qk_kernel` | 输入 packed K-MSB、Q-MSB、B0 indices/counts，只输出 B0 上的近似 logits。 | K-MSB 带宽减半；兼容 Quest/DS 的压缩候选。 | 对齐 Python MSB logits。 |
| `_stp_refine_qk_kernel` | 输入 Bpre 原始 token indices/counts，只计算这些 token 的 Full-INT8 logits。 | 真正避免对 B0 内所有 token 做 8×8。 | kernel dot 数与 Bpre 而非 B0 成正比。 |
| Python wrappers | 做 shape/dtype/device/contiguous 校验和 launch。 | 错误尽早暴露，benchmark不直接拼 launch参数。 | 错误输入有明确异常。 |

本原型将 batch 与 head 展平为 row：Q `[R,D]`，K8 `[R,S,D]`，packed K-MSB `[R,S,D/2]`。B0/Bpre 都用 `[R,C]` int32 indices和 `[R]` int32 counts表达。

```python
# 新增：STP packed-MSB 与候选 Full-INT8 Triton kernels。                           # 增
import math                                                                         # 增

import torch                                                                        # 增
import triton                                                                       # 增
import triton.language as tl                                                        # 增


# 第一阶段只读取 packed K-MSB，并按 B0 indices计算高位先验分数。                   # 增
@triton.jit                                                                         # 增
def _stp_msb_qk_kernel(                                                             # 增
    Q_MSB,                                                                          # 增
    K_MSB_PACKED,                                                                   # 增
    Q_SCALE,                                                                        # 增
    K_SCALE,                                                                        # 增
    INPUT_INDICES,                                                                  # 增
    INPUT_COUNTS,                                                                   # 增
    OUT,                                                                            # 增
    stride_qr: tl.constexpr,                                                        # 增
    stride_qd: tl.constexpr,                                                        # 增
    stride_kr: tl.constexpr,                                                        # 增
    stride_ks: tl.constexpr,                                                        # 增
    stride_kp: tl.constexpr,                                                        # 增
    stride_ksr: tl.constexpr,                                                       # 增
    stride_kss: tl.constexpr,                                                       # 增
    stride_ir: tl.constexpr,                                                        # 增
    stride_ic: tl.constexpr,                                                        # 增
    stride_or: tl.constexpr,                                                        # 增
    stride_oc: tl.constexpr,                                                        # 增
    CANDIDATE_CAPACITY: tl.constexpr,                                               # 增
    HEAD_DIM: tl.constexpr,                                                         # 增
    RECONSTRUCTION: tl.constexpr,                                                   # 增
    INV_SQRT_HEAD_DIM: tl.constexpr,                                                # 增
    BLOCK_N: tl.constexpr,                                                          # 增
    BLOCK_D: tl.constexpr,                                                          # 增
):                                                                                 # 增
    row = tl.program_id(0)                                                          # 增
    candidate_block = tl.program_id(1)                                              # 增
    offs_n = candidate_block * BLOCK_N + tl.arange(0, BLOCK_N)                      # 增
    offs_d = tl.arange(0, BLOCK_D)                                                   # 增
    count = tl.load(INPUT_COUNTS + row)                                              # 增
    valid_n = (offs_n < CANDIDATE_CAPACITY) & (offs_n < count)                      # 增
    # 无效槽位用 token 0 占位，但所有读取和输出都受 valid_n mask保护。              # 增
    token_idx = tl.load(                                                            # 增
        INPUT_INDICES + row * stride_ir + offs_n * stride_ic,                       # 增
        mask=valid_n,                                                               # 增
        other=0,                                                                    # 增
    )                                                                               # 增
    q = tl.load(                                                                    # 增
        Q_MSB + row * stride_qr + offs_d * stride_qd,                               # 增
        mask=offs_d < HEAD_DIM,                                                     # 增
        other=0,                                                                    # 增
    ).to(tl.int32)                                                                  # 增
    # 每个 byte 含相邻两个 signed-int4 channel。                                    # 增
    packed = tl.load(                                                               # 增
        K_MSB_PACKED                                                                # 增
        + row * stride_kr                                                           # 增
        + token_idx[:, None] * stride_ks                                            # 增
        + (offs_d[None, :] // 2) * stride_kp,                                       # 增
        mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),                       # 增
        other=0,                                                                    # 增
    ).to(tl.int32)                                                                  # 增
    nibble_shift = (offs_d[None, :] & 1) * 4                                        # 增
    nibble = (packed >> nibble_shift) & 0x0F                                        # 增
    # 将 0..15 nibble 显式符号扩展到 -8..7。                                       # 增
    k = tl.where(nibble >= 8, nibble - 16, nibble)                                  # 增
    integer_dot = tl.sum(q[None, :] * k, axis=1)                                    # 增
    q_scale = tl.load(Q_SCALE + row)                                                 # 增
    k_scale = tl.load(                                                              # 增
        K_SCALE + row * stride_ksr + token_idx * stride_kss,                        # 增
        mask=valid_n,                                                               # 增
        other=0.0,                                                                  # 增
    )                                                                               # 增
    logits = integer_dot * RECONSTRUCTION * q_scale * k_scale                       # 增
    logits *= INV_SQRT_HEAD_DIM                                                     # 增
    # padding 设为 -inf，后续 softmax 概率严格为 0。                               # 增
    logits = tl.where(valid_n, logits, -float("inf"))                               # 增
    tl.store(                                                                       # 增
        OUT + row * stride_or + offs_n * stride_oc,                                 # 增
        logits,                                                                     # 增
        mask=offs_n < CANDIDATE_CAPACITY,                                           # 增
    )                                                                               # 增


# 第二阶段只按 Bpre 的原始 token indices读取完整 INT8 K。                          # 增
@triton.jit                                                                         # 增
def _stp_refine_qk_kernel(                                                          # 增
    Q_INT8,                                                                         # 增
    K_INT8,                                                                         # 增
    Q_SCALE,                                                                        # 增
    K_SCALE,                                                                        # 增
    PRIOR_INDICES,                                                                  # 增
    PRIOR_COUNTS,                                                                   # 增
    OUT,                                                                            # 增
    stride_qr: tl.constexpr,                                                        # 增
    stride_qd: tl.constexpr,                                                        # 增
    stride_kr: tl.constexpr,                                                        # 增
    stride_ks: tl.constexpr,                                                        # 增
    stride_kd: tl.constexpr,                                                        # 增
    stride_ksr: tl.constexpr,                                                       # 增
    stride_kss: tl.constexpr,                                                       # 增
    stride_ir: tl.constexpr,                                                        # 增
    stride_ic: tl.constexpr,                                                        # 增
    stride_or: tl.constexpr,                                                        # 增
    stride_oc: tl.constexpr,                                                        # 增
    PRIOR_CAPACITY: tl.constexpr,                                                   # 增
    HEAD_DIM: tl.constexpr,                                                         # 增
    INV_SQRT_HEAD_DIM: tl.constexpr,                                                # 增
    BLOCK_N: tl.constexpr,                                                          # 增
    BLOCK_D: tl.constexpr,                                                          # 增
):                                                                                 # 增
    row = tl.program_id(0)                                                          # 增
    candidate_block = tl.program_id(1)                                              # 增
    offs_n = candidate_block * BLOCK_N + tl.arange(0, BLOCK_N)                      # 增
    offs_d = tl.arange(0, BLOCK_D)                                                   # 增
    count = tl.load(PRIOR_COUNTS + row)                                              # 增
    valid_n = (offs_n < PRIOR_CAPACITY) & (offs_n < count)                          # 增
    token_idx = tl.load(                                                            # 增
        PRIOR_INDICES + row * stride_ir + offs_n * stride_ic,                       # 增
        mask=valid_n,                                                               # 增
        other=0,                                                                    # 增
    )                                                                               # 增
    q = tl.load(                                                                    # 增
        Q_INT8 + row * stride_qr + offs_d * stride_qd,                              # 增
        mask=offs_d < HEAD_DIM,                                                     # 增
        other=0,                                                                    # 增
    ).to(tl.int32)                                                                  # 增
    k = tl.load(                                                                    # 增
        K_INT8                                                                      # 增
        + row * stride_kr                                                           # 增
        + token_idx[:, None] * stride_ks                                            # 增
        + offs_d[None, :] * stride_kd,                                              # 增
        mask=valid_n[:, None] & (offs_d[None, :] < HEAD_DIM),                       # 增
        other=0,                                                                    # 增
    ).to(tl.int32)                                                                  # 增
    integer_dot = tl.sum(q[None, :] * k, axis=1)                                    # 增
    q_scale = tl.load(Q_SCALE + row)                                                 # 增
    k_scale = tl.load(                                                              # 增
        K_SCALE + row * stride_ksr + token_idx * stride_kss,                        # 增
        mask=valid_n,                                                               # 增
        other=0.0,                                                                  # 增
    )                                                                               # 增
    logits = integer_dot * q_scale * k_scale * INV_SQRT_HEAD_DIM                    # 增
    logits = tl.where(valid_n, logits, -float("inf"))                               # 增
    tl.store(                                                                       # 增
        OUT + row * stride_or + offs_n * stride_oc,                                 # 增
        logits,                                                                     # 增
        mask=offs_n < PRIOR_CAPACITY,                                               # 增
    )                                                                               # 增


# 校验所有 STP kernel共用的 row-major metadata。                                   # 增
def _validate_metadata(                                                             # 增
    indices: torch.Tensor,                                                          # 增
    counts: torch.Tensor,                                                           # 增
    rows: int,                                                                      # 增
    device: torch.device,                                                           # 增
) -> None:                                                                         # 增
    if indices.dtype != torch.int32 or counts.dtype != torch.int32:                  # 增
        raise TypeError("STP indices/counts must be int32")                         # 增
    if indices.ndim != 2 or counts.shape != (rows,):                                 # 增
        raise ValueError("invalid STP indices/counts shape")                        # 增
    if not indices.is_contiguous() or not counts.is_contiguous():                    # 增
        raise ValueError("STP indices/counts must be contiguous")                   # 增
    if indices.device != device or counts.device != device:                         # 增
        raise ValueError("STP metadata must be on the same device as Q/K")          # 增


# 校验 Q/K/scale 都位于同一 CUDA device且满足 row-major prototype布局。             # 增
def _validate_tensor_layout(                                                        # 增
    reference: torch.Tensor,                                                        # 增
    *tensors: torch.Tensor,                                                         # 增
) -> None:                                                                         # 增
    if not reference.is_cuda:                                                       # 增
        raise ValueError("STP Triton kernels require CUDA tensors")                # 增
    if not reference.is_contiguous():                                                # 增
        raise ValueError("STP Triton inputs must be contiguous")                   # 增
    for tensor in tensors:                                                          # 增
        if tensor.device != reference.device:                                        # 增
            raise ValueError("all STP tensors must use the same CUDA device")       # 增
        if not tensor.is_contiguous():                                                # 增
            raise ValueError("STP Triton inputs must be contiguous")               # 增


# 启动 packed-MSB 先验 kernel。                                                     # 增
def stp_msb_scores(                                                                 # 增
    query_msb: torch.Tensor,                                                        # 增
    key_msb_packed: torch.Tensor,                                                   # 增
    query_scale: torch.Tensor,                                                      # 增
    key_scale: torch.Tensor,                                                        # 增
    input_indices: torch.Tensor,                                                    # 增
    input_counts: torch.Tensor,                                                     # 增
) -> torch.Tensor:                                                                 # 增
    if query_msb.dtype != torch.int8 or key_msb_packed.dtype != torch.uint8:         # 增
        raise TypeError("expected int8 Q-MSB and packed uint8 K-MSB")               # 增
    if query_msb.ndim != 2 or key_msb_packed.ndim != 3:                              # 增
        raise ValueError("expected Q[R,D] and packed K[R,S,D/2]")                  # 增
    rows, head_dim = query_msb.shape                                                 # 增
    if key_msb_packed.shape[0] != rows or key_msb_packed.shape[2] * 2 != head_dim:    # 增
        raise ValueError("packed K shape does not match Q")                        # 增
    seq_len = key_msb_packed.shape[1]                                                # 增
    if query_scale.shape != (rows,) or key_scale.shape != (rows, seq_len):           # 增
        raise ValueError("expected Q scale [R] and K scale [R,S]")                 # 增
    if query_scale.dtype != torch.float32 or key_scale.dtype != torch.float32:        # 增
        raise TypeError("STP scales must be float32")                              # 增
    _validate_tensor_layout(                                                        # 增
        query_msb, key_msb_packed, query_scale, key_scale, input_indices, input_counts # 增
    )                                                                               # 增
    _validate_metadata(input_indices, input_counts, rows, query_msb.device)           # 增
    output = torch.empty(                                                           # 增
        input_indices.shape,                                                        # 增
        dtype=torch.float32,                                                        # 增
        device=query_msb.device,                                                    # 增
    )                                                                               # 增
    block_n = 32                                                                    # 增
    block_d = triton.next_power_of_2(head_dim)                                       # 增
    grid = (rows, triton.cdiv(input_indices.shape[1], block_n))                      # 增
    _stp_msb_qk_kernel[grid](                                                        # 增
        query_msb,                                                                  # 增
        key_msb_packed,                                                             # 增
        query_scale,                                                                # 增
        key_scale,                                                                  # 增
        input_indices,                                                              # 增
        input_counts,                                                               # 增
        output,                                                                     # 增
        query_msb.stride(0),                                                        # 增
        query_msb.stride(1),                                                        # 增
        key_msb_packed.stride(0),                                                   # 增
        key_msb_packed.stride(1),                                                   # 增
        key_msb_packed.stride(2),                                                   # 增
        key_scale.stride(0),                                                        # 增
        key_scale.stride(1),                                                        # 增
        input_indices.stride(0),                                                    # 增
        input_indices.stride(1),                                                    # 增
        output.stride(0),                                                           # 增
        output.stride(1),                                                           # 增
        CANDIDATE_CAPACITY=input_indices.shape[1],                                  # 增
        HEAD_DIM=head_dim,                                                          # 增
        RECONSTRUCTION=float(1 << 8),                                               # 增
        INV_SQRT_HEAD_DIM=1.0 / math.sqrt(head_dim),                                # 增
        BLOCK_N=block_n,                                                            # 增
        BLOCK_D=block_d,                                                            # 增
    )                                                                               # 增
    return output                                                                   # 增


# 启动只读取 Bpre token 的 Full-INT8 refinement kernel。                           # 增
def stp_refine_scores(                                                              # 增
    query_int8: torch.Tensor,                                                       # 增
    key_int8: torch.Tensor,                                                         # 增
    query_scale: torch.Tensor,                                                      # 增
    key_scale: torch.Tensor,                                                        # 增
    prior_indices: torch.Tensor,                                                    # 增
    prior_counts: torch.Tensor,                                                     # 增
) -> torch.Tensor:                                                                 # 增
    if query_int8.dtype != torch.int8 or key_int8.dtype != torch.int8:               # 增
        raise TypeError("expected int8 Q/K")                                       # 增
    if query_int8.ndim != 2 or key_int8.ndim != 3:                                   # 增
        raise ValueError("expected Q[R,D] and K[R,S,D]")                           # 增
    rows, head_dim = query_int8.shape                                                # 增
    if key_int8.shape[0] != rows or key_int8.shape[2] != head_dim:                   # 增
        raise ValueError("K shape does not match Q")                               # 增
    seq_len = key_int8.shape[1]                                                      # 增
    if query_scale.shape != (rows,) or key_scale.shape != (rows, seq_len):           # 增
        raise ValueError("expected Q scale [R] and K scale [R,S]")                 # 增
    if query_scale.dtype != torch.float32 or key_scale.dtype != torch.float32:        # 增
        raise TypeError("STP scales must be float32")                              # 增
    _validate_tensor_layout(                                                        # 增
        query_int8, key_int8, query_scale, key_scale, prior_indices, prior_counts    # 增
    )                                                                               # 增
    _validate_metadata(prior_indices, prior_counts, rows, query_int8.device)          # 增
    output = torch.empty(                                                           # 增
        prior_indices.shape,                                                        # 增
        dtype=torch.float32,                                                        # 增
        device=query_int8.device,                                                   # 增
    )                                                                               # 增
    block_n = 32                                                                    # 增
    block_d = triton.next_power_of_2(head_dim)                                       # 增
    grid = (rows, triton.cdiv(prior_indices.shape[1], block_n))                      # 增
    _stp_refine_qk_kernel[grid](                                                     # 增
        query_int8,                                                                 # 增
        key_int8,                                                                   # 增
        query_scale,                                                                # 增
        key_scale,                                                                  # 增
        prior_indices,                                                              # 增
        prior_counts,                                                               # 增
        output,                                                                     # 增
        query_int8.stride(0),                                                       # 增
        query_int8.stride(1),                                                       # 增
        key_int8.stride(0),                                                         # 增
        key_int8.stride(1),                                                         # 增
        key_int8.stride(2),                                                         # 增
        key_scale.stride(0),                                                        # 增
        key_scale.stride(1),                                                        # 增
        prior_indices.stride(0),                                                    # 增
        prior_indices.stride(1),                                                    # 增
        output.stride(0),                                                           # 增
        output.stride(1),                                                           # 增
        PRIOR_CAPACITY=prior_indices.shape[1],                                      # 增
        HEAD_DIM=head_dim,                                                          # 增
        INV_SQRT_HEAD_DIM=1.0 / math.sqrt(head_dim),                                # 增
        BLOCK_N=block_n,                                                            # 增
        BLOCK_D=block_d,                                                            # 增
    )                                                                               # 增
    return output                                                                   # 增
```

实现注意：

- 当前 P1 packed kernel **只承诺 `msb_bits=4`**，`RECONSTRUCTION=1<<8` 与此对应。3-bit 虽可存入 signed-int4 container，但仍要显式传 `1 << 10`；5-bit 已超出 signed-int4 范围，必须设计新的 packing/读取格式，不能只改重构系数。
- 这两个 kernel实现了“少读 K-MSB”和“只精算 Bpre”，但 Top-p compact、候选容量规划和 attention gather仍是独立阶段；P2 再考虑 fusion。
- Triton 是否真正生成理想的低位整数指令必须用 PTX/SASS 和 profiler确认，不能仅凭 dtype 名称宣称获得 4-bit 算力。

# `twilight/kernel/triton/__init__.py`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 导出列表 | 只导出 channel/bgemv/qk。 | 导出 STP 两个 score wrapper。 | benchmark和后续模型层使用统一入口。 | `from twilight.kernel import stp_msb_scores` 成功。 |

```python
"""Triton Kernels of Twilight."""

from .channel import get_label_tensor
from .bgemv_int8 import bgemv_int8
from .qk_int8_per_block import qk_int8_per_block
# 导出 STP 的 MSB 先验和候选 refinement kernels。                                 # 增
from .stp import stp_msb_scores, stp_refine_scores                                  # 增
```

`twilight/kernel/__init__.py` 已有 `from .triton import *`，因此 P1 不需要再修改该文件。

# `tests/test_stp.py`（新增）

| 测试 | 覆盖目标 | 通过标准 |
|---|---|---|
| INT8 quant/dequant | scale、范围、误差。 | int8 范围正确；误差不超过约一个 scale。 |
| MSB slice | 负数算术右移。 | 与 Python 整数 `>>` reference一致。 |
| signed-int4 pack | nibble 顺序和符号扩展。 | round trip完全相等。 |
| Top-p non-sink max | `sampling.cuh` 上界修复。 | 最大 token不在 0 时 mask/indices仍正确。 |
| STP invariants | B0/Bpre/B1 包含关系。 | `B1 ⊆ Bpre ⊆ B0`，每行非空。 |
| Triton scores | packed MSB和候选 refinement。 | 与 P0 dense reference在容差内。 |

```python
# 新增：STP 的 CPU 量化测试和 CUDA 端到端/kernel测试。                            # 增
import math                                                                         # 增

import pytest                                                                       # 增
import torch                                                                        # 增

from twilight.kernel import (                                                       # 增
    stp_msb_scores,                                                                 # 增
    stp_refine_scores,                                                              # 增
    top_p_fp32_return_indices,                                                      # 增
    top_p_fp32_return_mask,                                                         # 增
)                                                                                   # 增
from twilight.pyimpl.quantize import (                                               # 增
    pack_signed_int4,                                                               # 增
    signed_msb_slice_int8,                                                          # 增
    symmetric_per_token_quant_int8,                                                 # 增
    unpack_signed_int4,                                                             # 增
)                                                                                   # 增
from twilight.pyimpl.stp import stp_pruner                                           # 增


# 验证 INT8 范围、scale shape和反量化误差。                                        # 增
def test_symmetric_per_token_quant_int8() -> None:                                   # 增
    tensor = torch.tensor(                                                          # 增
        [[[-2.0, -0.5, 0.25, 1.5], [0.0, 0.0, 0.0, 0.0]]],                         # 增
        dtype=torch.float32,                                                        # 增
    )                                                                               # 增
    quantized, scale = symmetric_per_token_quant_int8(tensor)                       # 增
    assert quantized.dtype == torch.int8                                             # 增
    assert scale.dtype == torch.float32                                              # 增
    assert scale.shape == tensor.shape[:-1] + (1,)                                  # 增
    assert int(quantized.min()) >= -127                                              # 增
    assert int(quantized.max()) <= 127                                               # 增
    reconstructed = quantized.float() * scale                                        # 增
    assert torch.all(torch.abs(reconstructed - tensor) <= scale + 1e-6)              # 增


# 验证负 INT8 的高位采用算术右移。                                                  # 增
def test_signed_msb_slice_int8() -> None:                                            # 增
    values = torch.tensor([-127, -33, -16, -1, 0, 15, 16, 127], dtype=torch.int8)   # 增
    actual = signed_msb_slice_int8(values, msb_bits=4)                               # 增
    expected = torch.tensor(                                                        # 增
        [int(value) >> 4 for value in values.tolist()],                             # 增
        dtype=torch.int8,                                                           # 增
    )                                                                               # 增
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)                     # 增


# 验证 signed-int4 两两打包后的 nibble 顺序和符号。                                # 增
def test_signed_int4_pack_round_trip() -> None:                                      # 增
    values = torch.tensor(                                                          # 增
        [[-8, -7, -1, 0, 1, 6, 7, -3]],                                            # 增
        dtype=torch.int8,                                                           # 增
    )                                                                               # 增
    packed = pack_signed_int4(values)                                                # 增
    assert packed.dtype == torch.uint8                                               # 增
    torch.testing.assert_close(                                                      # 增
        unpack_signed_int4(packed), values, rtol=0, atol=0                           # 增
    )                                                                               # 增


# CUDA 测试在无 GPU 环境中明确跳过。                                                # 增
cuda_only = pytest.mark.skipif(                                                     # 增
    not torch.cuda.is_available(),                                                   # 增
    reason="STP Top-p/Triton tests require CUDA",                                  # 增
)                                                                                   # 增


# 验证 Top-p 不再依赖 token 0 是最大概率。                                          # 增
@cuda_only                                                                           # 增
def test_top_p_when_maximum_is_not_attention_sink() -> None:                         # 增
    probs = torch.tensor(                                                           # 增
        [[[[0.10, 0.60, 0.20, 0.10]]]],                                            # 增
        dtype=torch.float32,                                                        # 增
        device="cuda",                                                            # 增
    )                                                                               # 增
    mask = top_p_fp32_return_mask(probs, 0.70)                                       # 增
    expected = torch.tensor(                                                        # 增
        [[[[False, True, True, False]]]],                                            # 增
        device="cuda",                                                            # 增
    )                                                                               # 增
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)                       # 增
    indices, counts = top_p_fp32_return_indices(probs, 0.70)                         # 增
    assert counts.item() == 2                                                        # 增
    assert set(indices.flatten()[:2].tolist()) == {1, 2}                             # 增


# 验证 STP 的三个候选集合包含关系和最小预算。                                      # 增
@cuda_only                                                                           # 增
def test_stp_mask_invariants() -> None:                                              # 增
    torch.manual_seed(7)                                                            # 增
    query = torch.randn(2, 4, 1, 32, dtype=torch.float16, device="cuda")            # 增
    key = torch.randn(2, 4, 128, 32, dtype=torch.float16, device="cuda")            # 增
    selector_mask = torch.rand(2, 4, 1, 128, device="cuda") > 0.20                  # 增
    result = stp_pruner(                                                            # 增
        query,                                                                      # 增
        key,                                                                        # 增
        selector_mask,                                                              # 增
        prior_threshold=0.95,                                                       # 增
        threshold=0.85,                                                             # 增
        quant_bits=8,                                                               # 增
        msb_bits=4,                                                                 # 增
        min_tokens=1,                                                               # 增
    )                                                                               # 增
    assert torch.all(~result.prior_mask | selector_mask)                             # 增
    assert torch.all(~result.final_mask | result.prior_mask)                         # 增
    assert torch.all(result.prior_mask.sum(dim=-1) >= 1)                             # 增
    assert torch.all(result.final_mask.sum(dim=-1) >= 1)                             # 增


# 验证 packed-MSB 和候选 Full-INT8 kernels 对齐 dense reference。                  # 增
@cuda_only                                                                           # 增
def test_stp_triton_scores_match_reference() -> None:                                # 增
    torch.manual_seed(11)                                                           # 增
    batch, heads, seq_len, head_dim = 2, 4, 96, 32                                  # 增
    query = torch.randn(batch, heads, 1, head_dim, device="cuda")                   # 增
    key = torch.randn(batch, heads, seq_len, head_dim, device="cuda")               # 增
    query_int8, query_scale = symmetric_per_token_quant_int8(query)                  # 增
    key_int8, key_scale = symmetric_per_token_quant_int8(key)                        # 增
    query_msb = signed_msb_slice_int8(query_int8, 4)                                 # 增
    key_msb = signed_msb_slice_int8(key_int8, 4)                                     # 增
    key_msb_packed = pack_signed_int4(key_msb)                                       # 增
    rows = batch * heads                                                            # 增
    q8 = query_int8.reshape(rows, head_dim).contiguous()                             # 增
    k8 = key_int8.reshape(rows, seq_len, head_dim).contiguous()                      # 增
    q4 = query_msb.reshape(rows, head_dim).contiguous()                              # 增
    k4 = key_msb_packed.reshape(rows, seq_len, head_dim // 2).contiguous()           # 增
    qs = query_scale.reshape(rows).contiguous()                                      # 增
    ks = key_scale.reshape(rows, seq_len).contiguous()                               # 增
    indices = torch.arange(seq_len, dtype=torch.int32, device="cuda").repeat(rows, 1) # 增
    counts = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")         # 增

    actual_msb = stp_msb_scores(q4, k4, qs, ks, indices, counts)                     # 增
    # 直接 reshape 已有 K-MSB，避免 reference 发生额外 copy。                      # 增
    k4_unpacked = key_msb.reshape(rows, seq_len, head_dim).contiguous()              # 增
    expected_msb = torch.matmul(                                                     # 增
        q4.float().unsqueeze(1), k4_unpacked.float().transpose(1, 2)                 # 增
    ).squeeze(1)                                                                    # 增
    expected_msb *= 256.0 * qs[:, None] * ks / math.sqrt(head_dim)                   # 增
    torch.testing.assert_close(actual_msb, expected_msb, rtol=1e-4, atol=1e-3)       # 增

    # 用不规则 Bpre 验证 Full-INT8 只读候选。                                       # 增
    prior_indices = indices[:, ::3].contiguous()                                    # 增
    prior_counts = torch.full(                                                      # 增
        (rows,), prior_indices.shape[1], dtype=torch.int32, device="cuda"           # 增
    )                                                                               # 增
    actual_full = stp_refine_scores(q8, k8, qs, ks, prior_indices, prior_counts)      # 增
    dense_full = torch.matmul(q8.float().unsqueeze(1), k8.float().transpose(1, 2))    # 增
    dense_full = dense_full.squeeze(1) * qs[:, None] * ks / math.sqrt(head_dim)      # 增
    expected_full = dense_full.gather(dim=-1, index=prior_indices.long())             # 增
    torch.testing.assert_close(actual_full, expected_full, rtol=1e-4, atol=1e-3)     # 增
```

# `requirements.txt`

| 修改位置 | 当前行为 | 修改意见 | 原因 | 验收 |
|---|---|---|---|---|
| 文件末尾 | 未声明测试 runner。 | 增加固定版本 `pytest`。 | 新增 `tests/test_stp.py` 需要可复现的 runner。 | 新环境可直接运行 STP tests。 |

```text
# 新增 STP 单元测试 runner。                                                        # 增
pytest==8.3.3                                                                       # 增
```

# `benchmark/efficiency/bench_stp.py`（新增）

| 输出指标 | 定义 | 目的 |
|---|---|---|
| `B0/Bpre/B1` | selector输入、MSB先验、最终 STP 的平均逐 head token数。 | 判断实际压缩链。 |
| `prior_recall` | `|Bpre ∩ Full8TopP| / |Full8TopP|`。 | 判断 MSB 先验是否漏掉 Full-INT8关键 token。 |
| `final_recall` | `|B1_STP ∩ B1_Full8| / |B1_Full8|`。 | 直接衡量 STP 对基线选择集合的影响。 |
| `retained_mass` | STP B1 在 Full-INT8 softmax分布上保留的概率质量。 | 比 token recall更贴近 attention误差。 |
| 分阶段延迟 | MSB QK、prior Top-p、Full8 refine、final Top-p。 | 定位瓶颈，禁止只报总和。 |
| baseline延迟 | B0 上 Full-INT8 QK + Top-p。 | 证明 STP 的真实加速对象。 |

核心 benchmark 代码如下；CLI/打印必须保留这些实验变量：

```python
# 新增：STP 与 Full-INT8 Top-p 的 correctness/latency benchmark。                  # 增
import argparse                                                                     # 增
import math                                                                         # 增
import sys                                                                          # 增
from pathlib import Path                                                            # 增

import torch                                                                        # 增

REPO_ROOT = Path(__file__).resolve().parents[2]                                      # 增
sys.path.insert(0, str(REPO_ROOT))                                                   # 增

from bench_utils import benchmark_forward_with_warmup                               # 增
from twilight.kernel import stp_msb_scores, stp_refine_scores                       # 增
from twilight.pyimpl.quantize import (                                               # 增
    pack_signed_int4,                                                               # 增
    signed_msb_slice_int8,                                                          # 增
    symmetric_per_token_quant_int8,                                                 # 增
)                                                                                   # 增
from twilight.pyimpl.top_p import top_p_unnormalized_return_indices                 # 增


# 将 compact indices/counts恢复为原序列 bool mask。                                 # 增
def indices_to_mask(                                                                # 增
    indices: torch.Tensor,                                                          # 增
    counts: torch.Tensor,                                                           # 增
    seq_len: int,                                                                   # 增
) -> torch.Tensor:                                                                 # 增
    capacity = indices.shape[-1]                                                     # 增
    positions = torch.arange(capacity, device=indices.device)                       # 增
    valid = positions.view(*([1] * (indices.ndim - 1)), capacity) < counts.unsqueeze(-1) # 增
    safe_indices = torch.where(valid, indices, torch.zeros_like(indices))            # 增
    # scatter_add 避免无效槽位都映射到 token 0 时覆盖有效的 token-0 True。          # 增
    hits = torch.zeros(                                                             # 增
        (*indices.shape[:-1], seq_len),                                              # 增
        dtype=torch.int32,                                                          # 增
        device=indices.device,                                                      # 增
    )                                                                               # 增
    hits.scatter_add_(dim=-1, index=safe_indices.long(), src=valid.to(torch.int32))  # 增
    return hits > 0                                                                  # 增


# 计算 Full-INT8 dense baseline logits。                                            # 增
def full_int8_logits(                                                               # 增
    query_int8: torch.Tensor,                                                       # 增
    key_int8: torch.Tensor,                                                         # 增
    query_scale: torch.Tensor,                                                      # 增
    key_scale: torch.Tensor,                                                        # 增
) -> torch.Tensor:                                                                 # 增
    head_dim = query_int8.shape[-1]                                                  # 增
    logits = torch.matmul(                                                          # 增
        query_int8.float().unsqueeze(1),                                             # 增
        key_int8.float().transpose(1, 2),                                            # 增
    ).squeeze(1)                                                                    # 增
    return logits * query_scale[:, None] * key_scale / math.sqrt(head_dim)           # 增


# 对一个 shape配置完成准备、正确性统计和分阶段计时。                               # 增
@torch.inference_mode()                                                              # 增
def bench_stp(                                                                      # 增
    batch_size: int,                                                                # 增
    num_heads: int,                                                                 # 增
    seq_len: int,                                                                   # 增
    head_dim: int,                                                                  # 增
    prior_p: float,                                                                 # 增
    final_p: float,                                                                 # 增
    warmups: int,                                                                   # 增
    repeats: int,                                                                   # 增
) -> None:                                                                         # 增
    torch.manual_seed(42)                                                           # 增
    device = "cuda"                                                               # 增
    query = torch.randn(                                                            # 增
        batch_size, num_heads, 1, head_dim, dtype=torch.float16, device=device       # 增
    )                                                                               # 增
    key = torch.randn(                                                              # 增
        batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device # 增
    )                                                                               # 增
    query_int8, query_scale = symmetric_per_token_quant_int8(query)                  # 增
    key_int8, key_scale = symmetric_per_token_quant_int8(key)                        # 增
    query_msb = signed_msb_slice_int8(query_int8, 4)                                 # 增
    key_msb_packed = pack_signed_int4(signed_msb_slice_int8(key_int8, 4))            # 增
    rows = batch_size * num_heads                                                    # 增
    q8 = query_int8.reshape(rows, head_dim).contiguous()                             # 增
    k8 = key_int8.reshape(rows, seq_len, head_dim).contiguous()                      # 增
    q4 = query_msb.reshape(rows, head_dim).contiguous()                              # 增
    k4 = key_msb_packed.reshape(rows, seq_len, head_dim // 2).contiguous()           # 增
    qs = query_scale.reshape(rows).contiguous()                                      # 增
    ks = key_scale.reshape(rows, seq_len).contiguous()                               # 增
    # selector=none 的隔离基准令 B0 等于完整序列。                                  # 增
    b0_indices = torch.arange(                                                       # 增
        seq_len, dtype=torch.int32, device=device                                    # 增
    ).repeat(rows, 1)                                                               # 增
    b0_counts = torch.full(                                                         # 增
        (rows,), seq_len, dtype=torch.int32, device=device                           # 增
    )                                                                               # 增

    # Full-INT8 Top-p baseline使用同一个候选 kernel扫完整 B0，保证计时公平。         # 增
    baseline_logits = stp_refine_scores(q8, k8, qs, ks, b0_indices, b0_counts)       # 增
    # PyTorch dense版本只做数值 reference，不作为性能分母。                         # 增
    baseline_reference = full_int8_logits(q8, k8, qs, ks)                            # 增
    torch.testing.assert_close(                                                      # 增
        baseline_logits, baseline_reference, rtol=1e-4, atol=1e-3                   # 增
    )                                                                               # 增
    baseline_indices, baseline_counts = top_p_unnormalized_return_indices(           # 增
        baseline_logits.view(batch_size, num_heads, 1, seq_len),                    # 增
        final_p,                                                                    # 增
        input_indices=b0_indices.view(batch_size, num_heads, 1, seq_len),            # 增
    )                                                                               # 增

    # STP stage 1：packed-MSB scores和 Bpre。                                       # 增
    msb_logits = stp_msb_scores(q4, k4, qs, ks, b0_indices, b0_counts)               # 增
    prior_indices, prior_counts = top_p_unnormalized_return_indices(                 # 增
        msb_logits.view(batch_size, num_heads, 1, seq_len),                         # 增
        prior_p,                                                                    # 增
        input_indices=b0_indices.view(batch_size, num_heads, 1, seq_len),            # 增
    )                                                                               # 增
    # setup阶段允许一次 host sync取得真实容量；该同步不能计入 steady-state kernel。 # 增
    prior_capacity = max(1, int(prior_counts.max().item()))                          # 增
    prior_indices = prior_indices[..., :prior_capacity].contiguous()                 # 增
    prior_indices_2d = prior_indices.reshape(rows, prior_capacity)                   # 增
    prior_counts_1d = prior_counts.reshape(rows).contiguous()                        # 增

    # STP stage 2：只对 Bpre 做完整 INT8×INT8。                                    # 增
    refined_logits = stp_refine_scores(                                              # 增
        q8, k8, qs, ks, prior_indices_2d, prior_counts_1d                            # 增
    )                                                                               # 增
    final_indices, final_counts = top_p_unnormalized_return_indices(                 # 增
        refined_logits.view(batch_size, num_heads, 1, prior_capacity),              # 增
        final_p,                                                                    # 增
        input_indices=prior_indices,                                                # 增
    )                                                                               # 增

    # 还原 mask并计算选择质量。                                                     # 增
    baseline_mask = indices_to_mask(baseline_indices, baseline_counts, seq_len)      # 增
    prior_mask = indices_to_mask(prior_indices, prior_counts, seq_len)               # 增
    final_mask = indices_to_mask(final_indices, final_counts, seq_len)               # 增
    baseline_size = baseline_mask.sum(dim=-1).clamp_min(1)                           # 增
    prior_recall = ((prior_mask & baseline_mask).sum(dim=-1) / baseline_size).float().mean() # 增
    final_recall = ((final_mask & baseline_mask).sum(dim=-1) / baseline_size).float().mean() # 增
    baseline_probs = torch.softmax(                                                  # 增
        baseline_logits.view(batch_size, num_heads, 1, seq_len),                    # 增
        dim=-1,                                                                     # 增
        dtype=torch.float32,                                                        # 增
    )                                                                               # 增
    retained_mass = (baseline_probs * final_mask).sum(dim=-1).mean()                 # 增

    # 分别计时，禁止把 setup和 host sync混入 kernel latency。                       # 增
    _, msb_t = benchmark_forward_with_warmup(                                        # 增
        stp_msb_scores, q4, k4, qs, ks, b0_indices, b0_counts,                      # 增
        warmups=warmups, repeats=repeats                                             # 增
    )                                                                               # 增
    _, prior_t = benchmark_forward_with_warmup(                                      # 增
        top_p_unnormalized_return_indices,                                           # 增
        msb_logits.view(batch_size, num_heads, 1, seq_len), prior_p,                 # 增
        b0_indices.view(batch_size, num_heads, 1, seq_len), None,                    # 增
        warmups=warmups, repeats=repeats                                             # 增
    )                                                                               # 增
    _, refine_t = benchmark_forward_with_warmup(                                     # 增
        stp_refine_scores, q8, k8, qs, ks, prior_indices_2d, prior_counts_1d,        # 增
        warmups=warmups, repeats=repeats                                             # 增
    )                                                                               # 增
    _, final_t = benchmark_forward_with_warmup(                                      # 增
        top_p_unnormalized_return_indices,                                           # 增
        refined_logits.view(batch_size, num_heads, 1, prior_capacity), final_p,      # 增
        prior_indices, None, warmups=warmups, repeats=repeats                        # 增
    )                                                                               # 增
    _, baseline_t = benchmark_forward_with_warmup(                                   # 增
        stp_refine_scores, q8, k8, qs, ks, b0_indices, b0_counts,                   # 增
        warmups=warmups, repeats=repeats                                             # 增
    )                                                                               # 增
    # Full-INT8 baseline必须同时计入自己的最终 Top-p。                              # 增
    _, baseline_top_p_t = benchmark_forward_with_warmup(                             # 增
        top_p_unnormalized_return_indices,                                           # 增
        baseline_logits.view(batch_size, num_heads, 1, seq_len), final_p,            # 增
        b0_indices.view(batch_size, num_heads, 1, seq_len), None,                    # 增
        warmups=warmups, repeats=repeats                                             # 增
    )                                                                               # 增

    print(                                                                          # 增
        f"B0={seq_len:.1f}, Bpre={prior_counts.float().mean().item():.2f}, "         # 增
        f"B1={final_counts.float().mean().item():.2f}, "                            # 增
        f"prior_recall={prior_recall.item():.6f}, "                                 # 增
        f"final_recall={final_recall.item():.6f}, "                                 # 增
        f"retained_mass={retained_mass.item():.6f}"                                 # 增
    )                                                                               # 增
    stp_total_us = (msb_t.mean + prior_t.mean + refine_t.mean + final_t.mean) * 1e6  # 增
    baseline_total_us = (baseline_t.mean + baseline_top_p_t.mean) * 1e6              # 增
    print(                                                                          # 增
        f"msb={msb_t.mean * 1e6:.2f}us, prior_p={prior_t.mean * 1e6:.2f}us, "        # 增
        f"refine={refine_t.mean * 1e6:.2f}us, final_p={final_t.mean * 1e6:.2f}us, "  # 增
        f"stp_total={stp_total_us:.2f}us, full8_qk={baseline_t.mean * 1e6:.2f}us, "  # 增
        f"full8_top_p={baseline_top_p_t.mean * 1e6:.2f}us, "                        # 增
        f"full8_total={baseline_total_us:.2f}us, "                                  # 增
        f"speedup={baseline_total_us / stp_total_us:.3f}x"                          # 增
    )                                                                               # 增


# 暴露可复现实验参数。                                                              # 增
def main() -> None:                                                                 # 增
    parser = argparse.ArgumentParser(description="Benchmark STP against Full-INT8 Top-p") # 增
    parser.add_argument("--batch-size", type=int, default=8)                       # 增
    parser.add_argument("--num-heads", type=int, default=32)                       # 增
    parser.add_argument("--seq-len", type=int, default=8192)                       # 增
    parser.add_argument("--head-dim", type=int, default=128)                       # 增
    parser.add_argument("--prior-p", type=float, default=0.95)                     # 增
    parser.add_argument("--final-p", type=float, default=0.85)                     # 增
    parser.add_argument("--warmups", type=int, default=20)                         # 增
    parser.add_argument("--repeats", type=int, default=1000)                       # 增
    args = parser.parse_args()                                                       # 增
    bench_stp(                                                                       # 增
        args.batch_size, args.num_heads, args.seq_len, args.head_dim,                # 增
        args.prior_p, args.final_p, args.warmups, args.repeats                       # 增
    )                                                                               # 增


if __name__ == "__main__":                                                         # 增
    main()                                                                           # 增
```

# `benchmark/LongBench/analyze_stp.py`（新增）

| 修改内容 | 说明 | 验收 |
|---|---|---|
| 读取 JSONL | 收集 `stp_selector_budget`、`stp_prior_budget`、`stp_final_budget`。 | 自动跳过非 STP/null 记录。 |
| 聚合 | 每个文件和总计输出 B0/Bpre/B1、`Bpre/B0`、`B1/Bpre`。 | 不把不同阶段含义混在一个 `budget` 中。 |
| 输出 | stdout + 可选 JSON。 | 可直接用于表格/画图。 |

```python
# 新增：聚合 LongBench 输出中的 STP 三阶段预算。                                  # 增
import argparse                                                                     # 增
import json                                                                         # 增
from pathlib import Path                                                            # 增
from statistics import mean                                                        # 增


# 对一组样本计算 B0/Bpre/B1 及压缩比。                                             # 增
def summarize(records: list[dict]) -> dict:                                         # 增
    valid = [                                                                       # 增
        record                                                                      # 增
        for record in records                                                       # 增
        if record.get("stp_selector_budget") is not None                            # 增
        and record.get("stp_prior_budget") is not None                              # 增
        and record.get("stp_final_budget") is not None                              # 增
    ]                                                                               # 增
    if not valid:                                                                   # 增
        return {"num_samples": 0}                                                   # 增
    b0 = mean(record["stp_selector_budget"] for record in valid)                    # 增
    bpre = mean(record["stp_prior_budget"] for record in valid)                     # 增
    b1 = mean(record["stp_final_budget"] for record in valid)                       # 增
    return {                                                                         # 增
        "num_samples": len(valid),                                                  # 增
        "B0": b0,                                                                  # 增
        "Bpre": bpre,                                                              # 增
        "B1": b1,                                                                  # 增
        "Bpre_over_B0": None if b0 == 0 else bpre / b0,                            # 增
        "B1_over_Bpre": None if bpre == 0 else b1 / bpre,                          # 增
    }                                                                               # 增


# 读取目录内每个预测 JSONL并同时生成总体统计。                                     # 增
def analyze(input_dir: Path) -> dict:                                                # 增
    result = {}                                                                     # 增
    all_records = []                                                                # 增
    for path in sorted(input_dir.glob("*.jsonl")):                                 # 增
        records = []                                                                # 增
        with path.open("r", encoding="utf-8") as file:                            # 增
            for line in file:                                                       # 增
                records.append(json.loads(line))                                    # 增
        result[path.name] = summarize(records)                                       # 增
        all_records.extend(records)                                                  # 增
    result["__overall__"] = summarize(all_records)                                 # 增
    return result                                                                   # 增


# 提供独立 CLI，不改变 LongBench 官方分数计算。                                    # 增
def main() -> None:                                                                 # 增
    parser = argparse.ArgumentParser(description="Summarize STP LongBench budgets") # 增
    parser.add_argument("input_dir", type=Path)                                    # 增
    parser.add_argument("--output", type=Path, default=None)                       # 增
    args = parser.parse_args()                                                       # 增
    result = analyze(args.input_dir)                                                 # 增
    rendered = json.dumps(result, ensure_ascii=False, indent=2)                      # 增
    print(rendered)                                                                 # 增
    if args.output is not None:                                                      # 增
        args.output.write_text(rendered + "\n", encoding="utf-8")                 # 增


if __name__ == "__main__":                                                         # 增
    main()                                                                           # 增
```

# `README.md`

| 修改位置 | 修改意见 | 原因 | 验收 |
|---|---|---|---|
| Accuracy Evaluation 后 | 增加 STP 算法定义、配置和 LongBench命令。 | 防止把 P0 dense reference当成性能实现。 | 新用户能复现实验。 |
| Efficiency Evaluation 后 | 增加 `bench_stp.py` 命令和指标解释。 | 对比必须同时包含 Full-INT8 Top-p总延迟。 | README 输出字段与脚本一致。 |

在 Accuracy Evaluation 章节加入：

````markdown
#### STP (MSB-slice Top-p)                                                        <!-- # 增 -->
                                                                                   <!-- # 增 -->
STP first evaluates the candidates selected by the base selector with the          <!-- # 增 -->
signed MSB slices of per-token INT8 Q/K (4 high bits by default). It then runs       <!-- # 增 -->
the full INT8-by-INT8 score only on the prior Top-p candidates and applies the       <!-- # 增 -->
final Top-p threshold. The Python implementation is a correctness reference and     <!-- # 增 -->
still computes dense full-INT8 logits; use `bench_stp.py` for the candidate-only     <!-- # 增 -->
Triton prototype.                                                                   <!-- # 增 -->
                                                                                   <!-- # 增 -->
```bash                                                                            <!-- # 增 -->
cd benchmark                                                                        <!-- # 增 -->
bash scripts/run_longbench.sh configs/config_quest_stp.json                         <!-- # 增 -->
python LongBench/analyze_stp.py results_longbench/Meta-Llama-3.1-8B-Instruct/pred   <!-- # 增 -->
```                                                                                 <!-- # 增 -->
````

在 Efficiency Evaluation 章节加入：

````markdown
#### Bench STP                                                                       <!-- # 增 -->
                                                                                     <!-- # 增 -->
```bash                                                                              <!-- # 增 -->
cd benchmark/efficiency                                                              <!-- # 增 -->
python bench_stp.py --batch-size 8 --num-heads 32 --seq-len 8192 \                  <!-- # 增 -->
    --head-dim 128 --prior-p 0.95 --final-p 0.85                                     <!-- # 增 -->
```                                                                                  <!-- # 增 -->
                                                                                     <!-- # 增 -->
Report B0/Bpre/B1, prior/final recall, retained Full-INT8 probability mass,          <!-- # 增 -->
each STP stage latency, Full-INT8 QK latency, Full-INT8 Top-p latency, and total      <!-- # 增 -->
speedup. Do not compare STP total latency against QK-only latency.                    <!-- # 增 -->
````

# `twilight/model/llama_patch.py`（P2 重构，不与 P0/P1 同批提交）

| 当前代码区域 | 必须修改成什么 | 原因 | 合入前置条件 |
|---|---|---|---|
| prefill 第 53-66 行 | 删除“直接返回 hidden_states”；执行真实 attention并把真实 K8、packed K-MSB、scale写 cache。 | 当前 prefill跳过 attention，无法用于正确模型。 | P0/P1 数值和 kernel已通过。 |
| Quest 第 108-118 行 | 让 `label_index` 真正成为 B0 indices/counts。 | 当前计算后丢弃。 | 定义 paged/GQA index布局。 |
| Twilight 第 122-133 行 | 删除固定 B0 indices与丢弃的 Top-p mask，替换为 `stp_msb_scores -> prior indices -> stp_refine_scores -> final indices`。 | 真正接通 STP。 | 支持 paged K cache的 kernel。 |
| attention 第 138-157 行 | page table由 STP final indices构建，不再是 `[0, B1)` 固定前缀。 | 最终 attention必须读取实际关键 token。 | FlashInfer per-head ragged布局确定。 |
| `enable_twilight` 第 190-241 行 | 删除硬编码 32×128 和所有随机 cache；按模型 config/device分配真实 cache。 | 当前只是合成 microbenchmark。 | cache manager完成。 |
| GQA | Q head映射到 `kv_head = q_head // num_key_value_groups`，K8/K-MSB不能按 Q head复制存储。 | 否则内存收益被复制抵消。 | P1 kernel增加 GQA stride/head映射。 |

当前 P2 首先应删除/替换的危险路径如下；这段是**重构边界**，不是建议先临时注释后继续跑模型：

```python
    # Prefill stage
    if q_len > 1:
        # 直接跳过 attention 会返回错误模型结果。                                  # 删
        return hidden_states, None, None                                             # 删
        # P2：调用真实 prefill并同步建立 K8、K-MSB、K-scale cache。                  # 增
        return self.stp_backend.prefill_forward(                                     # 增
            hidden_states=hidden_states,                                             # 增
            attention_mask=attention_mask,                                           # 增
            position_ids=position_ids,                                               # 增
            past_key_value=past_key_value,                                           # 增
            output_attentions=output_attentions,                                     # 增
            use_cache=use_cache,                                                     # 增
            **kwargs,                                                                # 增
        )                                                                            # 增

    if self.use_quest:
        tmp_scores = batched_sparse_gemv(...)
        # Quest 输出当前算出后没有进入后续 page table。                              # 删
        _, label_index = torch.topk(tmp_scores, self.B0 // 16, dim=-1)               # 删
        # P2：backend 返回可直接交给 STP 的 B0 原始 token indices/counts。           # 增
        b0_indices, b0_counts = self.stp_backend.quest_select(tmp_scores)             # 增

    if self.use_twilight:
        # 旧路径在固定 B0 上估分并丢弃 Top-p 结果。                                  # 删
        estimated_weights = batched_sparse_gemv_int8_k(                              # 删
            query_states, self.k_cache_int8, self.quant_scales, self.quant_zeros,    # 删
            *self.B0_indices, self.B0,                                                # 删
        )                                                                            # 删
        top_p_unnormalized(estimated_weights, 0.85)                                   # 删
        kv_seq_len = self.B1                                                         # 删
        # P2：STP backend必须返回真实原 token id，最终 page table直接使用它们。       # 增
        final_indices, final_counts = self.stp_backend.select(                       # 增
            query_states,                                                           # 增
            b0_indices=b0_indices,                                                   # 增
            b0_counts=b0_counts,                                                     # 增
            prior_threshold=self.stp_prior_threshold,                               # 增
            threshold=self.stp_threshold,                                           # 增
        )                                                                            # 增
```

`self.stp_backend` 不能只是空 wrapper；它的 contract 必须是：

```python
# P2 backend contract：prefill必须建立真实量化 cache。                              # 增
class STPAttentionBackend:                                                          # 增
    # 写入 K8、packed K-MSB、K-scale，并执行正确 dense/flash prefill。              # 增
    def prefill_forward(self, **kwargs): ...                                        # 增
    # 把 Quest chunk ids展开并裁剪为原始 token ids/counts。                         # 增
    def quest_select(self, chunk_scores): ...                                       # 增
    # 执行 packed-MSB prior、candidate Full-INT8和最终 compact Top-p。              # 增
    def select(self, query_states, b0_indices, b0_counts, prior_threshold, threshold): ... # 增
    # 用 final原 token ids构造 per-head ragged page table并运行 attention。          # 增
    def attend(self, query_states, final_indices, final_counts): ...                 # 增
```

在 backend、paged cache和 GQA contract 完成前，不应把 `llama_patch.py` 的结果用于准确率或端到端性能结论。

## 明确不需要修改的现有文件

| 文件 | 原因 |
|---|---|
| `benchmark/scripts/run_longbench.sh` | 已通过第一个参数接收任意算法 JSON，新 STP 配置可直接传入。 |
| `benchmark/LongBench/eval.py` | 新预算字段不会影响原有任务评分。 |
| `benchmark/LongBench/metrics.py` | STP 不改变预测指标定义。 |
| `twilight/pyimpl/top_p.py` | P0 继续复用统一 softmax + Top-p接口；底层最大值假设在 `sampling.cuh` 修正。 |
| `setup.py` | 新增文件都位于现有包目录，当前运行方式不要求新增编译 extension声明。 |

## 实验矩阵与合入门槛

| 维度 | 至少覆盖的值 |
|---|---|
| `msb_bits` | P0 扫描 3、4、5；当前 P1 只验收 4，扩展 packing 后再测 3/5。 |
| `prior_threshold` | 0.90、0.95、0.975、0.99。 |
| `final threshold` | 0.80、0.85、0.90。 |
| B0 selector | none、Quest-8192、DS-8192。 |
| sequence length | 4K、8K、16K、32K。 |
| batch | 1、4、8、16；区分 decode latency与吞吐。 |
| head layout | MHA 32/32；GQA 32/8（P2 前必须补）。 |

合入门槛：

1. CPU 量化/pack 单测和 CUDA Top-p/STP/Triton 单测全部通过；
2. P0 的 `B1 ⊆ Bpre ⊆ B0` 对所有测试输入成立；
3. P1 两个 score kernel与 Python reference对齐；
4. `prior_recall`、`final_recall`、`retained_mass` 必须与 LongBench/Passkey精度共同报告，不能只展示平均预算；
5. speedup 的分母必须是 Full-INT8 QK + 同一 Top-p实现，且双方都排除 setup/host sync；
6. P2 只有在真实 cache、真实 indices和 GQA 都接通后，才允许报告端到端模型 speedup。
