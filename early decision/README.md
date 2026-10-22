# Early decision：从 X 与投影权重提前生成 B0

日期：2026-10-22。当前包含 ED-R001 数学原型与 ED-R002 的 Twilight opt-in 接入、CPU attention 对照；尚未运行完整模型。此目录位于 `STP_test` 的 `early decision/`。

当前入口：

- [相关论文与公式级先例](RELATED_WORK.md)：CLOVER／A³、Palu、LoRA-Sparse、SALS、InfiniGen 等；不将联合权重分解本身声称为新意。
- [Early Decision-only attention 实验说明](TWILIGHT_INTEGRATION.md)：启用后绕过 DS 和 Twilight Top-p，仅用 B0 中 KV 计算正式 attention。
- [自动生成的结果摘要](measurements/2026-10-22-attention-summary.md)、[完整原始 JSON.gz](measurements/2026-10-22-attention-comparison.json.gz)：144 配置、576 个 decode step，输出质量、两套 KV oracle、CPU 时间与完整成本。
- [本轮历史总结](../.codex/history/ED-R002.md)：从前序数学原型到当前发现及下一步。

以下保留数学原型的定义、推导和 ED-R001 验证范围；新接入接口及本轮结果以上述 ED-R002 文档为准。

## 1. 先回答数学问题

采用用户的行向量约定，$X\in\mathbb R^{N\times D}$，$W_q,W_k\in\mathbb R^{D\times d}$，忽略 bias、RoPE 与激活 Q/DQ：

$$
QK^\top=(XW_q)(XW_k)^\top=X\underbrace{(W_qW_k^\top)}_A X^\top.
$$

Decode 将左右 X 分成当前 $x_t$ 与历史 $X_H$。静态权重可以离线处理；但 A 是 $D\times D$，直接乘 A 或扫描 D 维历史 X 通常更贵。**等价变换本身没有产生稀疏，也没有消除动态 token 比较。**

本原型研究：不形成 A、不先生成完整 Q/K 打分，以 A 的低秩因子从 X 建立历史特征，通过误差界生成保守 B0。它仍然是低维相似度预测器，不能解释成“完全没有类似 QK 的计算”，也不声称该机制已有新颖性。

## 2. 不构造 D×D 矩阵的分解

做 thin QR 和小矩阵 SVD：

$$
W_q=Q_qR_q,\quad W_k=Q_kR_k,\quad
R_qR_k^\top=L\Sigma V^\top.
$$

因此 $A=U\Sigma Z^\top$，其中 $U=Q_qL$、$Z=Q_kV$。当 $D\ge d$ 时，SVD 只处理 $d\times d$ 核心；离线开销包含 QR、SVD 与因子展开，不能作为免费工作。

取前 r 个奇异方向，定义：

$$
F_q=U_r\Sigma_r^{1/2},\qquad F_k=Z_r\Sigma_r^{1/2},
$$

$$
a_t=x_tF_q,\quad b_j=x_jF_k,\quad
\widehat s_{tj}=a_tb_j^\top/\sqrt d.
$$

历史 token 写入时缓存 $b_j$ 与一个误差范数，不保留原始历史 X；在线通过当前 $x_t$ 和缓存生成 B0。完整历史 K/V 可以照常存在，但 Selector 不读取它们。

与 DS 的固定输出通道不同，这里保留的是由 $W_qW_k^\top$ 决定的双线性奇异方向。两者都属于近似相似度计算；区别需要从质量、成本和复用证明，而非仅从“QK 之前”的名字判断。

## 3. 分数区间与提前排除

完整缩放分数为 $s_{tj}=x_tAx_j^\top/\sqrt d$。截断残差满足：

$$
|s_{tj}-\widehat s_{tj}|
\le\epsilon^{norm}_{tj}
=\frac{\sigma_{r+1}\|x_t\|_2\|x_j\|_2}{\sqrt d}.
$$

如果在写入时另外计算 $\|x_jZ_{tail}\|_2$，当前 query 计算 $\|x_tU_{tail}\|_2$，则有更紧的界：

$$
\epsilon^{tail}_{tj}=
\frac{\sigma_{r+1}\|x_tU_{tail}\|_2\|x_jZ_{tail}\|_2}{\sqrt d}.
$$

这是因为残差为 $U_{tail}\Sigma_{tail}Z_{tail}^\top$。定义 $L_j=\widehat s_j-\epsilon_j$、$U_j=\widehat s_j+\epsilon_j$，即可在完整 QK 前决策。全秩时残差为零。

### Top-k 候选

令 $\tau$ 为所有 $L_j$ 中第 k 大的值，保留 $B_0=\{j:U_j\ge\tau\}$。被排除项的真实分数小于至少 k 个保留项，因此 B0 包含真实 Top-k；边界 ties 保守保留。不能为了固定预算再硬截断 B0，否则该性质失效。

### 概率质量候选

任意集合 B 的真实 softmax 质量下界为：

$$
\underline m(B)=
\frac{\sum_{j\in B}e^{L_j}}
{\sum_{j\in B}e^{L_j}+\sum_{j\notin B}e^{U_j}}.
$$

代码按估计分数降序扩展前缀，直到 $\underline m(B_0)\ge p_0$，以 log-space prefix/suffix reduction 避免 exp 溢出。这不是精确 Top-p 集合，也不保证在所有子集中候选数最少。

如果后续仅在 B0 内按**同一完整参考**进行条件 Top-p，条件阈值为 $p_1$，最终全域质量至少为 $p_0p_1$。若后续使用近似 MSB 分数，这个乘积保证不能自动沿用。

以上是实数算术下的推导。实现使用 FP64 和可配置数值 guard；没有实现向外舍入或完整浮点误差证明，**不能称为严格浮点认证算法**。测试与 benchmark 用独立的直接 QK 检查数值覆盖。

## 4. 量化与 RoPE 的边界

- `quantize_projection` 对行布局 `[D,d]` 的每个输出列做对称 4/8-bit QDQ。它得到冻结的反量化权重，再对这些权重分解；不是真正整数 kernel。
- 区间约束的是**这一冻结权重配置的线性分数**，不是原始 FP 权重分数，更不是 RoPE 后再次量化的 Q8×K8 分数。历史特征也以 FP64 保存；将它压成 INT8／INT4 后需要另加特征量化误差界。
- 不支持 bias。bias 非零时要显式处理额外项或增广输入，不能直接沿用当前实现。
- 标准 RoPE 的分数为 $x_tW_qR_t^\top R_jW_k^\top x_j^\top$。中间变换随 token pair 改变，一个固定 A 一般不能用于整段历史。
- `score_transform=T` 仅支持**整份索引共享同一个** $d\times d$ 矩阵，等价分解 $W_qTW_k^\top$，不代表已经支持逐 token RoPE。按所有相对位置分别拟合索引的额外成本未实现，不能作为免费解决方案。
- 当前索引是单 head；GQA 可逐 Q/K head 配对调用，但不代表共享索引或物理访存已优化。只传入合法历史项；完整 prefill 的 causal mask 由调用者逐 query 处理。
- $W_v$ 不出现在 QK logits 中，本原型不使用它。面向输出误差的筛选属于另一种目标。

## 5. 文件与接口

| 文件／功能 | 修改前 | 修改后 | 验证 |
| --- | --- | --- | --- |
| [early_decision.py](early_decision.py) | 无输入空间 B0 原型 | 小矩阵分解、norm/tail 界、Top-k 超集、概率质量 B0、权重 QDQ | [单元测试日志](measurements/2026-10-22-unit-tests.txt) |
| [benchmark.py](benchmark.py) | 无该路线的数值及成本记录 | 合成谱对照、FP/4/8-bit 冻结权重、真实质量诊断、逻辑 MAC／缓存计数；支持外部张量 | [完整测量 JSON](measurements/2026-10-22-synthetic.json) |
| [tests/test_early_decision.py](tests/test_early_decision.py) | 无对应检查 | 数学等价、区间、ties、极端 logits、零输入、量化、RoPE 反例及参数验证 | 15 项通过 |

示例：

```python
from early_decision import fit_bilinear, mass_candidates

# PyTorch Linear.weight is [d,D]; transpose it to this API's [D,d].
model = fit_bilinear(W_q, W_k, rank=4)
cache = model.new_cache(bound_mode="tail")
cache.append(X_history)              # [N,D], only legal history for this query
bounds = cache.bounds(x_current)     # [D]; no full QK here
B0 = mass_candidates(bounds, p=0.9).mask
# Caller computes full QK / Top-p / final attention only within B0.
```

仅依赖 Python >=3.10 与 PyTorch；不依赖 Transformers、GPU 或 NumPy。仓库根目录运行：

```text
python -m unittest discover -s "early decision/tests" -v
python "early decision/benchmark.py" --output "early decision/measurements/my-run.json"
python "early decision/benchmark.py" --input tensors.pt --ranks 0,2,4 --output "early decision/measurements/provided-run.json"
```

`tensors.pt` 用 `torch.save` 保存字典：`W_q,W_k:[D,d]`、`x_query:[M,D]`、`x_keys:[N,D]`，可选 `score_transform:[d,d]`。数据通过 `weights_only=True` 读取；不要将输入输出设为同一路径。真实 post-RoPE Q/K trace 不能直接作为这里的投影前 X。

本机使用项目内既有 CPU Python 环境，版本记录在测量 JSON。复现不要求安装到系统 Python。

## 6. 成本与诚实的失败条件

历史特征构建为 $NDr$ MAC；tail 模式另外需要 $ND(m-r)$ MAC 的历史尾方向投影，$m=\min(D,d)$，范数计算另计。当前 query 同样承担 $Dr$ 与可选 $D(m-r)$ 投影；全域粗分数扫描为 $Nr$ MAC。保守 B0 确定后，若不能复用精确分数，再支付 $|B_0|d$ 完整 QK MAC。

与已有完整 KV cache 并用时，这些特征构建是**额外成本**，本版本未实现投影融合。tail 范数只保存每 token 一个标量，但计算它并不免费。代码的 append 还会复制已有特征；生产实现应使用预分配 cache，并重新评估访存。

每 token 实际保存 `(r+1)*8` bytes，因子与尾基还占额外权重存储；JSON 单列实际 FP64 大小，不将其冒充 INT4 cache。full-rank 时估计分数本身可复用，不能再强制重算来夸大 baseline 成本。

若谱不衰减，$\sigma_{r+1}$ 大，区间宽，B0 可能接近全集；若 logits 本身分散，即使精确知道分数也需要很多 token。两者是不同失败原因，实验同时报告 oracle 预算与 B0 预算。

## 7. 已执行结果

CPU、FP64、seed=20261022；两类合成权重，FP/4/8-bit QDQ、5 个 rank、两种界，共 **60 组配置 × 12 queries**；每组检查 Top-8 及质量目标 0.9/0.99。未发现区间覆盖错误、Top-k 漏项或质量下界违反。另有 **15 项单元测试通过**。

外部张量输入路径另做 [18 组 CLI 集成检查](measurements/2026-10-22-provided-integration.json)，使用单个共享旋转矩阵与小合成输入；同样没有区间或 Top-k 违反。这不代表逐 token RoPE 支持。

以下为 tail 模式、r=4，N=128 的均值；“FP”仅表示未量化合成权重：

| 谱／权重 | Top-8 的 B0 大小 | 质量目标 0.9 的 B0 大小 | 同参考 oracle Top-p 大小 |
| --- | --- | --- | --- |
| 快速衰减／FP | 8.00 | 29.00 | 29.00 |
| 快速衰减／4-bit QDQ | 9.50 | 32.33 | 30.00 |
| 快速衰减／8-bit QDQ | 8.00 | 29.08 | 29.08 |
| 平坦谱／FP | 127.50 | 127.50 | 11.83 |
| 平坦谱／4-bit QDQ | 128.00 | 127.92 | 10.42 |
| 平坦谱／8-bit QDQ | 128.00 | 128.00 | 11.92 |

这支持数学原型在规定假设下可运行，也显示谱结构不利时几乎无法提前排除。量化不保证改善谱；低位量化可能让原本很小的尾奇异值变大。**没有完整模型 PPL、真实 RoPE 接入、DS 对比性能或硬件加速结果。**

## 8. 下一步

先采集一个真实层／head 的 W 与合法投影前 X，检查实际谱、B0 预算及特征构建成本；对 RoPE 模型先解决位置相关的分解／界，不能将本目录输出直接当作有效 Llama mask。若界普遍太松，应保留这一负面证据，再探索投影部分和、块级边界或编码路由，不将低秩路线强行认定为最终方案。
