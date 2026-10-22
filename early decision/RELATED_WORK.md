# W＋X 提前生成 B0：相关工作与证据边界

类别：当前准备工作／文献依据。检索日期：2026-10-22；范围：重点 2023–2026，补充必要早期先例。代码对照基线：`f69ec467ce5a81e26b1e828a2a040839c389ac7f`，分支 `STP_test`。本报告不记录新的模型实验，也不将其他论文的结果作为本原型结果。当前原型与口径分别见 [README](README.md)、[ATTENTION_PROTOCOL](ATTENTION_PROTOCOL.md)；来源清单见 [JSON](measurements/2026-10-22-literature-sources.json)。这是本轮准备材料，不另行分配历史总结编号。

## 1. 可直接使用的判断

**已有充分一手文献支持：在完整历史 QK 评分之前，用更便宜的特征／预测器选择历史 KV 是可行的研究路线；已有明确数学先例把原始 Wq、Wk 合成输入空间双线性形式并做 SVD。不能把“W＋X”“不计算完整 QK”或“joint-SVD”本身作为新颖性保证。**

最接近当前路线的三个不同方面是（另需直接比较下文 CLOVER）：

1. **A³（ICML 2026）**直接研究 `Wq Wkᵀ` 的联合低秩近似及由 X 统计量加权的闭式解；目标是压缩 attention 维度，未据此建立本原型的残差认证 B0。原文 [§3.1，式 9–13](https://arxiv.org/html/2505.12942v2#S3.SS1)，[正式论文页](https://proceedings.mlr.press/v306/wong26b.html)。
2. **Palu（ICLR 2025，预印本 2024）**明确把分解 Wk 的重建因子离线融合入 Wq，用 X 直接产生 latent query；同时明确指出普通 RoPE 下固定融合不成立。原文 [§3.1.1，式 6；§3.1.2](https://arxiv.org/html/2407.21118v2#S3.SS1)。
3. **Sparse Attention Decomposition Applied to Circuit Tracing（2024 预印本）**对 `Wq Wkᵀ` 的双线性形式做 SVD，并把输入对的 score 分解成少数奇异方向贡献；目标是解释 GPT-2 的电路，没有 KV 筛选系统。原文 [§3，式 1–3；§4.1，式 5–6](https://arxiv.org/html/2410.00340v1#S4.SS1)。

与“生成 mask”的任务更直接接近的是 **LoRA-Sparse、SALS、Double Sparsity 和 InfiniGen**，但它们的输入和预测位置不同。尤其要纠正一个容易产生的误读：**LoRA-Sparse 原文式 4 是 `Q̂=QWq̂，K̂=KWk̂`，不是直接以 X 为输入的独立新投影。** 无位置变换时可以代数融合为 X 的线性映射；这是推导可行性，不能改写为论文已经采用该实现。见 [作者 PDF，§3.3，p.4，式 4](https://www.yingcong.me/publication/2024-song-lora/2024-song-lora.pdf#page=4)。

本轮尚未确认已有论文完整实现“冻结原始权重的 joint-SVD＋逐 token 残差区间＋认证 B0＋只在 B0 内执行原 attention”的同一组合；**这只是当前检索的未发现项，不能推出组合具有足够创新性或实际收益**。下文列明已覆盖与待查部分。

## 2. “在 QK 前”的分层定义

统一行向量记号：`Wq,Wk∈R^{D×d}`，`q_t=x_tWq`，`k_j=x_jWk`；`B0(t)` 是允许进行后续精确 attention 的历史位置集合。

| 层级 | 生成 B0 时已依赖什么 | 可以声称 | 不可自动声称 |
| --- | --- | --- | --- |
| 完整 QK 前 | 完整 q，以及完整／压缩历史 k | 避免全维全历史 QK | 避免完整 q/k 投影 |
| 部分／低维 QK | q/k 的 channel 子集、PCA 坐标、latent 或低精度表示 | 用廉价 proxy 先筛选 | 没有 query–key 相似度运算 |
| 原投影前的输入特征 | x_t、历史 x 的已缓存特征、由 W 导出的映射 | 可绕过原全维投影生成 selector 特征 | 不存在任何投影或 pairwise score |
| 前层预测 | x_t^(l−1) 代入目标层映射 | 在目标层真正输入形成前预取 | 是当前层精确输入的等价重写 |
| 编码／索引路由 | hash、码本、聚类中心、块统计 | 无须扫描完整原始 K | 完全无 q/k 衍生计算，或精确选出 oracle Top-k |

因此当前原型应称为“**基于原权重双线性形式的输入空间低维预测，在完整 QK 前生成 B0**”。它仍计算 `φq(x_t)·φk(x_j)`；这在数学上就是低维 QK 类似度。历史 token 原有 K/V 是否已经算过、当前 q 是否仍因下游执行而必算，也应与 selector 的依赖关系分开统计。

## 3. 三组公式级对应与两个直接 selector 近邻

### 3.1 A³：联合权重分解是直接先例

原文将单头 score 写为 `a_i=x_q Wq_i Wk_iᵀ x_kvᵀ=x_q Wqk_i x_kvᵀ`。在输入统计假设下，令 `Cq=R_xqxq^(1/2)`、`Ck=R_xkvxkv^(1/2)`，其低秩解为：

```math
\widetilde A=C_q^{-1}\operatorname{SVD}_r(C_q A C_k)C_k^{-1},\qquad A=W_qW_k^\top.
```

再把该结果的两侧因子作为新的低维 Q/K 权重。无权重化的情形对应本原型的普通截断 SVD；本原型通过 thin-QR 和小矩阵 SVD 避免显式分解 `D×D` 大矩阵，是计算方式差异，不能单凭这一等价计算认定方法新颖。原文 [§3.1，定理 3、式 9/12/13；§3.4 的 GQA 与 RoPE 分支](https://arxiv.org/html/2505.12942v2#S3.SS1)。

论文是 post-training 压缩，主要改变模型内部维度；它没有证明本原型的 B0 retained mass 或逐项排除保证。普通 NoPE 推导不能直接搬到 RoPE；其 RoPE 版本采用保留频率配对的 CUR 路线。真实系统证据包括原文 §4 的模型质量和 GPU throughput，而不是 B0 selector 的性能。会议身份由 [PMLR](https://proceedings.mlr.press/v306/wong26b.html)核实。

### 3.2 Palu：原始 W 的离线融合与 X latent

原文先分解 `Wk≈AkBk`，缓存 `Hk=XAk`。式 6 明确给出：

```math
q_tK^\top=x_tW_q(H_kB_k)^\top
=x_t[W_qB_k^\top]H_k^\top.
```

这里的等号针对采用分解表示后的 K；若使用截断分解，它相对原模型本来就有近似误差。融合矩阵 `WqBkᵀ∈R^{D×r}` 使评分不必先显式构造原 d 维 q。原文 [§3.1.1，式 6](https://arxiv.org/html/2407.21118v2#S3.SS1)。

这已经包含“W＋X→latent pairwise score”的关键代数思想，但 Palu 的主要任务是 KV cache 压缩／重建，没有通过该式先选 B0 再恢复原 attention。其 §3.1.2 对普通 RoPE 改为动态重建 K；§4.5 有 GPU latency 实验。需要校准／rank 配置，不需要为基本分解重新训练模型。会议版本见 [ICLR 2025 原文](https://openreview.net/pdf?id=LWMS4pk2vK)。

**RoPE 精确条件（本报告推导）：**按当前行向量约定，`q_t=x_tWqR_tᵀ，k_j=x_jWkR_jᵀ`，于是 score 含 `WqR_tᵀR_jWkᵀ`。RoPE 对固定位置是线性变换；障碍是该矩阵随相对位置变化，通常不能由单个固定 `WqWkᵀ` 索引覆盖。不能将原文对融合的限制泛化为“RoPE 数学上非线性”。

### 3.3 Franco–Crovella：双线性 SVD 与输入特征先例

**版本日期核验：**[arXiv 提交史](https://arxiv.org/abs/2410.00340)为 v1 2024-10-01、v2 2024-10-10、v3 2024-10-28；本轮读到的 v1 HTML 顶部却呈现正文日期 2026-08-24。这里按可核验提交史称“2024 预印本”，公式定位针对所读 v1 HTML；呈现日期不一致保留待核实，不据此臆定会议发表或公式出现的精确版本时间。

原文把 bias 并入齐次坐标 `x̃`，构造包含 `WqWkᵀ` 的矩阵 `Ω`，再分解 `Ω=UΣVᵀ`。忽略 bias 后其式 5–6 的贡献形式就是：

```math
s_{ij}=\sum_a (x_i u_a)\sigma_a(x_j v_a),\qquad
s_{ij}\approx\sum_{a\in S_{ij}}(x_i u_a)\sigma_a(x_j v_a).
```

这是本原型输入特征乘积的直接数学先例。差异是论文的方向集合 `S_ij` 依赖 token pair 和任务，并非统一截取前 r 大奇异值；实验是 GPT-2 small 的 IOI 电路分析，不是 KV 选择／硬件加速。原文 [§4.1，式 5–6 与集合构造](https://arxiv.org/html/2410.00340v1#S4.SS1)。

其后续 **Singular Vectors of Attention Heads Align with Features（ICML 2026）**补充条件性理论和 toy／真实模型分析，尤其指出重要 score 贡献可能来自小奇异值，稀疏贡献不等于谱能量集中。这个反例对“取前 r 即能可靠判重要 KV”构成直接提醒。[原文 §6、图 8–9](https://arxiv.org/html/2602.13524v2#S6)，[会议记录](https://proceedings.mlr.press/v306/franco26a.html)。上述证据支持可分析性，不支持任意 Llama head 的高召回保证。

### 3.4 LoRA-Sparse：先 proxy 再稀疏 attention，但原式依赖 Q/K

原文 [§3.3，式 4；§3.4；图 2](https://www.yingcong.me/publication/2024-song-lora/2024-song-lora.pdf#page=4)是：

```math
\widehat Q=QW_{\hat q},\quad \widehat K=KW_{\hat k},\quad
B_0=\operatorname{TopIndices}(\widehat Q\widehat K^\top).
```

`Wq̂,Wk̂` 是额外训练的低维映射；order-mimic 用完整 attention 排序作为训练监督。论文 §3.2 的无微调稀疏实验先算完整 map，不能当成无需训练的可执行 selector。§4 有 LLaMA／LLaVA 的质量和计算量证据；本轮没有核实到完整 selector 开销下的端到端 GPU 加速表，不能用“减少 attention 运算”替代“系统加速”。[CVPR 2024 官方身份](https://openaccess.thecvf.com/content/CVPR2024/html/Song_Low-Rank_Approximation_for_Sparse_Attention_in_Multi-Modal_LLMs_CVPR_2024_paper.html)。

**仅作等价推导：**若 `Q=XWq，K=XWk` 且没有介入的位置变换，可融合成 `X(WqWq̂)` 和 `X(WkWk̂)`；这使输入空间实现成为可能，但并非“原文证明独立从 X 学习了 selector”，也并非免训练原权重 SVD。

### 3.5 InfiniGen：前层输入＋目标层部分权重＋部分 K

InfiniGen 离线用正交 A 调整 Q/K 的坐标，使 `QA(KA)ᵀ=QKᵀ`；prefill 再挑选对应列 C。decode 在第 `l−1` 层用其 attention 输入猜测第 l 层：

```math
\widehat q_t^{(l)}=x_t^{(l-1)}(W_q^{(l)}A^{(l)})_{:,C},\qquad
\widehat s_{tj}^{(l)}=\widehat q_t^{(l)}\cdot(\widetilde k_j^{(l)})_C.
```

该式为按原文数据流统一后的记号，见 [OSDI 2024 原文 §4.2–4.3、图 8–10，论文页 161–162](https://www.usenix.org/system/files/osdi24-lee.pdf#page=7)。它无需先形成目标层真实完整 q，确实更早用到了 X 和 W；但历史侧依赖 partial key cache，仍有 partial QK，且跨层输入替代是近似。需要离线变换和 prefill 列选择，基本方案没有额外 predictor 训练；有 GPU／CPU offloading 的真实系统实验，不能推出当前层 WqWkᵀ 残差保证。

### 3.6 CLOVER：与普通原权重 joint-SVD 更直接对应

[ICML 2025 正式记录](https://proceedings.mlr.press/v267/meng25d.html)确认其不是只有博客版本。作者的 [ICLR 2025 Blogpost，Cross-Layer Absorption／Orthogonal Decomposition](https://iclr-blogposts.github.io/2025/blog/pessa/)明确先合并 `A=WqWkᵀ`，再做 `A=UΣVᵀ`，把非零／保留奇异方向作为新的小型投影，按奇异值裁剪 attention 维度；原始 [2024 v1，§3.1](https://arxiv.org/html/2411.17426v1#S3.SS1)也写出 absorb-decompose。该版本标题与正式版不同，引用时需保持版本一致。

按本报告行向量记号，其低维 score 可写为 `XUrΣrVrᵀXᵀ`；把 Σ 分配到单侧或开平方分到两侧不改变双线性结果。其目标是参数／cache 维度裁剪与后续微调，没有逐 query B0 或本原型尾残差认证。基础正交化／部分裁剪可以无训练，恢复质量的流程涉及微调；普通 RoPE 仍有限制。A³ [§4.2](https://arxiv.org/html/2505.12942v2#S4.SS2)明确把 CLOVER 对应为输入协方差取单位阵的联合近似。**因此“普通 joint-SVD＋原始 X 的低维特征”已有非常直接的方法先例。**

## 4. 其他候选的机制筛选

表中“训练”指额外训练要求；校准不等于训练。“原投影前”指 selector 能否不依赖目标层原完整 q/k 才开始，不能理解成历史 KV 从未生成。每行定位对应原论文，不从同名方法推断机制。

| 工作／来源状态 | B0 或选择依据／位置 | 训练／校准 | 实际证据及与本原型的关系 |
| --- | --- | --- | --- |
| [Double Sparsity，2024 预印本](https://arxiv.org/html/2408.07092v2#S4)，§4.1–4.2 | 对 Q/K 的静态 heavy channels 形成 label，再点积选 token；在完整历史 QK 前 | 无额外模型训练，有 offline calibration | GPU attention／端到端／offloading 实验；与输入空间原权重分解不同 |
| [Double Sparsity-Offload，同文 §5.1](https://arxiv.org/html/2408.07092v2#S5.SS1) | 前层 embedding 代入下一层 Wq 生成近似 query，再访问下一层 label cache | 同上 | 不能笼统说整个 DS 家族必须等目标层完整 Q 产生才预测 |
| [SparQ，ICML 2024，预印本 2023](https://arxiv.org/html/2312.04985v6#S3)，§3、Algorithm 1 | 由当前 q 的大绝对值 channel 读取 K 子集，proxy score 后 Top-k | 无额外训练／基本规则无需离线拟合 | 有实现与质量／效率证据；依赖 q，是低维 QK，不是原投影前 W＋X |
| [Loki，2024 预印本](https://arxiv.org/html/2406.02542v2#S4)，方法节 | K 的 PCA 低维空间内排序 token，再对选中项作更完整评分 | 无微调，需求 PCA 变换 | 有优化 kernel 实测；分解对象是 activation K，不是联合 WqWkᵀ |
| [SALS，NeurIPS 2025](https://arxiv.org/html/2510.24273v1#S4.SS3)，§4.2–4.3、Algorithm 1 | pre-RoPE 多头 K 校准协方差求 U；`q̃=Uᵀq,k̃=Uᵀk`，取前 r* 维点积；选中 K 重建并加 RoPE | offline calibration，无额外 predictor 训练步骤 | 有模型质量、GPU operator／端到端结果；算法输入明确是 pre-RoPE q/k，不是直接输入 X |
| [ShadowKV，ICML 2025](https://arxiv.org/html/2410.21465v2#S4.SS1)，§3–4；[作者代码](https://github.com/ByteDance-Seed/ShadowKV) | pre-RoPE key activation 低秩缓存，chunk 代表／稀疏重建与 V offload | training-free，prefill 分解 | 有 A100 真实吞吐；低秩存储与筛选有价值，但不能改写成离线联合原 W 的 X predictor |
| [Quest，ICML 2024](https://arxiv.org/abs/2406.10774)，方法节；[作者实现](https://github.com/mit-han-lab/quest) | 每页 K 的逐通道 min/max 与当前 q 形成页重要性上界，再 Top-k pages | 无额外训练 | GPU 稀疏执行；与“区间／边界先筛选”相邻，但边界建于 K 空间／页，不是本原型残差区间 |
| [ELSA，ISCA 2021](https://taejunham.github.io/data/elsa_isca21.pdf)，§III、图 4 | 对既有 Q/K 做 sign-random-projection hash、Hamming 角度估计和 norm 筛选 | 阈值校准 | 专用硬件设计评估；原文还报告近似算法在常规 GPU 上可能变慢，不是免 Q/K 投影路线 |
| [DOTA，ASPLOS 2022](https://par.nsf.gov/servlets/purl/10357543)，§3.1–3.2、图 4 | 检索定位到从 X 开始的 sparse-random-projection＋低秩 detector；仍形成低维 pairwise score | 联合训练 detector 与模型 | 一手全文本轮打开超时；仅已索引原文片段定位，公式与硬件实测性质保留待复核，不以其独立支撑核心结论 |
| [DeepSeek-V3.2，2025 技术报告](https://arxiv.org/html/2512.02556v1#S2)，DSA 部分 | `I_ts=Σ_h w_th ReLU(q^I_th·k^I_s)`，独立 lightning indexer 产生 Top-k 再 sparse MLA | indexer warmup／continued training | 模型及系统证据；可不使用主 attention 的完整 QK，但仍是训练后的低维 query/key indexer，非免相似度 W-only 预测 |
| [HISA，2026 预印本](https://arxiv.org/html/2603.28458v1)，方法与实验 | block 代表先粗筛，再用原 indexer 细排 | 无新增微调 | DSA 的层次索引加速，说明 selector 自身扫描成本须计入；非原 W 的 bilinear 分解 |
| [Fathom，2026-09 预印本](https://arxiv.org/html/2609.17652v1)，方法与实验 | 每 query 调整压缩 key 扫描读取深度 | 本轮未完成训练／校准协议细查 | 已核实论文存在与系统定位；非联合 W 的输入空间预测，列作 2026 近邻跟踪，不把其基线实现视作 DS 官方实现 |

**SALS 的关键边界：**其不加 RoPE 的 latent selector 对最后 post-RoPE 排序是经验近似；“只在重建后加 RoPE”并不证明 selector 保持真实相对位置。由 `q=XWq` 可代数融合 `WqU`，但原文 Algorithm 1 接口是 pre-RoPE q/k；应区分可推导融合与已实现融合。[原文 §3、§4.3](https://arxiv.org/html/2510.24273v1#S4.SS3)。

## 5. 本原型的窄差异、风险与下一步依据

当前实现 `A=WqWkᵀ≈UrΣrZrᵀ`，用 `φq(x)=xUr√Σr` 和 `φk(x)=xZr√Σr` 评分；残差提供 `ŝ_j±ε_j`，再做候选保留。这是仓库事实／数学推导，不是任何论文已验证的结果。

| 已被覆盖的思想 | 仍可核查的窄差异 | 必须新增的证据 |
| --- | --- | --- |
| A³／Franco 的联合 bilinear SVD、Palu 原 W 的离线吸收 | 保留原模型执行路径，仅以未训练的 joint-SVD predictor 建立逐项区间 B0 | 对比 activation-weighted joint-SVD、Palu 等价 latent 与 DS/SparQ，不能只对比完整 QK |
| LoRA-Sparse／SALS 的低维 proxy→Top-k→精确选中 attention | 用可证明残差排除代替经验 Top-k，允许候选数自适应 | 逐 query 的无漏选条件、retained mass、界宽与 B0 比例；谱不利时如实保留接近全选结果 |
| Quest 的上界筛选、各类量化／分层过滤 | 在输入特征中建立 score 或 cosine 的严格区间 | 还需补查 interval/MIPS/ANN 筛选理论；不能把普通 Cauchy–Schwarz 尾界本身当新定理 |
| InfiniGen／DS-Offload 的前层预取 | 当前层真实 X 的认证，而不是邻层 X 的经验猜测 | 时序图与可用输入，证明预取／选择发生时所需 X 已存在；不能同时借用前层时机与本层精确保证 |
| RoPE 下的 latent／重建已有系统路线 | 对真实逐 token 相对旋转给出有效选择器与误差界 | 当前固定 A 的保证不适用；需新增推导与真实模型 trace 验证，不能仅以 pre-RoPE 高相似度代替 |

可行性应分为三层：**数学上可重写：已支持；局部合法假设下可建界：由本原型检查；真实 RoPE LLM 上优于 DS 的质量／延迟：尚未由本报告或这些论文证明。** 这使路线值得做最小验证，但不构成强创新结论。

优先验证对象可以沿用相关工作已有的 Llama-2／Mistral／Llama-3.1 家族以及 WikiText-2、LongBench、RULER；必须固定模型、上下文、预算和量化口径。先比较 B0 召回／precision／F1／Jaccard、retained mass、输出余弦与相对误差，再测包含 SVD 摊销、cache append、特征存储、proxy 扫描、选择、选中 QK/AV 的总时延。相关工作的最大加速不能跨这些配置移植。

## 6. 检索范围、评分与缺口

采用 `ccf-literature-searcher` 标准模式，使用公开论文名与关键词检索，没有把未发表方案原文上传。入口包括 arXiv、CVF、PMLR、NeurIPS、USENIX、作者主页／代码；最终机制依据优先原文。排除来源只作内部筛选，不在此列入引用。检索过 21 个可区分候选／来源簇：上文 18 个去重工作（DS 的 offload 仍算同一工作），另有 Long Exposure、PADE 与 Lorsa 作为远邻／待深化项；没有把它们当成已核实近邻。

主要公开查询包括：`LoRA-Sparse CVPR 2024`、`DOTA Detect and Omit Weak Attentions`、`SALS Sparse Attention in Latent Space`、`InfiniGen partial weights`、`attention bilinear sparse SVD weights`、`Sparse Attention Decomposition Applied to Circuit Tracing`、`Analytical Low-Rank Approximation Framework for Attention`、`DeepSeek sparse attention indexer`。来源 JSON 记录稳定 URL 与定位，而不保存大段版权原文。

下列 1–5 分只评估**对当前检索问题的洞见、完整度、实验数字证据强度**，不是论文接收概率；根据本轮检查范围作诊断。类型使用技能标准分类，所有评分均中等置信，待查项降低完整度。

| 工作 | 类型 | 洞见／完整度／数值证据 | 评分依据 |
| --- | --- | --- | --- |
| A³ | pure method | 5／4／4 | 直接联合权重推导、真实模型与 GPU，但目标为压缩 |
| CLOVER | pure method | 5／4／3 | 普通联合 SVD 与裁剪，正式论文和可核验作者方法说明 |
| Palu | system/tool | 5／4／4 | 离线融合公式、RoPE 条件、kernel／质量齐备 |
| Sparse Attention Decomposition | pure method | 5／3／3 | 直接 SVD 数学先例，实验集中 GPT-2/IOI |
| Singular Vectors Align | theory/proof | 4／4／3 | 条件理论＋toy／特定真实模型，无推理系统 |
| LoRA-Sparse | pure method | 4／4／3 | 原文训练／selector 明确，端到端速度证据不足 |
| InfiniGen | system/tool | 5／5／4 | 预测时机、权重、offload 系统可定位 |
| DS | system/tool | 4／4／4 | 校准、label、GPU／offload 完整 |
| SparQ | pure method | 4／4／4 | 训练无关 selector、实现与多任务验证 |
| Loki | pure method | 4／4／3 | PCA 低维筛选与 kernel，适用模型有范围 |
| SALS | system/tool | 4／4／4 | 位置动机、latent 筛选与系统结果 |
| ShadowKV | system/tool | 4／5／4 | 重建／offload 集成及真实吞吐 |
| Quest | system/tool | 4／4／4 | 页上界与实际稀疏执行 |
| ELSA | system/tool | 4／4／3 | 原文算法／架构清晰，硬件指标不可当 GPU 实测 |
| DOTA | system/tool | 4／2／2 | 一手全文访问缺口，分数仅作检索优先级 |
| DeepSeek-V3.2 | system/tool | 4／4／4 | 学习式 indexer 与模型／系统，范围有别 |
| HISA | system/tool | 4／3／3 | 近期预印本，层次 selector 与质量／kernel 结果 |
| Fathom | system/tool | 3／2／2 | 近期预印本，仅完成定位，协议未彻查 |

检索缺口：

- **不能宣布穷尽 2026。** 本轮覆盖到了 2026-09 的公开候选，但没有系统遍历所有 2026-10 arXiv、ICLR 2027 投稿、专利及非英文资料。
- DOTA 原始 PDF 多次访问超时／出版站 403；保留公式和评估性质待全文复核。Long Exposure、PADE、Lorsa 尚未做公式级比较；不以其标题或二手摘要支持结论。
- 尚未穷尽训练无关 ANN/MIPS、branch-and-bound、量化界、safe screening 文献；“残差认证 B0”的创新性需要这些近邻检索，不能只查 LLM sparse attention。
- A³ 使用了输入统计加权；本原型按原权重谱截断可能不如数据感知基线。解释性工作的“稀疏方向”也可能不是最大奇异值，必须用真实 X 检查。
- 本轮读了原文机制与实验章节，未下载并运行任何他人实现；报告中的“系统证据”指论文报告的实际系统实验，非本机复现。

交接重点：将 A³、Palu、Franco 的公式作为数学先例；将 DS/SparQ、LoRA-Sparse/SALS、InfiniGen 分开作为 selector／训练／提前时机基线；保留 RoPE 与端到端成本为当前主要未验证项。后续应先补最小实验证据和上述缺口，再决定如何定位论文贡献。
