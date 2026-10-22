# Early decision v1 实现 Spec

日期：2026-10-22；类型：实验实现。基线：`STP_test`，`5fb68c518f5f24839b797d6a07a635301d9cc170`。数学定义与执行结果见 [README](README.md)。

目标：提供独立于 Twilight attention patch 的 W＋X → B0 原型，验证 `(X@W_q)@(X@W_k).T` 的分解和保守筛选；不修改现有 Selector、不调用模型加载器。

| 接口 | 输入 | 输出／约束 |
| --- | --- | --- |
| `fit_bilinear` | 同 device 的有限浮点 Wq/Wk `[D,d]`，rank | FP64 因子及尾基；不构造 `[D,D]` A；rank 允许 0 到 min(D,d) |
| `new_cache`／`append` | `norm` 或 `tail`；合法历史 X `[N,D]` | 只保存低维特征和每 token 范数；输入不修改 |
| `bounds` | 单 query `[D]` | estimate/lower/upper `[N]`；拒绝空历史 |
| `topk_candidates` | 分数区间、k | 布尔 B0；ties 保留，不限制候选上限 |
| `mass_candidates` | 分数区间、p∈(0,1] | 估计分数前缀 B0 与质量下界；p=1 返回全集 |
| `quantize_projection` | W `[D,d]`、bits=2..8 | 每输出列的对称 QDQ；以量化后的 W 定义参考 |

原型假设：单 head、无 bias、无激活 Q/DQ、无逐 token RoPE。可选共享 score_transform 只定义统一的双线性形式。FP64 guard 是数值裕量，不是完整浮点认证。

benchmark 的 dense QK 仅作独立 oracle 评估；核心模块不接收完整 Q/K 或 oracle mask。外部张量 CLI 使用固定 schema 和 `weights_only=True`，不覆盖输入文件。

验证：数学等价、误差界、Top-k 超集、质量下界、ties、零值、极端 logits、无效输入、量化参考和 RoPE 反例。合成实验保留有利与不利谱，报告元数据／特征构建开销，不报告伪硬件加速。

后续接入前的必需工作：bias／RoPE／激活量化的算术定义、真实 trace、GQA 复用、cache 更新与 attention 接口设计，以及与 DS、全分数、MSB 路径的公平比较。
