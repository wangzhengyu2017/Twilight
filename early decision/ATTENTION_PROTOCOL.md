# ED-R002：Early Decision-only attention 对照协议

日期：2026-10-22；类别：准备工作。前序为 [ED-R001 数学原型](https://github.com/wangzhengyu2017/Twilight/blob/f69ec467ce5a81e26b1e828a2a040839c389ac7f/early%20decision/README.md)。本轮实验状态以执行记录为准。

## 目标与执行路径

固定输入、投影权重、实际执行 Q/K/V、位置变换及最终 attention 精度，比较：

- REF：标准 dense attention，$o=\operatorname{softmax}(qK^\top/\sqrt d)V$。
- RESULT：Early Decision 用 W、X 和历史特征生成 B0；仅取 B0 中的 K/V 计算 $\widehat o=\operatorname{softmax}(qK_{B0}^\top/\sqrt d)V_{B0}$。

RESULT 不调用 DS selector，也不调用 Twilight Top-p／pruner。`estimate_topk` 模式仅按 Early Decision 的估计分数取指定数量；`topk_candidates`（bound）模式以 requested-k 为目标保留区间判定的超集。它们都是 Early Decision 的候选规则；不把前序 mass-candidates 的概率质量筛选默认混入本实验。

REF／oracle 在独立评估阶段计算，不能参与 B0 的输入、排序、预算调节或运行时回退。允许全保留及 full-rank sanity 对照，但需报告其实际成本，不能强迫 REF 重算已有 score。

## 三组质量指标

### 1. 最终 attention 结果

比较实际 $o$ 与 $\widehat o$：向量余弦、相对 L2、最大绝对误差；可补充 RMSE。先按 query／head 统计，再给均值、最差或分位数；若给 flatten cosine，明确它与逐 query 均值不同。

相对 L2 采用 $\|o-\widehat o\|_2/\max(\|o\|_2,\epsilon)$。两向量都为零时相等；余弦的零范数约定需写明。只给 cosine 会漏掉幅度误差，不足以判断数值一致性。

将 sparse 概率 scatter 回完整 N 位置，补充概率 TV 或 JS，可解释 mask 与归一化造成的变化。不用加 epsilon 后的有限 KL 隐藏被剪位置原本为零的支持差异。

### 2. 重要 KV 预测

同时建立两套**仅评估用** oracle：

1. 标准 logits：$s_j=q\cdot k_j/\sqrt d$，按分数取 Top-k；它与标准 attention softmax 排序一致。
2. 用户要求的逐 token 余弦：$c_j=q\cdot k_j/(\|q\|\|k_j\|)$，按 c 取 Top-k；零范数约定另记。

两套 oracle 会因 K 范数而不同，必须分别报告。另给完整估计 logits 行与真实 logits 行的 cosine／误差，避免混淆“两个 score 向量的余弦”与“每个 q、k 的余弦”。

对预测集合 B0 与 oracle 集合 R，报告：

$$
\operatorname{precision}=|B_0\cap R|/|B_0|,\qquad
\operatorname{recall}=|B_0\cap R|/|R|.
$$

同时报告 F1、Jaccard、实际候选数／比例、全域真实 softmax retained mass。bound 模式候选数可能大于 requested-k，应区分 requested-k oracle 与同实际候选数 oracle。不采用受大量 TN 主导的集合 accuracy 作为主结论。

### 3. 保留质量

$$
m(B_0)=\sum_{j\in B_0}\operatorname{softmax}(s)_j.
$$

这是离线 oracle 指标。B0 内的 softmax 概率和恒为 1，不能把这个 1 报成全域 retained mass。

若两条路径对保留项使用同一组真实 logits，令 $m=m(B_0)$，则 $o_{ref}=m\,o_{B0}+(1-m)o_{outside}$，所以：

$$
\|o_{B0}-o_{ref}\|_2
=(1-m)\|o_{B0}-o_{outside}\|_2
\le 2(1-m)\max_j\|v_j\|_2.
$$

这是从标准 attention 归一化直接推出的范数界：保留质量高有助于限制误差，但误差还依赖 V 的幅值与方向。它不把 token 余弦召回或 retained mass 直接等同于最终输出 cosine，也不提供相对误差保证（ref 可能因抵消接近零）。

## 效率账本

| 阶段 | 必须记录 |
| --- | --- |
| 公共投影／cache | Q/K/V 投影与写入；是否已缓存、是否两路径共享 |
| Early Decision 初始化 | QR／SVD／因子构造、模型索引存储；冷启动成本与摊销次数 |
| 历史特征维护 | 首次构建与逐 token append；额外投影、tail 范数、内存复制 |
| 在线选择 | query 特征／tail 投影、低维扫描、排序／比较／mask；不把它记为零 |
| 选中执行 | 实际 B0 中完整 QK、softmax、AV；输出投影是否计入需写明 |
| 存储和读取 | cache、因子、元数据、选中 K/V；实际 dtype 字节和逻辑目标字节区分 |

报告 warm-step、初始化／append 和含初始化的摊销口径。逻辑 MAC／bytes 只是成本代理；CPU 测量注明实现、线程、warmup、重复次数，不能叫 GPU／硬件加速。若测量方法与逻辑计数范围不一致，明确分别说明。

## 数值及实现边界

- 单 head／多 head、GQA 映射、batch、合法 causal 域、位置、bias、量化配置显式记录。
- no-RoPE 的原型区间不能直接用于逐 token RoPE。若采用 pre-RoPE B0 与 post-RoPE 正式执行，标为 heuristic，重新测所有质量指标；禁止继续声称原始界已认证真实分数。
- 当前无 GPU、无完整模型 checkpoint。允许 CPU 的 Torch attention 组件、合成投影与外部 trace 先行验证；它们不等价于完整 Llama PPL 或端到端模型运行。
- 权重 QDQ 控制组不自动意味着 Q/K/V cache INT8；本轮与原三阶段模型实验分开报告。

## 复现与历史

执行时保留参数、seed、输入来源、环境、完整原始结果和测试日志。完成后形成 ED-R002 总结，关联本协议、Spec、功能 diff、执行依据、前序 ED-R001 和下一轮方案，并更新 [.codex/history/README.md](../.codex/history/README.md)。
