# ED-R002：Torch attention 接入与对照记录

日期：2026-10-22。类别：实验。基线：`f69ec467ce5a81e26b1e828a2a040839c389ac7f`；本轮变更与产物由 [历史总结](../.codex/history/ED-R002.md)关联。协议为 [ATTENTION_PROTOCOL.md](ATTENTION_PROTOCOL.md)，前序为该基线的 [README](https://github.com/wangzhengyu2017/Twilight/blob/f69ec467ce5a81e26b1e828a2a040839c389ac7f/early%20decision/README.md)。本轮沿用 input-space QR／小 SVD、持久特征 cache 和两种残差界；新增正式 selected-KV attention，改变的是执行路径与输出质量、效率证据。前序数学检查继续适用于 no-RoPE 区间，不构成真实逐 token RoPE 的证明。

## 功能变化与状态

| 功能 | 修改前 | 修改后 | 涉及文件／diff | 验证结果 |
| --- | --- | --- | --- | --- |
| Early Decision 后正式 attention | 数学原型仅产 B0 | B0 后 gather 实际 K/V，只对候选计算正式 QK、softmax、AV | [adapter](../twilight/early_decision_attention.py) | CPU float32／float64 全保留与 dense 一致；无 DS／Top-p 调用 |
| 固定预算与保守超集 | 未接 attention | `estimate_topk` 按低秩 estimate 取 k；`topk_candidates` 按区间保留 requested-k 超集 | 同上 | no-RoPE 保守模式所有实测 requested-k recall=1；固定预算没有此保证 |
| 持久 decode | 原型有 metadata append，未与 KV 执行连接 | 每 head 只初始化一次 SVD，append 历史特征及实际 KV；repeat attend 不 append／refit | 同上 | 原历史 features 不变、模型对象不变、序列长度递增、reset 验证通过 |
| Twilight runtime hook | 原 forward 先完整 QK 再 DS／pruner | forward 第一句 opt-in dispatch，默认未设置 flag 时保留旧路径；新路径包含 causal prefill 与 decode | [attention.py](../twilight/pyimpl/attention.py) | AST 执行检查 dispatch；手写 Transformers stub 的 2-query-head／1-KV-head GQA 与逐位置 dense 对照通过 |
| RoPE 与合法域 | 原型只支持固定共享变换 | no-RoPE 数学组件；runtime 只接受显式 `pre_rope_heuristic`，pre-RoPE 选 KV、post-RoPE 执行 | adapter、[tests](tests/test_twilight_integration.py) | CPU 位置旋转与 full-keep 一致；heuristic 不沿用区间／top-k 保证 |
| 质量与效率 | 未有正式输出对照 | 输出误差、两套集合 oracle、逐 token cosine、retained mass、TV、成本与 CPU 时间 | [compare_attention.py](compare_attention.py) | 合成 144 配置／576 个 persistent decode step 已执行；用户 tensor 文件入口通过临时 fixture 测试 |

实现、执行、验证边界：独立 Torch 组件和 hook 的 stub 验证已完成；真实安装版 Transformers／Llama／Mistral、完整 checkpoint、语言模型 PPL、GPU latency 均未执行，不能由本轮组件结果推断。环境为 Python 3.11、Torch `2.5.0+cpu`，没有 NumPy／Transformers／GPU；Torch 的 NumPy 初始化 warning 不影响本实验。

## 使用与合法域

独立组件可用 `importlib.util.spec_from_file_location` 加载 `twilight/early_decision_attention.py`。benchmark 与 runtime 使用同一个 `EarlyDecisionAttention` 实现；不用导入 Twilight 顶层、FlashAttention 或 Triton。

```python
engine = EarlyDecisionAttention(wq, wk, rank=4, budget=32,
                                mode="estimate_topk", rope_mode="none")
engine.append(x_history, values=v_history)
engine.append(x_current, values=v_current)
out, mask, bounds = engine.attend(x_current)
```

权重采用 `[D,d]` 行布局，X 为 `[N,D]`，单 query 为 `[1,D]`；values 为 `[N,d]`。key／query 可由公共投影预先提供，此时调用方必须保证它们确实来自同一冻结权重；heuristic 模式必须显式提供最终 post-RoPE key／query。组件不读取未来位置，只允许每次单 query；causal 性由调用方按历史→当前的顺序 append 保证。API 不接受未来索引、任意 mask、projection bias 或 attention bias。所有输入必须有限、同 device／dtype。weight QDQ 控制组是冻结的反量化浮点权重，未实现 activation QDQ 或原生整数 kernel。

在已经被 Twilight patch 的 Llama／Mistral attention module 上，先完成最终 `.to(device, dtype)` 并冻结权重，再调用：

```python
from twilight.early_decision_attention import enable_early_decision
enable_early_decision(attention_module, rank=4, budget=32,
                      mode="estimate_topk", bound_mode="norm",
                      rope_mode="pre_rope_heuristic")
```

旧 DS／Top-p 配置不再执行。必须对所有目标层调用；不是完整模型配置加载器。启用阶段先构造全部 head，成功后才发布 hooks；失败保留旧 hooks。权重或 device／dtype 变化后需要重新 enable。runtime 仅支持 batch=1、标准 GQA head 映射、从零开始连续 position_ids、legacy rotary API、无 mask／bias／sliding-window／prefill compression，以及不返回完整 attention map。未支持组合直接报错。输入前缀按 token 顺序 append 后 attend，因此 prefill 没有未来泄漏；它也是 sparse causal 组件路径，未调用原 FlashAttention prefill。

`use_cache=True` 时必须提供匹配长度的外部 DynamicCache，同时保留每 head 的 early cache。新序列必须同时新建／清空外部 cache，并调用 `attention_module.early_decision_reset()`；重复从 position 0 开始而不 reset 会报错。`use_cache=False` 的独立调用完成后清空内部 cache，避免下一样本污染；已有缓存时不允许以无 cache 模式继续。当前实现每 query head 重复存储真实 KV／权重副本／metadata，GQA 共用与 cache 预分配尚未优化。

## 可复现实验与输入

在仓库根目录执行（`python` 指向上述 CPU runtime）：

```powershell
python -m unittest discover -s "early decision/tests" -v
python "early decision/compare_attention.py" --output "early decision/measurements/2026-10-22-attention-comparison.json.gz"
python "early decision/compare_attention.py" --tensor-input trace.pt --output comparison-user.json
```

外部输入为 `torch.save` 的 tensor dict：`wq,wk:[D,d]`、`x:[N+4,D]`、`values:[N+4,d]`，d 必须为偶数；最后四个 token 是 decode steps。文件用 `weights_only=True` 加载，结果仅记录输入 label 与 SHA256，不记录输入绝对路径。单元测试实际创建并加载该格式 fixture，执行全部比较配置并检查全保留一致性。

合成 seed=`20261022`，float64，D=64、d=16，初始 N=256，再逐 token append 4 次。分别随机 QR 生成独立 Uq、Uk，令 `Wq=Uq sqrt(sigma)`、`Wk=Uk sqrt(sigma)`，因此 bilinear A 的奇异谱可控且 Wq 与 Wk 不相等。favorable 的 sigma 从 10 对数递减到 1e-5；flat 为全 1。X 与 V 为独立标准高斯；当前 causal token 包含在可见 KV 内。该数据不模拟自然语言分布，也未给随机 V 赋予语义。

矩阵权重 FP／4-bit／8-bit（每输出列 symmetric QDQ），rank=4／16，budget=32／128／260，estimate／bound 两模式，以及 no-RoPE／位置旋转 heuristic 两模式，共144配置。全部配置共享对应合成源，KV 保持浮点，公平对照在每配置固定实际 Q/K/V 与最终 FP32 softmax 再转 float64。softmax reference 同一份概率用于输出与 retained-mass；浮点概率求和可出现约 1e-7 的超 1，不做伪造截断。

每个 step 先完成 RESULT 再执行独立 dense oracle。每个 attention phase warmup 2 次，重复7次取中位数，重复不 append／不 refit；四步均保留原始时延 samples。实际 append／初建 cache 单独实测一次，构造权重副本与 QR／SVD 初始成本包含在 init_ms。单线程 CPU，Python 检查、float64 metadata、topk、allocation 和 gather 都在 RESULT 时延里。完整结果在 [JSON.gz](measurements/2026-10-22-attention-comparison.json.gz)，运行 [log](measurements/2026-10-22-attention-comparison.log)，最终24项测试记录在 [test log](measurements/2026-10-22-integration-tests.txt)。gzip 解压得到完整 JSON，逐 token cosine、索引及时间样本未删减；输出路径使用 `.json` 时可直接生成未压缩版。

新增的 [summarize_comparison.py](summarize_comparison.py)可读取普通或 gzip JSON，审计所有576个 step 的候选合法性、质量范围、TV／retained-mass 关系及全保留一致性，生成 [摘要表](measurements/2026-10-22-attention-summary.md) 和 [摘要 JSON](measurements/2026-10-22-attention-summary.json)。父 agent 在交付检查中增加了输出压缩往返测试，并独立审核上述条件；子 agent 在收尾时遇到调用额度限制，最终验证和归档由父 agent完成。

## 实测结论与适用范围

以下为 FP、rank=4、requested-k=32 的四步均值；实际 N 为257至260。logit recall 和 cosine recall 分别针对 requested-k 的两套 oracle。完整 precision／recall／F1／Jaccard、same-budget oracle、逐 token cosine 数组、selected indices、logits row cosine、NRMSE／最大误差／TV 均保留在 JSON。NRMSE 使用 reference RMS 归一化，因此与相对 L2 数值相同。两个输出向量都为零时 cosine=1、仅一方为零时为0；逐 token cosine 的零分母记0；topk 平分时采用 Torch topk 的任一合法 tie 集合。

| 谱／规则／位置 | 实际候选数 | output cosine | relative L2 | true mass | logit recall | cosine recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| favorable／estimate／none | 32 | 0.9994 | 0.0670 | 0.9435 | 0.9766 | 0.6406 |
| favorable／estimate／RoPE heuristic | 32 | 0.6550 | 1.2596 | 0.4162 | 0.2969 | 0.1875 |
| favorable／bound／none | 236.25 | 1.0000 | 0.000003 | 0.999998 | 1.0000 | 1.0000 |
| favorable／bound／RoPE heuristic | 236.25 | 0.7905 | 0.6049 | 0.7669 | 0.8125 | 0.8672 |
| flat／estimate／none | 32 | 0.6677 | 1.7584 | 0.2555 | 0.3125 | 0.3047 |
| flat／estimate／RoPE heuristic | 32 | 0.4080 | 2.0851 | 0.1737 | 0.1953 | 0.1953 |
| flat／bound／none | 258.5 | 1.0000 | 0 | 1.0000 | 1.0000 | 1.0000 |
| flat／bound／RoPE heuristic | 258.5 | 1.0000 | 0 | 1.0000 | 1.0000 | 1.0000 |

48个全保留配置的192步均通过 REF／RESULT `allclose(atol=1e-6, rtol=1e-5)`。所有 no-RoPE full-rank logits-row cosine 最小约0.9999999999999997；full rank 只说明分数恢复，budget 小于 N 时仍会丢失 softmax mass。所有 no-RoPE bound 配置的 requested-k recall=1，但 metadata norm 界偏松，往往保留大多数 KV。`score_interval_applies` 只在 no-RoPE 有效；`topk_superset_claim` 还要求 bound 模式。均为解析实数域结论加 FP64 guard，未实现 directed rounding 严格浮点认证。

上述低秩固定预算将 warm logical MAC 降至 dense QK+AV 的约0.311倍，含新增 query／append 低维投影、扫描与 selected QK+AV；但把初始 metadata 构建仅摊销到4个 decode step 时，该 MAC 比值升至约2.291（仍未将单列的 QR／SVD 时间虚构成 MAC）。CPU RESULT 仍约0.20至0.31ms，dense 对照约0.020至0.025ms，未获得 CPU 加速。bound 路径 warm MAC 约为 dense 的1.10至1.19倍，并未节省总逻辑工作。RoPE heuristic 明显损伤重要 KV 与输出质量；不能据 no-RoPE favorable 的高相似度声称真实 Llama 接入质量已验证。softmax 与逐 token cosine 的 oracle recall 差异也表明两种重要性不可替代。

## 成本账本及下一步

逻辑 warm-step MAC 为 `2D(r+tail)+Nr+2Bd`：逐 token metadata append 和 query 特征投影、低秩扫描、selected QK 与 AV。`norm` 模式 tail=0；norm、比较、softmax、验证、复制和分配属于额外非 MAC 工作，已计 CPU 时间。初始 N 个历史 token 的 metadata 投影额外为 `NDr`；结果给四步摊销量。QR／SVD 及 basis 构造以实际 init_ms 单列，不隐去其成本，也没有虚构等价 MAC；`amortized_logical_mac_excluding_factorization` 的名称明确这一边界。CPU amortized 字段包含 init 与 initial cache build，再加每步 append 和 warm attention，摊销分母为4。

两条 incremental 对比均排除公共原始 Q/K/V 投影及原始 KV cache 写入、output projection；它们已在实际组件调用前完成。RESULT 额外复制／维护 cache 的工作包含在 append 时间。字节账本分别给 factor、冻结 wq/wk clone、真实 KV、metadata、selected KV read、dense KV read、selector metadata read 和 append destination copy；cat source read 还需额外流量。runtime 每 query head 重复 KV，外部 DynamicCache 另有一份，不能把 factor_bytes 当作完整内存，也不能把 QDQ bits 当作 cache 位宽。当前 torch.cat 使 append 存在 O(N) copy，GPU／真实 GQA cache 复用尚待实现。

下一轮应在冻结真实模型权重及 trace 上重验谱、rank／预算与 retained mass，明确处理 token-dependent RoPE 后再谈认证选择，并用预分配共享 GQA cache 与设备 kernel 重新计端到端成本。现有证据支持“组件路径能独立运行、no-RoPE 解析机制和正式输出可核验”；不支持完整模型质量、真实生成吞吐或硬件加速结论。
