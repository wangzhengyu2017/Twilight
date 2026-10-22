# ED-R002 实测摘要

由完整记录自动生成；固定 FP、rank=4、requested-k=32，四个 decode step 的均值。逻辑 MAC 比值不是实测加速。

| 谱／模式／RoPE | 候选数 | 输出cos | 相对L2 | 真实mass | logits召回 | token-cos召回 | warm MAC比 | 含4步初始化摊销MAC比* | CPU ms / dense ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| favorable/estimate_topk/none | 32.00 | 0.9994 | 0.0670 | 0.9435 | 0.9766 | 0.6406 | 0.311 | 2.291 | 0.3099/0.0250 |
| favorable/estimate_topk/pre_rope_heuristic | 32.00 | 0.6550 | 1.2596 | 0.4162 | 0.2969 | 0.1875 | 0.311 | 2.291 | 0.2029/0.0244 |
| favorable/topk_candidates/none | 236.25 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.101 | 3.081 | 0.2486/0.0202 |
| favorable/topk_candidates/pre_rope_heuristic | 236.25 | 0.7905 | 0.6049 | 0.7669 | 0.8125 | 0.8672 | 1.101 | 3.081 | 0.2490/0.0201 |
| flat/estimate_topk/none | 32.00 | 0.6677 | 1.7584 | 0.2555 | 0.3125 | 0.3047 | 0.311 | 2.291 | 0.2230/0.0199 |
| flat/estimate_topk/pre_rope_heuristic | 32.00 | 0.4080 | 2.0851 | 0.1737 | 0.1953 | 0.1953 | 0.311 | 2.291 | 0.2134/0.0198 |
| flat/topk_candidates/none | 258.50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.187 | 3.168 | 0.2904/0.0199 |
| flat/topk_candidates/pre_rope_heuristic | 258.50 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.187 | 3.168 | 0.2638/0.0199 |

* 摊销 MAC 含初始 metadata 构建；QR/SVD 用实际初始化时间另列，没有虚构其等价 MAC。实际 CPU 的摊销时间字段包含这些初始化时间。

共检查 144 配置、576 steps；192 个全保留 step 与 dense allclose；144 个 no-RoPE 保守 step 的 requested-k recall=1。

本摘要不支持完整模型质量或 GPU 加速结论；逐 token RoPE 使用 pre-RoPE 筛选启发式，线性区间保证不适用。
