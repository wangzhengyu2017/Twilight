# STP_test 历史索引

轮次使用本分支的稳定编号；未在仓库中还原的更早研究不补写虚构时间线。每轮记录“当时问题 → 实现与执行 → 支持范围 → 修改原因 → 下一轮”。

| 轮次 | 当时方案及固定版本 | 执行依据 | 结论与后继 |
| --- | --- | --- | --- |
| ED-R001：双线性提前筛选原型 | [f69ec46 的数学说明](https://github.com/wangzhengyu2017/Twilight/blob/f69ec467ce5a81e26b1e828a2a040839c389ac7f/early%20decision/README.md) 与 [Spec](https://github.com/wangzhengyu2017/Twilight/blob/f69ec467ce5a81e26b1e828a2a040839c389ac7f/early%20decision/SPEC.md) | 同提交中的 15 项测试、60 组合成配置、18 组接口检查 | 线性假设和有利谱下可提前筛选；平坦谱界太松。下一轮扩展到真实 attention 输出对照及近邻文献核查 |
| [ED-R002：Twilight 接入与文献核查](ED-R002.md) | [对照协议](../../early%20decision/ATTENTION_PROTOCOL.md)、[功能与执行说明](../../early%20decision/TWILIGHT_INTEGRATION.md) | 24项测试、144配置／576个decode step；原始压缩结果、摘要与产物清单见总结 | 有利 no-RoPE 固定预算质量较好；CPU未加速，保守界多数全保留，RoPE启发式质量下降。下一轮处理位置与真实trace |

当前准备工作吸收 ED-R002 结果：先检验真实投影权重与 X 的结构，并解决位置依赖和缓存／kernel 成本，再判断是否继续用该 predictor。后继实验轮次尚未建立，不把以上合成组件结果当成完整模型证据。
