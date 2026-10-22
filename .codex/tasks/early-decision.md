# Early Decision 协作入口

开始时阅读 [仓库规则](../../AGENTS.md)、[文档约定](../documentation-conventions.md)、[历史索引](../history/README.md) 与 [当前对照协议](../../early%20decision/ATTENTION_PROTOCOL.md)。

当用户明确要求并行 agent 时，按以下职责分配；这是任务约定，不是客户端自动执行配置：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| 文献核查 | 公共论文名、机制关键词、已有阅读笔记；区分 W＋X／低维 QK／前层预测／索引路由 | `early decision/RELATED_WORK.md` 与来源清单；逐条原文链接、公式位置、机制相似性和缺口 |
| 实现与实验 | 固定 Spec／协议、代码基线、可用设备与 trace | Twilight opt-in adapter、对照脚本、测试、原始指标与功能变更说明 |
| 主 agent 汇总 | 两份产出及版本／执行依据 | 复核 oracle 泄漏、指标与成本，完成 ED 轮次总结、当前文档更新及精确路径提交 |

先约定文件所有权；不同 agent 不覆盖对方文件。文献事实与自推结论分开，实验 oracle 不进入运行时生成器。没有 checkpoint／GPU 时照实记录执行范围，不能用合成组件结果声称完整模型已验证。
