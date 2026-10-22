# STP_test 的项目级 Codex 配置

日期：2026-10-22。本地 STP 原先没有项目级 `.codex/`；本目录是为仓库创建的可移植配置与工作约定。

| 文件 | 作用 |
| --- | --- |
| [config.toml](config.toml) | 将项目指令读取上限设为 65536 bytes |
| [../AGENTS.md](../AGENTS.md) | 仓库级研究、开发与实验规则 |
| [documentation-conventions.md](documentation-conventions.md) | 当前文档与历史迭代的组织约定 |
| [history/README.md](history/README.md) | 可追溯的研究轮次与前后方案索引 |

Codex 可识别仓库的 AGENTS.md；项目 `.codex/` 配置的加载受客户端信任与管理配置约束。配置项与行为依据 [OpenAI 配置参考](https://learn.chatgpt.com/docs/config-file/config-reference) 和 [AGENTS.md 官方说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)，核查日期 2026-10-22。

研究入口为 [Early Decision](../early%20decision/README.md)。本目录中的历史和工作文档通过 AGENTS.md 显式关联，不声称任意 Markdown 会自动作为 Codex 配置执行。
