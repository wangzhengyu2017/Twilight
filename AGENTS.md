# Subagent Model Mapping

Builder check subagents use `gpt-6.1-sol` with `high` reasoning effort.

For implementer subagents, use the following model mappings:
- Cheap: `gpt-6-luna` with `high` reasoning effort.
- Standard: `gpt-6.1-sol` with `low` reasoning effort.
- Most capable: `gpt-6.1-sol` with `medium` reasoning effort.

For task reviewer and re-reviewer subagents, use the following model mappings:
- Cheap: `gpt-6.1-sol` with `low` reasoning effort.
- Standard: `gpt-6.1-sol` with `low` reasoning effort.
- Most capable: `gpt-6.1-sol` with `medium` reasoning effort.

Code reviewer subagents use `gpt-6.1-sol` with `high` reasoning effort.

Always dispatch task reviewers, re-reviewers, and code reviewers as fresh subagents
with `fork_turns: "none"`.

# Language Policy
English is the default language for all content within the project.

# Documentation Workflow

The upstream STP policy is pinned to [STP AGENTS.md](https://github.com/wangzhengyu2017/STP/blob/57926aee44ab066a136b7cf7d6ecf0d0db5d17d0/AGENTS.md).
Its `PADE/docs/` paths belong to the upstream project. In this branch, follow
[.codex/documentation-conventions.md](.codex/documentation-conventions.md) and keep
[.codex/history/README.md](.codex/history/README.md) and the current preparation /
experiment entry points synchronized. Maintain current preparation, experiment
evidence, and causal round summaries separately. Record planned, implemented,
executed, and verified scopes distinctly and freeze historical evidence before
updating the current design. A chat or file is not automatically a new round.

# Twilight STP_early_decision 项目规则

适用于本仓库及其子目录的研究、实现、实验与文档。开始编写文档前，阅读 [.codex/documentation-conventions.md](.codex/documentation-conventions.md)。用户新指示优先。

## 文档与历史

1. 按“准备工作 → 实验 → 本轮总结归档 → 更新下一轮准备工作”循环。
2. 准备工作和实验文档面向当前方案整合；历史方案和旧证据进入总结，不堆积在当前 Spec 中。
3. 每轮总结使用稳定编号，关联前序、当时方案／Spec、diff／提交、配置、执行记录与结果、后继方案。
4. 历史关联解释沿用、变化、修改原因、证据适用范围及重验要求；仅有链接列表不足。
5. 历史依据使用固定提交或归档副本。更正追加日期、原因与新证据，保留当时结论。
6. 代码修改按“功能｜修改前｜修改后｜涉及文件／diff｜验证结果”组织。
7. 分清规划、实现、执行、验证；局部测试、合成数据、模型质量、CPU 时间、GPU／硬件测量分别报告。
8. 归档后更新 [.codex/history/README.md](.codex/history/README.md) 与前后链接。未知历史标为待核实，不从文件日期推断轮次。
9. 技能、子 agent 与交接均继承这些规则；需要委派时说明文件所有权，避免覆盖其他人的工作。

## 当前 Early Decision 分支

- 当前对照路径：W＋X → Early Decision B0 → 只在 B0 内计算 attention。该路径绕过 DS selector 与 Twilight Top-p；其他旧路径按原配置工作。
- 数学原型见 [early decision/README.md](early%20decision/README.md)，比较口径见 [ATTENTION_PROTOCOL.md](early%20decision/ATTENTION_PROTOCOL.md)。新实验不得用完整 QK 的 oracle 帮助生成 B0。
- 明确 no-RoPE、共享变换、逐 token RoPE 的差异。使用 pre-RoPE 预测、post-RoPE 执行时标为近似，不沿用原线性区间保证。
- QK softmax logits 与逐 token cosine(q,k) 分别定义重要性。输出余弦、相对误差、重要 KV precision／recall／F1／Jaccard 与 retained mass，不用多数负样本造成的高 accuracy 掩盖漏选。
- 效率包含初始化／SVD、cache append、额外特征投影、选择、选中 QK 与 AV、元数据；给出 warm-step 和摊销口径。逻辑 MAC／字节不等于实测硬件加速。

## 开发与验证

- 检索优先使用 `rg`。修改前检查 Git 状态，保留现有用户改动；只提交本轮涉及路径。
- CPU 数值检查：`python -m unittest discover -s "early decision/tests" -v`。
- 独立原型：`python "early decision/benchmark.py" --output "early decision/measurements/local-run.json"`。
- 文献与代码说明链接到直接来源，标明原文事实、推导、假设与未验证项。不得虚构论文或实验结果。
- 文本使用 UTF-8／LF；结果与说明记录版本、输入、seed、shape、dtype、设备及命令。公开文件使用仓库相对路径。
