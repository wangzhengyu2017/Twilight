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

Follow [PADE/docs/documentation-conventions.md](PADE/docs/documentation-conventions.md),
adapted from the user's STP idea-feasibility documentation conventions and history
index. Maintain current preparation, experiment evidence, and causal round
summaries separately. Keep [PADE/docs/README.md](PADE/docs/README.md) and
[PADE/docs/history/README.md](PADE/docs/history/README.md) synchronized. Record planned,
implemented, executed, and verified scopes distinctly and freeze historical
evidence before updating the current design. A chat or file is not automatically
a new round.
