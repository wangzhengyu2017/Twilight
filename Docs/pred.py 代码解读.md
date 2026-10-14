# pred.py 代码解读

> [!TIP]
>
> CheckList
>
> - [x] 是否有 TL;DR
> - [x] 是否有整体流程
> - [x] 是否能清晰拆分步骤
> - [x] 每一步是否能加入代码块和注释
> - [x] 是否有“重要”的语法 / 模型推理过程

## TL;DR

> [!IMPORTANT]
>
> `benchmark/LongBench/pred.py` 是 Twilight 的 LongBench **逐任务推理与结果落盘脚本**。
>
> 1. 它读取模型、LongBench 数据和稀疏算法配置，并通过 `enable_sparse_attention()` 把 Llama/Mistral Attention 的 `forward` 替换成 Twilight 实现。
> 2. 每条样本先套用数据集 prompt 模板；超过模型输入上限时，保留 token 序列的前半段和后半段。
> 3. 普通任务不直接调用 `model.generate()`，而是手写 `context prefill → question token-by-token decode → answer greedy decode`，以便让问题阶段也经过单 token Decode 路径。
> 4. 每次模型调用的核心张量是 `input_ids: [1, seq_len]`；逐 token Decode 时变成 `[1, 1]`，输出 logits 典型形状为 `[1, 1, vocab_size]`。
> 5. 每条样本结束后记录预测文本、参考答案、最终稀疏 budget 和保留的真实 Attention 概率质量，再清空统计器。
> 6. 当前源码存在明确的执行风险：`samsum` 分支没有定义后续使用的 `generated_content`；ChatGLM3 路径的数据类型和 `q_input` 也与后续公共逻辑不一致。

源码位置：[benchmark/LongBench/pred.py #1-373](./benchmark/LongBench/pred.py)

## 解读范围与证据边界

本文以当前仓库中的 `benchmark/LongBench/pred.py` 为主，只有在解释下列两类接口时读取直接依赖：

| 文件 | 本文使用它确认的内容 |
| --- | --- |
| `benchmark/LongBench/pred.py` | 推理主流程、函数、参数、模型调用、输出格式和当前风险 |
| `benchmark/LongBench/config/*.json` | 模型最大长度、模型路径、数据集 prompt 和最大生成长度 |
| `twilight/pyimpl/attention.py` | `enable_sparse_attention()` 返回值及 monkey patch 对象 |
| `twilight/pyimpl/state.py` | `budget` 与 `score_sum` 两个输出字段的真实统计口径 |
| `twilight/pyimpl/__init__.py` | `enable_sparse_attention` 的公共导出入口 |

文中分三种表述：

- **源码事实**：当前文件或直接依赖明确执行的行为；
- **代码意图**：来自注释、命名或流程安排，但当前实现可能没有完整实现；
- **外部行为**：依赖具体 Transformers、Tokenizer、模型远程代码版本的行为，本文不会把它写成已由当前仓库证明的事实。

## 整体流程

```text
命令行启动 pred.py
        │
        ├── seed_everything(42)
        ├── parse_args()
        ├── 读取 model2maxlen.json / model2path.json
        │
        ▼
load_model_and_tokenizer(...)
        ├── AutoTokenizer.from_pretrained(...)
        ├── AutoModelForCausalLM.from_pretrained(...)
        ├── model.eval()
        ├── 读取稀疏算法 JSON
        └── enable_sparse_attention(...)
                └── 替换 Llama/Mistral Attention.forward
        │
        ▼
读取 dataset2prompt.json / dataset2maxlen.json
        │
        ▼
load_dataset(...)
        ├── 本地路径不存在：THUDM/LongBench
        └── 本地路径存在：本地 JSONL
        │
        ▼
get_pred(...)
        │
        └── 对每一条样本
                ├── prompt_format.format(**json_obj)
                ├── tokenize；过长则保留首尾
                ├── 可选 build_chat(...)
                ├── 拆成 context prompt 与 question
                ├── context 一次 Prefill
                ├── question 逐 token Decode
                ├── answer 逐 token 贪心 Decode
                ├── tokenizer.decode(...) + post_process(...)
                ├── 读取并 reset budget/score 统计器
                └── 保存一条预测字典
        │
        ▼
构造输出文件名
        │
        ▼
逐条写入 JSONL
```

### 函数调用关系

```text
__main__
├── seed_everything(42)
├── parse_args()
├── load_model_and_tokenizer(...)
│   ├── AutoTokenizer.from_pretrained(...)
│   ├── AutoModelForCausalLM.from_pretrained(...)
│   └── enable_sparse_attention(...)
└── get_pred(...)
    ├── build_chat(...)
    ├── model(...) / model.generate(...)
    ├── post_process(...)
    ├── budget_info.get_total_avg_budget()
    └── score_info.get_total_avg_score()
```

---

# 导入与脚本依赖

## 1. Python、PyTorch 和 Hugging Face 依赖（#1-19）

```python
# Modified from: Quest 的 LongBench 推理脚本

import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from twilight.pyimpl import enable_sparse_attention
```

依赖职责：

| 名称 | 当前文件中的作用 |
| --- | --- |
| `os` | 判断数据集路径是否存在、创建输出目录 |
| `load_dataset` | 从 Hugging Face Hub 或本地 JSONL 加载 LongBench |
| `torch` | CUDA、随机种子、模型推理和 Tensor 操作 |
| `json` | 读取配置、写 JSONL |
| `AutoTokenizer` | 根据模型路径自动选择 tokenizer |
| `AutoModelForCausalLM` | 根据模型路径自动选择因果语言模型类 |
| `tqdm` | 展示数据集逐样本推理进度 |
| `numpy`、`random` | 设置随机种子 |
| `argparse` | 解析命令行参数 |
| `enable_sparse_attention` | 给模型 Attention 层安装 Twilight 稀疏实现 |

当前被导入但没有使用：

```python
AutoConfig
LlamaTokenizer
LlamaForCausalLM
```

这些未使用导入不改变脚本行为，只增加了阅读噪声。

---

# parse_args

## 2. 函数功能、参数与返回值（#22-59）

```python
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    ...
    return parser.parse_args(args)
```

作用：定义并解析 `pred.py` 的命令行参数。

| 参数 | 类型/默认值 | 是否显式必填 | 作用 |
| --- | --- | --- | --- |
| `--model` | `str` / `None` | 否 | 模型逻辑名称，只能取 `choices` 中的值 |
| `--model_path` | `str` / `None` | 否 | 覆盖 `model2path.json` 中的模型路径 |
| `--e` | `int` / `0` | 否 | 非 0 时运行 LongBench-E 路径 |
| `--task` | `str` | 是 | 本进程只运行的一个数据集名称 |
| `--output-dir` | `str` / `None` | 否 | 输出根目录 |
| `--dataset-path` | `str` / `None` | 否 | 本地 LongBench JSONL 所在目录 |
| `--config-path` | `str` / `None` | 否 | LongBench 四个配置 JSON 所在目录 |
| `--t` | `str` / `"benchmark launch time"` | 否 | 拼入输出文件名的实验标识 |
| `--algo-config-path` | `str` / `None` | 否 | Twilight 稀疏算法 JSON 路径 |

支持的 `--model` 标识共有 10 个：

```python
choices=[
    "llama2-7b-chat-4k",
    "longchat-v1.5-7b-32k",
    "xgen-7b-8k",
    "internlm-7b-8k",
    "chatglm2-6b",
    "chatglm2-6b-32k",
    "chatglm3-6b-32k",
    "vicuna-v1.5-7b-16k",
    "Mistral-7B-Instruct-v0.3",
    "Meta-Llama-3.1-8B-Instruct",
]
```

调用方式有两种：

```python
# 1. 正常命令行调用：读取 sys.argv[1:]
args = parse_args()

# 2. 测试或其他 Python 代码直接传参数列表
args = parse_args([
    "--model", "Meta-Llama-3.1-8B-Instruct",
    "--task", "qasper",
])
```

### 重要：`argparse` 的 `choices`、`required` 和默认值

```python
parser.add_argument(
    "--model",
    type=str,
    default=None,
    choices=[...],
)
```

- `type=str`：把命令行文本作为字符串；
- `choices=[...]`：传入不在列表中的模型名时，参数解析阶段直接报错；
- `default=None`：没有传 `--model` 时结果仍然可以是 `None`；
- `required=True`：只有 `--task` 设置了它，因此其他实际上必需的路径参数不会在解析阶段被检查。

也就是说，“没有写 `required=True`”不代表后续真的可以省略。例如：

```python
has_local_data_set = os.path.exists(args.dataset_path)
```

如果 `--dataset-path` 没有传入，`args.dataset_path` 为 `None`，这里可能在进入推理前就报类型错误。

---

# build_chat

## 3. 给不同模型构造聊天输入（#62-87）

```python
def build_chat(tokenizer, prompt, model_name):
    ...
    return prompt
```

参数与返回值：

| 名称 | 典型类型 | 作用 |
| --- | --- | --- |
| `tokenizer` | Transformers tokenizer | 部分模型使用 tokenizer 自带 chat 构造方法 |
| `prompt` | 通常是 `str` | 已由 LongBench 数据集模板生成的任务提示 |
| `model_name` | `str` | 通过子串判断选择哪个聊天模板 |
| 返回值 | 通常期望为 `str` | 包装后的模型输入；ChatGLM3 分支可能不是字符串 |

### 3.1 ChatGLM3 与 ChatGLM（#64-67）

```python
if "chatglm3" in model_name:
    # 使用 tokenizer 的 ChatGLM3 专用构造接口
    prompt = tokenizer.build_chat_input(prompt)
elif "chatglm" in model_name:
    # ChatGLM2 等模型使用另一套 prompt 构造接口
    prompt = tokenizer.build_prompt(prompt)
```

分支顺序很重要：`"chatglm3"` 同时也包含 `"chatglm"`，所以必须先判断更具体的 ChatGLM3。

### 3.2 LongChat 与 Vicuna（#68-74）

```python
elif "longchat" in model_name or "vicuna" in model_name:
    from fastchat.model import get_conversation_template

    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], prompt)  # user 消息
    conv.append_message(conv.roles[1], None)    # assistant 等待生成
    prompt = conv.get_prompt()
```

逻辑结构：

```text
原始 LongBench prompt
        ↓
放入 Vicuna conversation 的 user role
        ↓
追加一个内容为 None 的 assistant role
        ↓
conv.get_prompt() 序列化为最终字符串
```

`fastchat` 是在这个分支内部导入的，因此不运行 LongChat/Vicuna 时，不会因为该行主动加载 FastChat。

### 3.3 Llama2、XGen、InternLM 与 Llama 3.1（#75-86）

```python
elif "llama2" in model_name:
    prompt = f"[INST]{prompt}[/INST]"

elif "xgen" in model_name:
    header = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )
    prompt = header + f" ### Human: {prompt}\n###"

elif "internlm" in model_name:
    prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"

elif "Llama-3.1" in model_name:
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|> "
        f"{prompt} <|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )
```

未匹配任何分支时原样返回：

```python
return prompt
```

### 重要：Chat template 不是普通的自然语言前后缀

聊天模型通常通过特殊 token 或固定角色文本区分：

```text
system / user / assistant
```

因此同一条任务 prompt 给不同模型时，最终输入文本并不相同。这里的 `build_chat()` 是模型适配层，而不是 LongBench 数据集模板本身：

```text
dataset2prompt.json       build_chat()
任务说明 + context + 问题  →  模型家族要求的 user/assistant 格式
```

当前实现是手写模型名分支，并没有统一调用 `tokenizer.apply_chat_template()`。这些模板是否与模型当前发布版本完全一致，需要目标 tokenizer/模型版本验证。

---

# post_process

## 4. 清理生成文本（#90-95）

```python
def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response
```

分支含义：

| 模型 | 处理 |
| --- | --- |
| XGen | 去掉首尾空白，并删除所有字面值 `Assistant:` |
| InternLM | 只保留第一次出现 `<eoa>` 之前的内容 |
| 其他模型 | 原样返回 |

小例子：

```python
post_process("  Assistant: Paris  ", "xgen-7b-8k")
# " Paris"，replace 后可能仍保留内部原有空格

post_process("答案<eoa>后续文本", "internlm-7b-8k")
# "答案"
```

注意：XGen 分支先 `strip()`，再删除 `Assistant:`，删除标签后不会再次 `strip()`。

---

# get_pred

## 5. 函数入口、参数和返回值（#98-110、#257）

```python
def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    budget_info,
    score_info,
):
    ...
    return preds
```

| 参数 | 作用 |
| --- | --- |
| `model` | 已加载、已 `eval()`、已安装 Twilight Attention 的因果语言模型 |
| `tokenizer` | 与模型配套的 tokenizer |
| `data` | Hugging Face Dataset 或其他可迭代样本集合 |
| `max_length` | prompt 截断目标长度，来自 `model2maxlen.json` |
| `max_gen` | 最多生成 token 数，来自 `dataset2maxlen.json` |
| `prompt_format` | 带 `{context}`、`{input}` 等占位符的数据集模板 |
| `dataset` | 当前 LongBench 任务名 |
| `device` | 设计上表示推理设备，但函数内部没有使用它 |
| `model_name` | 选择 chat 模板和输出后处理规则 |
| `budget_info` | Twilight 最终选中 token 数量统计器 |
| `score_info` | Twilight 被选 token 覆盖的真实 Attention 概率统计器 |

返回 `preds`，类型是“字典列表”：

```python
[
    {
        "pred": "模型答案",
        "answers": ["参考答案1", "参考答案2"],
        "all_classes": [...],
        "length": 12345,
        "budget": 2048.5,
        "score_sum": 0.91,
    },
    ...,
]
```

## 6. 创建结果列表并逐样本推理（#111-118）

```python
preds = []

for json_obj in tqdm(data):
    torch.cuda.empty_cache()
    prompt = prompt_format.format(**json_obj)
```

每轮含义：

1. `tqdm(data)` 包装迭代器，显示完成数量和速度；
2. `torch.cuda.empty_cache()` 释放 PyTorch CUDA allocator 中当前未被 Tensor 占用的缓存块；
3. `prompt_format.format(**json_obj)` 用样本字段填充模板。

### 重要：`**json_obj` 是字典关键字解包

假设：

```python
prompt_format = "文章：{context}\n问题：{input}"
json_obj = {
    "context": "太阳系资料",
    "input": "最大的行星是什么？",
    "answers": ["木星"],
}
```

这行：

```python
prompt_format.format(**json_obj)
```

等价于概念上的：

```python
prompt_format.format(
    context=json_obj["context"],
    input=json_obj["input"],
    answers=json_obj["answers"],
)
```

模板只读取自己出现的占位符，多余字段不会进入字符串。若模板要求的字段在样本中不存在，则会抛出 `KeyError`。

## 7. Tokenize 与“保留首尾”的截断（#119-131）

### 7.1 第一次 tokenize（#120-126）

```python
tokenized_prompt = tokenizer(
    prompt,
    truncation=False,
    return_tensors="pt",
).input_ids[0]

if "chatglm3" in model_name:
    tokenized_prompt = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]
```

一般形状变化：

```text
tokenizer(...).input_ids : [batch=1, prompt_token_num]
.input_ids[0]            : [prompt_token_num]
```

`truncation=False` 表示 tokenizer 自己不截断。ChatGLM3 会重做一次 tokenize，并关闭自动特殊 token。

### 7.2 超长时保留前后两半（#127-131）

```python
if len(tokenized_prompt) > max_length:
    half = int(max_length / 2)
    prompt = (
        tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
        + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    )
```

假设：

```text
max_length = 8
原 token IDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
half = 4

保留开头：[0, 1, 2, 3]
保留结尾：[8, 9, 10, 11]
删除中间：[4, 5, 6, 7]
```

设计意图是保住开头任务说明与结尾问题，而不是只从一侧截断。

需要注意，这里把 token IDs `decode` 回字符串，后面还会再次 tokenize；同时聊天模板在截断之后才添加。因此，后续真正输入模型的 token 数不保证严格等于或小于这里的 `max_length`。

## 8. 决定是否套聊天模板（#132-140）

```python
if dataset not in [
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
]:
    prompt = build_chat(tokenizer, prompt, model_name)
```

源码注释表达的意图是：这些任务不套 chat prompt 时效果更好。

```text
trec / lsht             分类
triviaqa                问答
samsum                  对话摘要
lcc / repobench-p       代码补全
```

这是按**数据集**做的统一跳过：只要任务在列表中，无论具体模型是什么，都不会进入 `build_chat()`。

## 9. 把 prompt 拆成 context 与 question（#142-159）

### 9.1 按数据集寻找分隔文本（#142-152）

```python
if dataset in ["qasper", "hotpotqa"]:
    q_pos = prompt.rfind("Question:")
elif dataset in ["multifieldqa_en", "gov_report"]:
    q_pos = prompt.rfind("Now,")
elif dataset in ["triviaqa"]:
    q_pos = prompt.rfind("Answer the question")
elif dataset in ["narrativeqa"]:
    q_pos = prompt.rfind("Do not provide")
else:
    q_pos = -1
```

`str.rfind(x)` 返回子串 `x` 最后一次出现的**字符下标**，找不到时返回 `-1`。

### 9.2 把 question 限制为末尾约 100 个字符（#154-159）

```python
q_pos = max(len(prompt) - 100, q_pos)

if q_pos != None:
    question = prompt[q_pos:]
    prompt = prompt[:q_pos]
```

关键点：这里的 `100` 是 Python 字符串字符数，不是 token 数。

例 1：标记离末尾只有 60 个字符。

```text
q_pos = 标记位置
len(prompt) - 100 更靠前
max(...) = 标记位置
question = 从标记到末尾
```

例 2：标记后的内容有 300 个字符。

```text
q_pos = 标记位置
len(prompt) - 100 更靠后
max(...) = len(prompt) - 100
question = 只取末尾 100 个字符
```

例 3：任务没有专用标记，且 prompt 长度大于 100。

```text
初始 q_pos = -1
最终 q_pos = len(prompt) - 100
question = 末尾 100 个字符
```

### 重要：`q_pos != None` 当前恒为 True

`q_pos` 在前面的每条分支里都会被赋为整数：

```python
# 找到子串：0 或正整数
# 没找到：-1
q_pos = prompt.rfind(...)
```

所以：

```python
if q_pos != None:
```

不会表达“是否找到了分隔符”，它对 `-1` 也成立。对少于 100 字符且没找到标记的 prompt，最终通常会使用 `q_pos = -1`：

```python
question = prompt[-1:]  # 最后一个字符
prompt = prompt[:-1]    # 前面所有字符
```

从代码行为看，context/question 拆分是必做步骤，而不是可选步骤。

## 10. 再次 tokenize，并移动到 CUDA（#161-172）

```python
if "chatglm3" in model_name:
    input = prompt.to("cuda")
else:
    input = tokenizer(
        prompt,
        truncation=False,
        return_tensors="pt",
    ).to("cuda")

    q_input = tokenizer(
        question,
        truncation=False,
        return_tensors="pt",
    ).to("cuda")

    q_input.input_ids = q_input.input_ids[:, 1:]

context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]
```

普通模型的典型形状：

```text
input.input_ids          [1, context_token_num]
q_input.input_ids        [1, question_token_num_with_BOS]
q_input.input_ids[:, 1:] [1, question_token_num]
```

去掉 `q_input` 的第一个 token，意图是避免分别 tokenize context 和 question 时重复加入开头特殊 token。

`context_length` 是两段 token 数之和：

```text
context_length = context token 数 + 去掉首 token 后的 question token 数
```

但在普通手写 Decode 分支中，变量 `context_length` 没有参与模型调用，只在 `samsum` 的 `min_length` 中使用。

### 重要：Tokenizer 返回的 BatchEncoding 与 `.to("cuda")`

普通路径中的 `input` 不是单个 Tensor，而是类似字典的 tokenizer 输出对象：

```python
input = tokenizer(..., return_tensors="pt")

# 概念结构
{
    "input_ids": Tensor[1, seq_len],
    "attention_mask": Tensor[1, seq_len],
}
```

调用：

```python
input.to("cuda")
```

会把对象内部的 Tensor 移动到 CUDA。当前手写模型调用只显式传 `input.input_ids`，没有传 `attention_mask`。

源码虽然计算了：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

但本函数把设备硬编码为 `"cuda"`，因此 `device` 参数没有产生作用，CPU fallback 也没有真正贯通。

## 11. `samsum` 的 `model.generate()` 特例（#174-189）

```python
if dataset == "samsum":
    output = model.generate(
        **input,
        max_new_tokens=max_gen,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        min_length=context_length + 1,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.encode("\n", add_special_tokens=False)[-1],
        ],
    )[0]
```

参数含义：

| 参数 | 当前设置 | 含义 |
| --- | --- | --- |
| `**input` | context 的 tokenizer 输出 | 把 `input_ids`、`attention_mask` 等按关键字传给模型 |
| `max_new_tokens` | `max_gen` | 限制新生成 token 数 |
| `num_beams` | `1` | 不扩展多条 beam |
| `do_sample` | `False` | 不做随机采样 |
| `temperature` | `1.0` | 在关闭采样的当前设置下不负责随机抽样 |
| `min_length` | `context_length + 1` | 在达到总序列长度下限前不允许正常结束 |
| `eos_token_id` | 模型 EOS + 换行 token | 遇到二者之一即可作为终止符 |

源码注释说明，加入换行终止符是为了避免模型不断重复 `"\nDialogue"`。

### 重要：`**input` 是模型关键字参数解包

如果 tokenizer 输出近似为：

```python
input = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
}
```

那么：

```python
model.generate(**input)
```

概念上等价于：

```python
model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
)
```

### 当前 `samsum` 分支的断链

这一分支只赋值：

```python
output = ...
```

但公共后处理读取的是：

```python
pred = tokenizer.decode(generated_content, ...)
```

`generated_content` 只在下面的非 `samsum` 分支创建。因此按当前源码执行 `samsum` 时，会在解码处遇到未定义局部变量。

此外，前面已经拆出的 `q_input` 没有合并进 `model.generate(**input)`；而 `min_length` 却使用了 context 与 question 两者的 token 总数。这里的输入和长度口径不一致。

## 12. 普通任务：context Prefill（#190-197）

```python
else:
    with torch.no_grad():
        output = model(
            input_ids=input.input_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = output.past_key_values
```

假设：

```text
batch size = 1
context token 数 = C
词表大小 = V
层数 = L
```

典型输入/输出形状：

```text
input.input_ids       [1, C]
output.logits         [1, C, V]
output.past_key_values 每层保存截至这 C 个 token 的 K/V
```

这次一次性输入多个 token，通常对应 Twilight `attention_forward()` 中的 Prefill 路径。

参数含义：

- `past_key_values=None`：这是当前样本第一次模型调用，没有传入历史 Cache；
- `use_cache=True`：要求模型返回更新后的 KV Cache；
- `torch.no_grad()`：关闭这段代码中的梯度记录，减少推理所需显存和 autograd 开销。

## 13. Question token-by-token Decode（#198-205）

```python
for input_id in q_input.input_ids[0]:
    output = model(
        input_ids=input_id.unsqueeze(0).unsqueeze(0),
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = output.past_key_values
```

### 13.1 `unsqueeze()` 的形状变化

循环中单个 `input_id` 是 0 维 Tensor：

```text
input_id                         []
input_id.unsqueeze(0)            [1]
input_id.unsqueeze(0).unsqueeze(0) [1, 1]
```

模型要求 `input_ids` 具有 `[batch, seq_len]` 两个维度，所以需要连续增加两个维度。

### 13.2 为什么逐 token 输入 question

源码注释写的是：

```python
# split the prompt and question (simulate decoding in the question stage)
```

其执行效果是：

```text
context：一次输入 C 个 token → Prefill
question：每次输入 1 个 token → Decode
```

由于 Twilight 的 `attention_forward()` 用 `q_len > 1` 与 `q_len == 1` 区分 Prefill/Decode，所以拆开 question 可以让问题尾部的一段 token 也进入稀疏 Decode 及其统计路径。

## 14. 生成第一个回答 token（#206-207）

```python
pred_token_idx = (
    output.logits[:, -1, :]
    .argmax(dim=-1)
    .unsqueeze(1)
)
generated_content = [pred_token_idx.item()]
```

形状变化：

```text
output.logits             [1, current_seq_len, vocab_size]
output.logits[:, -1, :]   [1, vocab_size]
argmax(dim=-1)            [1]
unsqueeze(1)              [1, 1]
```

数学上：

$$
t_1=\arg\max_{v\in\mathcal{V}}\operatorname{logits}(v)
$$

这不是 Top-p/Top-k 生成采样，而是确定性的 greedy decoding：选择 logit 最大的词表 token。

`generated_content` 保存 Python 整数 token ID：

```python
[first_token_id]
```

## 15. 继续逐 token 贪心生成（#208-221）

```python
for _ in range(max_gen - 1):
    outputs = model(
        input_ids=pred_token_idx,
        past_key_values=past_key_values,
        use_cache=True,
    )

    past_key_values = outputs.past_key_values
    pred_token_idx = (
        outputs.logits[:, -1, :]
        .argmax(dim=-1)
        .unsqueeze(1)
    )
    generated_content += [pred_token_idx.item()]

    if pred_token_idx.item() == tokenizer.eos_token_id:
        break
```

每轮状态转移：

```text
上一轮选出的 token [1, 1]
        +
此前所有 token 的 KV Cache
        ↓
model(..., use_cache=True)
        ├── 新的 past_key_values
        └── 下一个位置的 logits [1, 1, vocab_size]
        ↓
argmax
        ↓
下一个 token [1, 1]
```

### 重要：KV Cache 避免每一步重算全部历史 token

令完整上下文为：

```text
[c1, c2, c3, q1, q2]
```

模型调用序列是：

```text
1. 输入 [c1, c2, c3]，建立 context Cache
2. 输入 [q1] + Cache，Cache 更新到 q1
3. 输入 [q2] + Cache，Cache 更新到 q2，并预测 a1
4. 输入 [a1] + Cache，Cache 更新到 a1，并预测 a2
5. 输入 [a2] + Cache，继续预测
```

因此每个生成步只显式输入一个新 token；历史 K/V 来自 `past_key_values`。

这里的 `past_key_values` 不是 `hidden_states`：

- `hidden_states` 是某一层当前 token 的特征；
- `past_key_values` 是各层历史 token 的 Key/Value 缓存；
- Twilight `LocalState` 则是共享稀疏配置与统计对象，也不是 KV Cache。

### 重要：`max_gen` 与 EOS 的实际边界

代码先无条件生成第一个 token，再循环 `max_gen - 1` 次。因此在配置中的 `max_gen >= 1` 时，最多生成 `max_gen` 个 token。

但有两个边界：

1. 第一个 token 生成后没有立刻检查 EOS；如果第一个 token 就是 EOS 且 `max_gen > 1`，代码仍会再执行一次模型调用，之后才检查新 token。
2. 如果外部传入 `max_gen=0`，当前代码仍会先创建一个 token，而不是生成 0 个。仓库自带 `dataset2maxlen.json` 中的值都大于 0，所以主入口不会触发第二种情况。

## 16. Decode 时间线与统计覆盖范围

假设：

```text
context = C 个 token
question = Q 个 token
最终生成 A 个 token
```

普通路径模型调用大致为：

```text
1 次 context Prefill
+ Q 次 question 单 token Decode
+ (A - 1) 次回答 token 单 token Decode
```

为什么生成 A 个 token 只额外调用约 `A - 1` 次？因为最后一个 question token 的输出 logits 已经用来选择第一个回答 token。

Twilight 的 budget/score 统计主要在 `q_len == 1` 的 Decode 路径更新。因此当前样本的统计不只覆盖“回答生成”，还覆盖人为拆出的 question 单 token 阶段。

## 17. 解码 token IDs 并做模型专用清理（#231-233）

```python
pred = tokenizer.decode(
    generated_content,
    skip_special_tokens=True,
)
pred = post_process(pred, model_name)
```

数据变化：

```text
generated_content
[token_id_1, token_id_2, ...]
        ↓ tokenizer.decode
字符串
        ↓ post_process
最终预测字符串 pred
```

`skip_special_tokens=True` 会要求 tokenizer 在解码时忽略它识别出的特殊 token，包括通常的 EOS；具体特殊 token 集合由 tokenizer 决定。

源码还保留了原 `model.generate()` 解码方式的注释：

```python
# pred = tokenizer.decode(
#     output[context_length:],
#     skip_special_tokens=True,
# )
```

当前真正执行的是手写 greedy decode 的 `generated_content` 路径。

## 18. 读取并重置 Twilight 统计（#235-242）

```python
avg_budget = budget_info.get_total_avg_budget()
avg_score = score_info.get_total_avg_score()

budget_info.reset()
score_info.reset()
```

顺序必须是：

```text
先读取当前样本统计
        ↓
再 reset
        ↓
下一条样本从空统计开始
```

### 18.1 `budget` 的准确含义

Attention 层中最终 mask 的典型形状是：

```text
[bs, num_heads, 1, kv_seq_len]
```

对最后一维求和得到每个 head 最终保留的 KV token 数：

$$
B_{q,l,h}=\sum_{k=1}^{K}\mathbf{1}[\operatorname{mask}_{q,l,h,k}=\text{True}]
$$

`get_total_avg_budget()` 的当前口径可概括为：

```text
每个 query
  → 每层先对 head 求平均
  → 只纳入 layer_id > 1 的层
  → 对层求平均
再对全部被记录 query 求平均
```

所以输出字段 `budget` 不是总 token 数，而是跨 query、层和 head 汇总后的**平均最终保留 token 数**。

### 18.2 `score_sum` 的准确含义

统计器先对未剪枝 Attention logits 做 FP32 Softmax：

$$
P=\operatorname{softmax}(S)
$$

然后把最终 mask 外的位置设为 0，并求被保留位置的概率和：

$$
R_{q,l,h}=\sum_k P_{q,l,h,k}\cdot
\mathbf{1}[\operatorname{mask}_{q,l,h,k}=\text{True}]
$$

接着当前实现：

```text
每层取所有 head 中最小的 R
  → 每个 query 只平均 layer_id > 1 的层
  → 最后平均所有 query
```

因此：

```python
"score_sum": avg_score
```

字段名看起来像“总和”，实际保存的是上述多级聚合后的**平均最差 head 保留概率质量**，通常在 0 到 1 附近，而不是原始 logits 总和。

### 重要：Selector → Estimator → Pruner 与这里的统计位置

在 Twilight Decode 中，目标维度是 KV token：

```text
Selector
先给每个 query/head 选候选 KV token，得到候选 mask（B0）
        ↓
Weight Estimator
估算候选 token 的 Attention score
        ↓
Weight Pruner
继续删除低价值候选，得到最终 mask（B1）
        ↓
budget_info
统计 B1 中 True 的 token 数
        ↓
score_info
用真实完整 Attention 概率衡量 B1 覆盖了多少概率质量
```

`pred.py` 只读取聚合结果，不在本文件中执行 Selector、Estimator 或 Pruner 的具体算法。

## 19. 保存一条预测记录（#245-256）

```python
preds.append(
    {
        "pred": pred,
        "answers": json_obj["answers"],
        "all_classes": json_obj["all_classes"],
        "length": json_obj["length"],
        "budget": avg_budget,
        "score_sum": avg_score,
    }
)
```

字段对应关系：

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| `pred` | 模型生成并清理后的字符串 | 下游 `eval.py` 的预测文本 |
| `answers` | LongBench 样本 | 一个或多个参考答案 |
| `all_classes` | LongBench 样本 | 分类任务的类别集合 |
| `length` | LongBench 样本 | 原样本长度，用于 LongBench-E 分桶 |
| `budget` | Twilight 统计器 | 最终稀疏 mask 的平均 token budget |
| `score_sum` | Twilight 统计器 | 最终 mask 的平均最差 head 概率覆盖率 |

注释中的 B0/B1 输出当前没有执行：

```python
# "B0": avg_B0,
# "B1": avg_budget,
```

最终函数返回整个数据集的预测列表：

```python
return preds
```

这意味着所有样本先保存在内存里，等整个任务推理完成后，主入口才写文件。

---

# seed_everything

## 20. 设置随机种子与 cuDNN 选项（#260-267）

```python
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
```

覆盖的随机源：

| 代码 | 目标 |
| --- | --- |
| `torch.manual_seed(seed)` | PyTorch CPU，且按 PyTorch 接口语义也影响 CUDA 默认生成器 |
| `torch.cuda.manual_seed(seed)` | 当前 CUDA 设备 |
| `np.random.seed(seed)` | NumPy |
| `random.seed(seed)` | Python 标准库 `random` |
| `torch.cuda.manual_seed_all(seed)` | 所有 CUDA 设备 |

cuDNN 设置：

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

- 关闭 benchmark，避免根据运行时输入自动选择可能变化的实现；
- 请求 cuDNN 优先使用确定性算法。

当前普通生成路径使用 `argmax`，本身不随机；设置种子仍可约束模型远程代码或其他依赖中的随机行为。但仅设置这些选项不等价于保证所有 GPU kernel、第三方 CUDA 扩展和所有模型实现都完全确定。

---

# load_model_and_tokenizer

## 21. 加载 tokenizer 和模型（#270-275）

```python
def load_model_and_tokenizer(path, model_name, device, args):
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = model.eval()
```

参数：

| 参数 | 当前用途 |
| --- | --- |
| `path` | Hugging Face 模型 ID 或本地模型目录 |
| `model_name` | 传入但本函数内部没有使用 |
| `device` | 传入但本函数内部没有使用 |
| `args` | 读取 `args.algo_config_path` |

模型加载选项：

- `torch_dtype=torch.bfloat16`：按 BF16 加载模型权重；
- `device_map="auto"`：由 Transformers/Accelerate 相关加载逻辑自动安排设备；
- `model.eval()`：关闭 Dropout 等训练态行为；
- `eval()` 不会自动关闭梯度，因此普通手写推理仍额外使用了 `torch.no_grad()`。

### 重要：`trust_remote_code=True`

```python
AutoTokenizer.from_pretrained(path, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, ...)
```

这个选项允许 Transformers 加载模型仓库提供的自定义 Python 实现。它用于兼容 ChatGLM、InternLM 等不一定只靠内置类即可加载的模型，但也意味着加载来源必须可信，因为自定义代码可能在本机执行。

## 22. 读取算法配置并安装稀疏 Attention（#277-287）

```python
with open(args.algo_config_path, "r") as f:
    algo_config = json.load(f)

budget_info, score_info = enable_sparse_attention(
    model,
    sparse_config=algo_config,
    enable_budget_info=True,
    enable_score_info=True,
)

return model, tokenizer, algo_config, budget_info, score_info
```

数据流：

```text
算法 JSON 文件
    ↓ json.load
Python dict: algo_config
    ↓
enable_sparse_attention(model, sparse_config=algo_config, ...)
    ├── 解析 compressor / selector / estimator / pruner
    ├── 创建 budget_info
    ├── 创建 score_info
    └── monkey patch LlamaAttention / MistralAttention.forward
```

`enable_sparse_attention()` 返回的是两个统计器，不会返回另一个模型：

```python
budget_info, score_info = enable_sparse_attention(...)
```

模型对象本身在原地被修改，然后原模型引用继续返回。

### 重要：Monkey patch 如何改变后续 `model(...)`

直接依赖 `attention.py` 的当前实现会遍历：

```python
for name, module in model.named_modules():
    if isinstance(module, (LlamaAttention, MistralAttention)):
        module.forward = types.MethodType(attention_forward, module)
        module.state = local_state
```

因此 `pred.py` 后面仍然写普通调用：

```python
output = model(...)
```

但模型内部走到匹配的 Llama/Mistral Attention 层时，实际调用的已经是 Twilight `attention_forward()`。

边界也很明确：当前 monkey patch 的 `isinstance` 只匹配代码中导入的 Llama/Mistral Attention 类。模型名出现在 `pred.py` 的 `choices` 中，不等于该模型内部一定会被当前 patch 命中；这需要看实际模型模块类型与远程代码版本。

---

# 主入口 `__main__`

## 23. 只在脚本直接运行时执行（#290-292）

```python
if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
```

### 重要：`if __name__ == "__main__"`

```text
python benchmark/LongBench/pred.py ...
    → __name__ == "__main__"
    → 执行主流程

import benchmark.LongBench.pred
    → __name__ 是模块名
    → 只加载函数定义，不自动跑完整评测
```

这使 `parse_args()`、`get_pred()` 等函数可以被其他 Python 代码导入。

## 24. 读取模型最大长度与模型路径（#294-307）

```python
config_path = args.config_path
model2maxlen = json.load(
    open(f"{config_path}/model2maxlen.json", "r")
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model_name = args.model
if args.model_path is None:
    model2path_list = json.load(
        open(f"{config_path}/model2path.json", "r")
    )
    model2path = model2path_list[model_name]
else:
    model2path = args.model_path
```

路径选择：

```text
传了 --model_path
    → 直接使用它

没传 --model_path
    → model2path.json[model_name]
```

随后加载并安装稀疏 Attention：

```python
model, tokenizer, algo_config, budget_info, score_info = (
    load_model_and_tokenizer(
        model2path,
        model_name,
        device,
        args,
    )
)

max_length = model2maxlen[model_name]
```

这里的 `max_length` 是仓库评测配置给出的 prompt 截断目标，并不是从 `model.config` 动态读取。

## 25. 读取任务 prompt 与生成长度配置（#310-312）

```python
datasets = [args.task]
dataset2prompt = json.load(
    open(f"{config_path}/dataset2prompt.json", "r")
)
dataset2maxlen = json.load(
    open(f"{config_path}/dataset2maxlen.json", "r")
)
```

虽然变量叫 `datasets`，当前每个进程只有一个任务：

```python
datasets = [args.task]
```

外层保留列表循环结构，方便调用者通过 shell 为不同任务分别启动脚本。

## 26. 创建普通或 LongBench-E 输出目录（#314-321）

```python
if args.e:
    dir = f"{args.output_dir}/pred_e"
    if not os.path.exists(dir):
        os.makedirs(dir)
else:
    dir = f"{args.output_dir}/pred"
    if not os.path.exists(dir):
        os.makedirs(dir)
```

```text
--e 0  → <output-dir>/pred
--e 非0 → <output-dir>/pred_e
```

这里的变量名 `dir` 覆盖了 Python 内建函数 `dir()`；当前局部流程不再调用内建 `dir()`，所以不会直接影响结果，但命名容易造成混淆。

## 27. 从 Hub 或本地 JSONL 加载数据（#323-339）

```python
local_data_set = args.dataset_path
has_local_data_set = os.path.exists(local_data_set)

for dataset in datasets:
    if args.e:
        if not has_local_data_set:
            data = load_dataset(
                "THUDM/LongBench",
                f"{dataset}_e",
                split="test",
            )
        else:
            data = load_dataset(
                "json",
                data_files=f"{local_data_set}/{dataset}_e.jsonl",
                split="train",
            )
    else:
        if not has_local_data_set:
            data = load_dataset(
                "THUDM/LongBench",
                dataset,
                split="test",
            )
        else:
            data = load_dataset(
                "json",
                data_files=f"{local_data_set}/{dataset}.jsonl",
                split="train",
            )
```

选择矩阵：

| 模式 | 本地路径不存在 | 本地路径存在 |
| --- | --- | --- |
| 普通 LongBench | Hub 配置 `dataset` / `test` | `<path>/<dataset>.jsonl` / `train` |
| LongBench-E | Hub 配置 `dataset_e` / `test` | `<path>/<dataset>_e.jsonl` / `train` |

本地 JSONL 使用 `split="train"` 只是 Hugging Face `datasets` 为用户提供的数据文件创建的 split 名称；不代表脚本在训练模型。

## 28. 取得任务配置并运行预测（#341-358）

```python
dataset_prefix = dataset.split("-")[0]

prompt_format = dataset2prompt[dataset]
max_gen = dataset2maxlen[dataset]

preds = get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    budget_info,
    score_info,
)
```

`dataset_prefix` 用于让输出文件名与下游评测键匹配：

```text
repobench-p → repobench
qasper      → qasper
```

这里传入的 `dataset` 仍是基础任务名；LongBench-E 只改变实际加载的数据配置/文件名，不把 `dataset` 变量改成 `dataset_e`。

## 29. 构造输出文件名（#360-368）

```python
if "config_path" in algo_config["selector"]:
    algo_config["selector"].pop("config_path")

out_fn = f"{dataset_prefix}-{str(algo_config)}-{args.t}"
out_fn = out_fn.replace(" ", "")
out_fn = out_fn.replace("'", "")

if len(out_fn) > 245:
    out_fn = out_fn[:245] + "..."

out_path = f"{dir}/{out_fn}.jsonl"
```

输出名包含：

```text
数据集前缀 + 算法配置字典 + 实验时间/标识 + .jsonl
```

删除 selector 的 `config_path` 有两个效果：

1. 避免长路径让文件名过长；
2. 原地修改 `algo_config`，因此后面看到的字典不再包含该键。

当前进程只有一个任务，且模型已经完成 patch，所以这次修改通常不会重新改变运行中的稀疏配置；但它确实改变了 Python 字典对象本身。

## 30. 写入 JSONL（#370-373）

```python
with open(out_path, "w", encoding="utf-8") as f:
    for pred in preds:
        json.dump(pred, f, ensure_ascii=False)
        f.write("\n")
```

JSONL 与一个大 JSON 数组不同：

```json
{"pred": "答案1", "answers": ["参考1"], "length": 1000}
{"pred": "答案2", "answers": ["参考2"], "length": 2000}
```

每一行都是独立 JSON 对象。`ensure_ascii=False` 让中文等非 ASCII 字符直接以 UTF-8 写入，而不是强制变成 `\uXXXX`。

文件以 `"w"` 打开：如果同一路径已经存在，当前运行会覆盖它。

---

# 重要：完整模型推理过程小例子

假设截断并拆分后的 token 是：

```text
context  token IDs = [10, 11, 12]
question token IDs = [20, 21]
max_gen = 3
```

执行过程：

```text
步骤 1：Context Prefill
input_ids = [[10, 11, 12]]        # [1, 3]
past = KV(10, 11, 12)

步骤 2：Question Decode 第 1 个 token
input_ids = [[20]]                 # [1, 1]
past = KV(10, 11, 12, 20)

步骤 3：Question Decode 第 2 个 token
input_ids = [[21]]                 # [1, 1]
past = KV(10, 11, 12, 20, 21)
last logits → argmax → answer token 30
generated_content = [30]

步骤 4：Answer Decode
input_ids = [[30]]                 # [1, 1]
past = KV(..., 30)
last logits → argmax → answer token 31
generated_content = [30, 31]

步骤 5：Answer Decode
input_ids = [[31]]                 # [1, 1]
past = KV(..., 31)
last logits → argmax → EOS 2
generated_content = [30, 31, 2]
遇到 EOS，停止

步骤 6：Decode 文本
tokenizer.decode([30, 31, 2], skip_special_tokens=True)
→ "最终回答"
```

与 Twilight Attention 的关系：

```text
[1, 3] Context
    → q_len=3
    → Prefill Attention

[1, 1] Question/Answer token
    → q_len=1
    → Selector → Estimator → Pruner → sparse mask
    → 更新 budget / score
```

---

# 当前实现的注意点与风险

## 1. `samsum` 使用未定义的 `generated_content`

`samsum` 分支创建 `output`：

```python
output = model.generate(...)[0]
```

公共路径却解码：

```python
pred = tokenizer.decode(generated_content, ...)
```

`generated_content` 只在 `else` 分支创建，因此当前 `samsum` 路径断开。这是由源码直接可见的错误，不是外部版本差异。

## 2. `samsum` 没有把拆出的 `q_input` 传给 `generate`

代码先计算：

```python
context_length = context_tokens + question_tokens
```

但实际只传：

```python
model.generate(**input)
```

这里的 `input` 只对应拆分后的 context，question 没进入输入；`min_length` 却按两段总长度计算。

## 3. ChatGLM3 路径的数据类型契约不闭合

`build_chat()` 中：

```python
prompt = tokenizer.build_chat_input(prompt)
```

后面公共代码仍假定 `prompt` 支持字符串方法：

```python
prompt.rfind(...)
len(prompt)
prompt[q_pos:]
```

再后面 ChatGLM3 分支假定它支持：

```python
prompt.to("cuda")
```

而且没有创建 `q_input`，公共代码却立即读取：

```python
q_input.input_ids
```

因此该路径要求同一个对象同时满足字符串切分、设备移动和 tokenizer 输出接口，当前文件本身没有建立这个契约。

## 4. `q_pos != None` 不能判断 `rfind()` 是否成功

`rfind()` 失败返回 `-1`，不是 `None`。所以 context/question 拆分始终执行，短 prompt 甚至可能只把最后一个字符分给 question。

## 5. 截断发生在 chat 包装之前，且会 decode 后重新 tokenize

```text
第一次 tokenize 检查长度
    → token IDs 首尾截断
    → decode 回字符串
    → build_chat 添加额外文本/特殊 token
    → 再 tokenize
```

最终模型输入长度可能超过 `max_length`，也可能因为 decode/re-tokenize 改变 token 边界。

## 6. `device` 参数与 CPU fallback 没有真正使用

主入口创建了 CPU/CUDA 自适应对象：

```python
device = torch.device(...)
```

但 `get_pred()` 使用硬编码：

```python
.to("cuda")
```

所以当前脚本实际上要求 CUDA。`load_model_and_tokenizer()` 接收 `device`，也没有使用它。

## 7. 首个生成 token 为 EOS 时不会立即停止

EOS 检查只在第二阶段循环中。第一个由 question 最后位置 logits 选出的 token 没有立刻检查，可能多执行一次生成。

## 8. `max_gen=0` 仍会生成一个 token

首个 `argmax` 位于 `range(max_gen - 1)` 外。仓库配置不会传 0，但独立复用 `get_pred()` 时应知道这个边界。

## 9. `torch.cuda.empty_cache()` 每条样本都调用

它不会删除仍被 Tensor 引用的模型权重或 KV Cache，只尝试释放 allocator 的空闲缓存块。频繁调用可能影响性能，因此用本脚本测端到端耗时时，需要把这项开销纳入解释。

## 10. `budget_info` / `score_info` 假定一定存在统计数据

当前加载函数固定传入：

```python
enable_budget_info=True
enable_score_info=True
```

所以两个对象不会是 `None`。但如果没有任何有效 Decode 统计，底层对空列表求均值可能产生 `NaN`；外部单独复用 `get_pred()` 时也必须传入兼容统计器。

## 11. 手写普通推理只传 `input_ids`

```python
model(input_ids=..., past_key_values=..., use_cache=True)
```

tokenizer 生成的 `attention_mask` 没传入普通手写路径。当前脚本每次处理单样本、没有 batch padding，减小了 padding mask 缺失的影响；但具体模型是否还需要其他输入字段，取决于模型实现。

## 12. `trust_remote_code=True` 扩大了外部代码边界

模型与 tokenizer 可以执行仓库自定义代码。实验复现需要记录模型 revision，并确认来源可信；当前脚本没有固定 revision。

## 13. Sparse monkey patch 不保证覆盖 choices 中所有模型

当前直接依赖只 patch `LlamaAttention` 和 `MistralAttention` 实例。ChatGLM、InternLM 等自定义模型即使能加载，也不一定使用这两个类；若没有匹配模块，`pred.py` 仍能继续调用模型，但不会因为模型名在 choices 中就自动获得 Twilight Attention。

## 14. 输出文件名处理仍有文件系统边界

代码按 Python 字符数截到 `245 + "..."`，之后还要追加 `.jsonl`；文件系统通常按字节而非 Python 字符计长度。配置中若还有 `/` 等路径字符，也可能产生非预期路径。源码只特意删除了 selector 的 `config_path`、空格和单引号。

## 15. 多条预测先全部留在内存，最后统一落盘

如果中途异常，尚未进入写文件阶段的预测不会被保存；数据集很大时，预测字典列表也会持续占用主机内存。

---

# 一句话理解每个函数

```text
parse_args
└── 收集模型、任务、路径与稀疏算法配置

build_chat
└── 把 LongBench prompt 包装成不同模型的聊天格式

post_process
└── 清理 XGen / InternLM 的模型专用输出标记

get_pred
├── 构造与截断 prompt
├── context Prefill
├── question + answer 单 token Decode
├── 贪心生成
└── 收集预测与 Twilight 稀疏统计

seed_everything
└── 设置 Python / NumPy / PyTorch 随机种子和 cuDNN 选项

load_model_and_tokenizer
├── 加载 tokenizer / BF16 模型
├── 切换 eval 模式
└── 安装 Twilight Attention 并创建统计器

__main__
├── 读取四类配置
├── 加载一个 LongBench 任务
├── 调用 get_pred
└── 写 JSONL
```

> [!TIP]
>
> CheckList 最终复核
>
> - [x] TL;DR：开头已给出脚本入口、数据流、推理方式、输出和关键风险。
> - [x] 整体流程：已覆盖参数解析、模型加载、稀疏 Attention 安装、数据加载、Prefill/Decode、统计与 JSONL 落盘。
> - [x] 清晰拆分步骤：按 `parse_args`、`build_chat`、`post_process`、`get_pred`、`seed_everything`、`load_model_and_tokenizer` 和主入口逐一拆解。
> - [x] 每步代码块和注释：各函数关键阶段均附源码块、形状、数据流或小例子。
> - [x] 重要语法 / 模型推理：已单列字典解包、BatchEncoding、`unsqueeze`、KV Cache、greedy decoding、`no_grad`、`eval`、`trust_remote_code`、monkey patch 与完整推理示例。
