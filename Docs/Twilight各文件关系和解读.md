# Twilight 各文件关系和解读

## 阅读范围与约定

本文档按仓库目录组织，阅读范围是：

- `benchmark/efficiency` 下全部 Python 文件；
- `benchmark/LongBench` 下全部 Python 文件；
- `benchmark/scripts` 下全部 Shell 文件；
- `twilight/pyimpl`、`twilight/kernel`、`twilight/model` 下全部 Python 文件。

文中的“第 x-y 行”均指本文档生成时对应源码文件的物理行号。空行、注释和格式调整会令行号漂移，因此后续修改源码后应重新核对。这里严格描述**当前源码实际行为**；“注意/潜在问题”表示静态阅读即可确认的未接通路径、接口不一致或强假设，不等同于已经通过运行复现的缺陷。

本文档只分析 Python/Shell 层。`csrc`、`twilight.ops`、`flash-topk-attention/ftka` 和 FlashInfer 内部实现仅在调用边界处说明，不继续逐行展开。

## 全局关系总览

### 准确率评测主链

```text
benchmark/scripts/run_longbench.sh
    -> benchmark/LongBench/pred.py
        -> twilight.pyimpl.enable_sparse_attention(...)
            -> twilight/pyimpl/attention.py::enable_sparse_attention(...)
                -> 解析 compressor / selector / estimator / pruner 配置
                -> _enable_sparse_attention(model, ...)
                    -> 遍历 model.named_modules()
                    -> 找到 LlamaAttention / MistralAttention
                    -> 用 types.MethodType 把其 forward 替换为 attention_forward
        -> 模型逐 token 解码
            -> prefill: FlashAttention，可选 SnapKV 压缩写入 KV cache
            -> decode: Selector -> Weight Estimator -> Pruner -> softmax -> @ Value
        -> 写 pred/*.jsonl
    -> benchmark/LongBench/eval.py
        -> benchmark/LongBench/metrics.py
        -> 写 pred/result.json

benchmark/scripts/run_passkey.sh
    -> benchmark/passkey/passkey.py
        -> 同样调用 twilight.pyimpl.enable_sparse_attention(...)
```

### Twilight decode 的 Select-then-Prune 主链

```text
attention_forward(...), q_len == 1
    -> 计算完整 QK^T / sqrt(head_dim)，得到真实 attn_weights
    -> Selector 生成初筛 mask
       NONE | StreamingLLM | Quest | Oracle TopK | Tidal Decode | SparQ | Double Sparse
    -> Weight Estimator 生成 estimated_weights
       NONE | min-max 量化 K | abs-max 量化 K
    -> Pruner 在估计权重上进一步裁剪
       NONE | elementwise threshold | Top-p CUDA kernel
    -> mask = selector_mask & pruner_mask
    -> 可选记录每层预算和保留概率质量
    -> 用 mask 把未选 logit 设为 dtype 最小值
    -> softmax(fp32) -> 与 V 相乘 -> o_proj
```

### 高性能算子/基准链

```text
benchmark/efficiency/bench_breakdown.py
    -> flash-topk-attention/ftka: Quest GEMV、INT4 GEMV、RAFT TopK
    -> twilight/pyimpl/top_p.py
        -> twilight/kernel/cuda/sampling.py
            -> JIT/预编译 csrc/src/sampling.cu
    -> FlashInfer BatchDecodeWithPagedKVCacheWrapper

benchmark/efficiency/bench_gqa.py
    -> 复用 bench_breakdown.py 的 padded / ragged attention wrapper

benchmark/efficiency/bench_top_p.py
    -> 对比 top-p 的 mask、动态 indices、预分配 indices 三种接口
```

---

## `benchmark`

### `benchmark/efficiency`

#### `benchmark/efficiency/bench_utils.py`

##### 功能、函数与调用关系

这是本目录通用的 CUDA 前向计时辅助文件。

- `benchmark_forward_with_warmup(fn, *inputs, ...)`：先进行指定次数 warmup，再借助 `torch.utils.benchmark.Timer` 对函数计时，并返回 `(timer, measurement)`。
- 内部函数 `amp_wrapper(...)`：把真实调用包在可开关的 CUDA autocast 上下文中。
- 被 `bench_breakdown.py`、`bench_top_p.py` 直接导入；`bench_gqa.py` 通过 `bench_breakdown.py` 间接取得同一个函数。
- `bench_gemv.py` 没有使用本文件，而是使用子模块 `ftka.utils.benchmark` 中的同名实现。

##### 逐行/代码块解读

- 第 1-2 行：导入 PyTorch 和 `torch.utils.benchmark`。
- 第 5-13 行：函数签名把待测函数、位置/关键字参数、warmup/repeat 次数以及 AMP 配置统一起来；默认不启用 AMP。
- 第 14-16 行：warmup 调用。即使 `amp=False` 也进入 autocast 上下文，但上下文被禁用。
- 第 18 行：在正式计时前同步 CUDA，确保 warmup kernel 已经完成。
- 第 20-22 行：定义真正交给 Timer 的包装器，使正式测量与 warmup 使用一致的 autocast 设置。
- 第 24-28 行：`Timer` 通过字符串语句调用包装器，线程数沿用 PyTorch 当前线程配置；`timeit(repeats)` 会返回包含均值等统计量的 Measurement。
- 第 29 行：同时返回 Timer 和 Measurement，调用方通常只使用后者的 `mean` 并换算成微秒。

#### `benchmark/efficiency/bench_breakdown.py`

##### 功能、函数与调用关系

这是 Twilight 高性能路径的**算子级延迟拆分基准**。它分别构造 Quest selector、TopK、INT4 稀疏 GEMV、Top-p 和 FlashInfer attention 的包装器，测量各组件耗时，并可与完整序列 dense/padded attention 比较。

主要函数/类：

- `parse_int_list`：把逗号分隔参数转成整数列表。
- `scaled_head_budgets`：生成 uniform 或按第 4 层实测 profile 缩放的逐 head 预算。
- `get_indices`：构造 page size 为 1、各 batch 等长的 FlashInfer page table 元数据。
- `SpGEMVINT4Wrapper`：包装 FTKA 的 INT4 K-cache GEMV。
- `QuestSelectorWrapper`：包装 FTKA 的 Quest chunk-label GEMV。
- `TopKWrapper`：包装 FTKA/RAFT TopK。
- `TopPWrapper`：包装返回布尔 mask 的 Top-p 路径。
- `TopPIndicesWrapper`：包装写入预分配 indices/counts 的 Top-p 路径。
- `build_ragged_attention_metadata`：把 Top-p 的逐行紧凑 indices/counts 转为 FlashInfer CSR/page-table 元数据。
- `PaddedAttentionWrapper`：固定长度、正常多 head KV cache 的 FlashInfer decode attention。
- `VarlenAttentionWrapper`：从每个逻辑请求的长度列表构造 ragged attention 元数据。
- `VarlenAttentionCSRWrapper`：直接接收已经构造好的 CSR 元数据。
- `bench_component`：统一计时并打印微秒延迟。
- `bench_op_breakdown`：创建数据、选择 `quest` 或 `quest-twi` 路径并汇总各组件时间。
- `main`：解析 CLI，并对多个 batch size 运行基准。

调用关系：本文件调用 `bench_utils.py`；Top-p 经 `twilight/pyimpl/top_p.py` 下沉到 `twilight/kernel/cuda/sampling.py`；Quest/INT4/TopK 来自 `flash-topk-attention/ftka` 子模块；attention 来自 FlashInfer。`bench_gqa.py` 又反向复用本文件的预算函数、计时函数和两个 attention wrapper。

##### 逐行/代码块解读

- 第 1-8 行：导入 CLI、路径、类型、FlashInfer 和 PyTorch。
- 第 9-12 行：计算仓库根目录与 FTKA 子模块根目录，并插到 `sys.path` 最前面，使脚本从 `benchmark/efficiency` 直接运行时也能导入两个项目。
- 第 14-21 行：导入统一计时器、FTKA 三类算子，以及 Twilight Top-p 的 mask/indices/out 三种接口。
- 第 23-56 行：常量 `LAYER4_HEAD_BUDGETS` 保存 32 个 query head 的非均匀预算 profile；末四个 head 的预算明显更大，供 ragged attention 模拟真实 head 间不均衡。

`parse_int_list`（第 59-60 行）：

- 用逗号切分并忽略空字符串，适配 `--batch-sizes 16,32,64` 形式。

`scaled_head_budgets`（第 63-74 行）：

- 第 64-65 行：uniform 模式让每个 head 都等于目标平均预算。
- 第 66-70 行：只接受 `uniform/layer4`；`layer4` 强制 query head 数为 32。
- 第 72-74 行：用 `目标均值 / 原 profile 均值` 得到缩放系数，逐 head 四舍五入且至少保留 1 个 token。

`get_indices`（第 77-84 行）：

- `kv_page_indices` 为每个 batch 重复 `[0, seq_len)`；`kv_page_indptr` 是步长为 `seq_len` 的 CSR 行指针；最后一页长度全为 1。这隐含 `page_size=1`，不同 batch 在合成基准中可共享同一组物理 page id。

`SpGEMVINT4Wrapper`（第 87-135 行）：

- 第 88-96 行：从 packed INT4 cache 推导 page/head 维度；最后一维每个 int8 存两个 int4，因此 `head_size` 乘 2。
- 第 97-106 行：构造 query `[B,H,D]` 和输出 `[B,H,S]`。
- 第 107-120 行：构造逐 page/head 的 scale 与 zero。
- 第 121-123 行：构造等长 page table。
- 第 125-135 行：`forward` 只调用 `batched_sparse_gemv_int4_k` 并让 kernel 写入预分配输出。

`QuestSelectorWrapper`（第 138-171 行）：

- 第 139-147 行：读取 label cache 形状并要求 page size 为 1。
- 第 148-160 行：构造 query、输出和等长 page table。
- 第 162-171 行：把同一个 `k_label` 同时作为 Quest kernel 的 K/V 参数；这里的目的只是测 selector/估分 kernel，并不生成真实 attention value 输出。

`TopKWrapper`（第 174-207 行）：

- 第 175-197 行：准备 `[B,S]` 输入值与输入索引、`[B,K]` 输出值与索引，以及按固定公式分配的 RAFT 临时缓冲区。
- 第 199-207 行：调用 `raft_topk`，结果写入预分配 tensor。

`TopPWrapper`（第 210-228 行）：

- 第 211-215 行：保存阈值并生成 logits。
- 第 217-225 行：token 0 从 `[1.5,2.2]` 采样以模拟 attention sink，其他 token 从 `[-9,-1]` 采样。
- 第 227-228 行：调用 `top_p_unnormalized`，内部先 softmax 再返回布尔 mask；wrapper 丢弃返回值，因为只测延迟。

`TopPIndicesWrapper`（第 231-275 行）：

- 第 232-240 行：先调用动态输出接口得到所有 indices/counts，同步后读取 `counts.max()`，确定安全的最大输出容量。
- 第 241-250 行：按最大真实 count 预分配输出 indices 和 counts。
- 第 251-257 行：先运行一次 out 版本并同步，完成初始化/warmup。
- 第 259-267 行：logits 分布与 `TopPWrapper` 相同。
- 第 269-275 行：正式 `forward` 复用输出 buffer，避免每轮分配。

`build_ragged_attention_metadata`（第 278-299 行）：

- 第 282-283 行：把 `[B,H,1]` counts 和 `[B,H,1,Kmax]` indices 展平为逐 `(batch, head)` 的逻辑请求。
- 第 284-289 行：用 `position < count` 的布尔筛选去除每行 padding，得到一维有效 page indices。
- 第 291-298 行：counts 的前缀和生成 CSR `indptr`；page size 为 1，所以 `kv_last_page_len` 全为 1。
- 第 299 行：返回值可直接交给 `VarlenAttentionCSRWrapper`。

`PaddedAttentionWrapper`（第 302-350 行）：

- 第 303-316 行：从 `[pages,2,page_size,kv_heads,head_dim]` cache 推断元数据，并允许 query head 数不同于 KV head 数（GQA）。
- 第 317-330 行：创建 query、128 MiB workspace 和等长 page table。
- 第 331-347 行：构造 FlashInfer paged decode wrapper，并在计时外执行 `plan`；因此正式测量只包含 `run`。
- 第 349-350 行：执行 padded/equal-length decode attention。

`VarlenAttentionWrapper`（第 353-412 行）：

- 第 354-367 行：把 `seq_lens` 的每个元素视为一个逻辑请求；可令每个逻辑请求只有一个 KV head、一个或一组 query heads。
- 第 368-378 行：创建 query、保存 cache、分配 workspace。
- 第 379-392 行：逐请求拼接 `[0, seq_len)` page id 并累积 `indptr`，形成不同长度的 ragged page table。
- 第 393-409 行：在计时外完成 FlashInfer `plan`。
- 第 411-412 行：正式运行 ragged decode attention。

`VarlenAttentionCSRWrapper`（第 415-464 行）：

- 与上一类的差异是第 418-420、442-444 行直接接收 Top-p 转好的 indices/indptr/last-page-len，不再从长度列表重建。
- 第 426 行从 `kv_last_page_len.numel()` 得出展平后的逻辑 batch 大小。
- 第 445-461 行仍在计时外完成 `plan`，第 463-464 行只计 `run`。

`bench_component`（第 467-471 行）：

- 用统一计时器测量无参闭包/绑定方法，把秒转成微秒，打印并返回均值。

`bench_op_breakdown`（第 474-608 行）：

- 第 490-493 行：固定 page size 1、最多 32768 page、FP16 和 `cuda:0`。
- 第 495-512 行：建立正常多 head KV cache，以及把每个 `(batch,head)` 当独立请求时使用的单 KV-head cache。
- 第 513-519 行：建立 packed INT4 K cache。
- 第 521-529 行：Quest 以 16-token chunk 工作，创建 chunk label GEMV 和从 chunk 分数选 `b0/16` 个 chunk 的 TopK wrapper。
- 第 531-537 行：把 Quest GEMV 与 TopK 放进同一个 lambda，得到“token selector”总延迟。
- 第 539-557 行：`mode=quest` 时跳过 INT4 GEMV/Top-p，attention 固定读取前 `b0` 个 page；可额外测完整 `seq_len` 并打印 attention-only 与端到端比值。
- 第 559 行：`quest-twi` 路径创建在 `b0` 候选上运行的 INT4 GEMV。
- 第 560-576 行：`attention-source=top-p` 时，Top-p 自己生成的 logits 产生 indices/counts，再展平成 `(B*H)` ragged attention；统计平均/最大预算。
- 第 577-584 行：`attention-source=profile` 时，Top-p 仍被单独计时，但 attention 长度来自合成的逐 head profile，而不是 Top-p 输出。
- 第 588-595 行：分别测 INT4 GEMV、Top-p、ragged attention 并打印 profile。
- 第 596-608 行：可测完整序列 attention；端到端时间是四个独立测量均值之和。

`main`（第 611-667 行）：

- 第 612-645 行：提供 batch、head、序列、两级预算 `b0/b1`、Top-p 阈值、模式、预算来源和计时次数等参数。
- 第 648-663 行：逐 batch size 调用 `bench_op_breakdown`。
- 第 666-667 行：脚本入口。

##### 注意与当前实现边界

- 该文件是**算子拆分基准而非真实数据流 benchmark**：Quest TopK 的输出没有传给 INT4 GEMV；INT4 GEMV 的输出也没有传给 Top-p。`attention-source=top-p` 使用独立随机 logits，`profile` 使用静态预算。相加得到的 end-to-end 是组件延迟估算。
- `TopPIndicesWrapper` 初始化时先申请完整序列长度的 indices，再缩到 `counts.max()`；这属于计时外 setup，不计入 `forward`。
- `TopKWrapper.topk_buf` 容量公式包含固定 `8192`，改变 `b0/page_size` 或 kernel 约束时需要单独核对容量。

#### `benchmark/efficiency/bench_gqa.py`

##### 功能、函数与调用关系

该文件比较三种 FlashInfer decode attention 组织方式：逐 query-head 展平、按 GQA group 合并、以及用最大预算 padding。它从 `bench_breakdown.py` 复用 wrapper、预算 profile 和计时函数。

- `group_budgets`：每个 GQA group 取组内 query head 预算最大值，使共享 KV head 覆盖该组需求。
- `bench_gqa`：创建合成 KV cache并测三种布局。
- `main`：解析 CLI 参数并启动测试。

##### 逐行/代码块解读

- 第 1-8 行：导入参数、路径、PyTorch，并把仓库根加入 `sys.path`。
- 第 10-16 行：从同目录 `bench_breakdown` 导入 padded/ragged wrapper、计时器和预算工具。`benchmark_forward_with_warmup` 在 `bench_breakdown` 中又来自 `bench_utils`。

`group_budgets`（第 19-23 行）：

- 按 `group_size` 切片，每组取最大预算；如果 query heads 为 32、KV heads 为 8，则每 4 个 query heads 合为一组。

`bench_gqa`（第 26-105 行）：

- 第 37-38 行：GQA 必须满足 query head 数可整除 KV head 数。
- 第 40-46 行：设置 page size 1、FP16、GPU，计算 group size、逐 head 预算和逐 group 最大预算。
- 第 48-56 行：`flat_kv_cache` 只有一个 KV head，用于把每个 head/group 当成独立逻辑请求。
- 第 57-65 行：`padded_kv_cache` 含真实 `num_kv_heads`，用于标准 GQA padded baseline。
- 第 67-69 行：对每个 batch size，把预算列表重复 `batch_size` 次，形成 batch-major 的逻辑请求长度。
- 第 71-75 行：逐 head 方案有 `B*query_heads` 个请求，每个请求 1 个 query head。
- 第 76-80 行：按 group 方案有 `B*kv_heads` 个请求，每个请求带 `group_size` 个 query heads。
- 第 81-86 行：padded 方案保留原 batch，以所有 head 的最大预算作为统一长度。
- 第 88-105 行：打印预算信息并分别计时，结果换算为微秒。

`main`（第 108-143 行）：

- 参数默认比较 batch 16/32/64、32 query heads、8 KV heads、平均预算 256；默认 profile 为 `layer4`，warmup 100 次、测量 5000 次。
- 第 130-139 行转交所有参数；第 142-143 行是入口。

##### 注意与当前实现边界

- GQA group 预算取组内最大值，保证覆盖但会牺牲一部分逐 head 稀疏性；这个差异正是本基准想量化的权衡。
- 三种 wrapper 都在计时前完成 FlashInfer `plan`，因此结果代表稳态 `run`，不包含重规划成本。

#### `benchmark/efficiency/bench_top_p.py`

##### 功能、函数与调用关系

该文件专门比较 Twilight Top-p 的四种计时口径：`softmax+mask`、纯 mask kernel、动态分配 indices，以及复用预分配 buffer 的 indices-out。

- `parse_int_list`：解析逗号分隔整数列表。
- `random_logits`：生成带 attention sink 的 FP16 logits。
- `bench_top_p`：验证 mask 与 indices counts 一致并计时。
- `main`：解析测试矩阵。

它调用 `bench_utils.py` 和 `twilight/pyimpl/top_p.py`；后者再调用 `twilight/kernel/cuda/sampling.py`。

##### 逐行/代码块解读

- 第 1-8 行：导入参数、路径、PyTorch并把仓库根加入搜索路径。
- 第 10-16 行：导入计时器、CUDA top-p 三种底层接口，以及包含 softmax 的高层接口。

`parse_int_list`（第 19-20 行）：同 `bench_breakdown.py` 的简单 CLI 转换器。

`random_logits`（第 23-39 行）：

- 第 29-30 行：固定随机种子，使不同配置/重复运行可比较。
- 第 32-36 行：输出形状 `[batch, heads, 1, seq]`，dtype 为 FP16、设备为 CUDA。
- 第 37-38 行：token 0 设为较高 logits 模拟 sink，其余 token 为较低分布。

`bench_top_p`（第 42-112 行）：

- 第 51-53 行：遍历 batch/sequence length 组合并生成 logits。
- 第 55-61 行：测 `top_p_unnormalized`，其中包含 FP32 softmax 和 mask kernel。
- 第 63-67 行：在计时外生成 FP32 probs、mask、逐行 count、平均预算和最大预算。
- 第 69-72 行：让动态 indices 接口只分配 `max_budget` 容量，并断言其 counts 与 mask 求和完全相同。
- 第 74-75 行：为 out 接口预分配同形状 indices/counts。
- 第 77-101 行：分别测纯 mask、动态 indices 和预分配 indices-out。
- 第 103-112 行：输出配置、预算统计和四项延迟。

`main`（第 115-144 行）：

- 默认 batch 为 4/8/16/32，序列为 4K/8K/16K/32K，32 heads，`p=0.85`。
- 第 133-140 行调用 benchmark；第 143-144 行是入口。

##### 注意与当前实现边界

- `indices_cap` 仍在每次调用中分配输出 tensor；`indices_out` 才代表 buffer 可复用时的 kernel 路径。
- 底层接口声明 count 不受输出容量裁剪。因此调用方必须像第 63-75 行一样先确保容量至少为 `counts.max()`，否则不能假定完整索引列表都被保存。

#### `benchmark/efficiency/bench_gemv.py`

##### 功能、函数与调用关系

这是 FTKA 稀疏 GEMV 的功能/性能实验脚本，覆盖单 batch FP16、batched FP16、INT8 K cache、INT4 K cache和 Quest GEMV。

- `test_single_sparse_gemv`：比较单请求 FTKA GEMV 与 `torch.matmul`。
- `test_sparse_gemv_int8`：测试量化 INT8 K cache 的 batched GEMV。
- `bench_sparse_gemv_fp16`：测试 FP16 paged sparse GEMV，并构造 PyTorch reference。
- `test_quest_sparse_gemv`：只测 Quest GEMV 延迟。
- `test_sparse_gemv_int4`：测试 packed INT4 K cache GEMV。
- 主入口依次执行单请求、INT8、INT4、FP16 四组测试。

本文件直接依赖 `flash-topk-attention/ftka` 的 CUDA/Triton/benchmark 工具，不调用 `twilight/kernel`。它与 `bench_breakdown.py` 测的是同类 FTKA 算子，但更偏底层 correctness/长度扫描。

##### 逐行/代码块解读

- 第 1-5 行：导入 PyTorch、NumPy 和 time；其中 `sdpa`、`nn`、NumPy、time 在当前有效代码中未使用。
- 第 8-10 行：导入 FTKA 的五种 GEMV、token 搬运 Triton kernel 和 FTKA 自己的计时函数。

`test_single_sparse_gemv`（第 13-63 行）：

- 第 15-20 行：固定 32 heads、head dim 128，扫描 256 到 16384 的序列长度。
- 第 23-31 行：Q/K 全 1，输出预分配；形状分别为 `[H,D]`、`[S,H,D]`、`[H,S]`。
- 第 33 行：运行 FTKA kernel。
- 第 35-39 行：把 Q/K 变成 matmul 需要的 `[H,1,D] @ [H,D,S]` reference。
- 第 41-42 行：可选数值校验。
- 第 44-63 行：分别 warmup/计时 kernel 与 `torch.matmul` 并打印微秒。

`test_sparse_gemv_int8`（第 67-187 行）：

- 第 69-76 行：固定 head 配置，page size 1，最多 32768 page。
- 第 78-94 行：源码注明 kernel 期望交错并偏移 `+128` 的无符号表示；先构造值为 3 的 int8 cache，再转 uint8 加 128。
- 第 98-112 行：scale 全 1、zero 全 0。
- 第 114-133 行：扫描到 32768，构造 `[B,H,D]` Q、`[B,H,S]` 输出和等长 page table。
- 第 135-160 行：先运行并同步，再计时 INT8 kernel；第 162 行以纯数字/逗号输出便于粘贴绘图。
- 第 165-181 行：把原 int8 cache、scale、zero 展开为 FP32 PyTorch reference，计算反量化后的 QK。
- 第 186-187 行：可选正确性检查，但当前比较变量写成 `out_int8`，函数内没有定义该名字；若 `check_correctness=True` 会触发 `NameError`，预分配 kernel 输出实际名为 `o`。

`bench_sparse_gemv_fp16`（第 191-297 行）：

- 第 193-209 行：固定配置并创建 `[pages,1,H,D]` FP16 K cache。
- 第 211-232 行：逐长度创建 Q、四维输出 `[B,H,1,S]` 和 page table。
- 第 234-255 行：运行/计时 `batched_sparse_gemv`。
- 第 257-267 行：构造 reference Q `[B,H,1,D]` 和 K `[B,H,D,S]`。
- 第 271-284 行：额外复制 cache并定义 `_torch_load_dot`，先用 `discontinuous_move_tokens` 模拟不连续搬运，再 matmul；返回值仅用于计时。
- 第 286-297 行：计时 PyTorch load+dot；可选对 `q @ k` 与 FTKA 输出 `o` 做校验。

`test_quest_sparse_gemv`（第 300-374 行）：

- 第 302-328 行：固定 batch 2，创建 FP16 K/V cache。
- 第 330-349 行：逐序列长度构造 Q、输出和 page table。
- 第 351-359 行：一次直接调用被注释掉。
- 第 361-374 行：只计 `quest_sparse_gemv`，未构造 reference/未使用 `check_correctness`。

`test_sparse_gemv_int4`（第 377-497 行）：

- 第 379-403 行：最后一维为 `head_size/2`，表示每个 int8 packed 两个 int4；转为 uint8，但当前没有执行注释里的 `+128`。
- 第 408-422 行：scale/zero 分别为 1/0。
- 第 424-454 行：逐长度构造 Q、输出、page table并运行 INT4 kernel。
- 第 456-472 行：同步、计时、输出微秒。
- 第 475-497 行：reference 解包/校验代码不完整且主体已注释；尤其第 477-483 行直接把 packed 最后一维按 `head_size` reshape，并没有显式 nibble 解包，不能作为真实 INT4 reference。

主入口（第 499-513 行）：

- 默认 `check_correctness=False`，依次跑 single、INT8(batch 64)、INT4(batch 64)、FP16(batch 64)，组间清 CUDA cache。
- Quest 测试入口被注释。

##### 注意与当前实现边界

- INT8 correctness 分支引用未定义的 `out_int8`；默认关闭校验，所以普通入口不会碰到该行。
- INT4 reference/校验尚未完成。
- 多处计算了 `flops` 但没有使用；输出主要是延迟序列，而非吞吐率。

### `benchmark/LongBench`

#### `benchmark/LongBench/metrics.py`

##### 功能、函数与调用关系

该文件实现 LongBench 各任务的基础评价指标，由 `eval.py` 的 `dataset2metric` 表按数据集选择调用。

- `normalize_answer`：英文答案归一化，转小写、去标点/冠词/多余空格。
- `normalize_zh_answer`：中文答案归一化，转小写、去中英文标点和空白。
- `count_score`：从预测中抽取数字，计算目标数字在所有预测数字中的占比。
- `retrieval_score` / `retrieval_zh_score`：分别评估英文/中文段落编号检索。
- `code_sim_score`：取预测中的首个非注释代码行，用模糊字符串相似度评分。
- `classification_score`：按预测中出现的候选类别数对正确类别分数做惩罚。
- `rouge_score` / `rouge_zh_score`：英文 ROUGE-L；中文先分词再复用 ROUGE-L。
- `f1_score`：通用 token/字符 multiset F1。
- `qa_f1_score` / `qa_f1_zh_score`：英文/中文 QA F1。

##### 逐行/代码块解读

- 第 1-10 行：导入正则、标点、结巴、FuzzyWuzzy、Counter 和 Rouge；`difflib`、`List` 在当前文件未使用。

`normalize_answer`（第 12-28 行）：

- 第 15-16 行：仅移除独立单词 `a/an/the`。
- 第 18-19 行：按任意空白切分，再用单空格连接。
- 第 21-23 行：删除 ASCII `string.punctuation` 中的字符。
- 第 25-28 行：先小写，再去标点、冠词、修复空白。

`normalize_zh_answer`（第 31-45 行）：

- 第 34-35 行：删除全部空白而不是用空格连接。
- 第 37-40 行：合并 ASCII 与显式中文标点集合并全部删除。
- 第 42-45 行：小写后依次去标点、去空白。

`count_score`（第 47-54 行）：

- 从 prediction 找出所有连续数字；每个等于 ground truth 的数字计一次。没有数字得 0，否则为 `正确出现次数 / 所有数字数`，因此多报错误数字会降分。

`retrieval_score`（第 56-66 行）：

- 从 ground truth 的 `Paragraph N` 中取第一个 N，再对预测中全部数字使用与 `count_score` 相同的精确率式评分。

`retrieval_zh_score`（第 68-78 行）：

- 与英文版本相同，只是 ground truth 模式为 `段落N`。

`code_sim_score`（第 80-87 行）：

- 去除开头换行后逐行扫描，跳过含反引号、`#` 或 `//` 的行；只取第一个剩余行。
- 用 `fuzz.ratio` 计算 0-100 相似度并除以 100。

`classification_score`（第 89-102 行）：

- 第 91-94 行：收集 prediction 中作为子串出现的所有候选类别。
- 第 95-97 行：试图去除“作为 ground truth 子串但不等于 ground truth”的匹配，以降低类别名嵌套干扰。
- 第 98-102 行：若正确类别存在，得分为 `1 / 命中类别数`，否则为 0。

`rouge_score`（第 104-110 行）：

- 每次调用新建 `Rouge`，用列表形式求平均 ROUGE-L F1；任何异常都返回 0。

`rouge_zh_score`（第 112-116 行）：

- 用结巴精确模式切分中英文预测/答案，以空格连接后调用英文 `rouge_score`。

`f1_score`（第 118-126 行）：

- Counter 交集保留每个 token 的最小出现次数；无交集返回 0，否则计算 precision、recall 和调和平均。
- 输入既可以是 token 列表，也可以是字符串；字符串会按字符计数。

`qa_f1_score`（第 128-134 行）：

- 英文归一化后按空格切成 token，再调用通用 F1。

`qa_f1_zh_score`（第 137-144 行）：

- 先结巴分词，再逐 token 中文归一化，删除空 token，最后计算 multiset F1。

##### 注意与当前实现边界

- 两个 retrieval 函数直接读取 `matches[0]`，ground truth 若不符合预期格式会抛出 `IndexError`。
- `classification_score` 在遍历 `em_match_list` 时原地删除元素，连续需要删除的元素可能因迭代索引移动而被跳过。
- `rouge_score` 用裸 `except` 把依赖/输入等所有异常都折算成 0，会隐藏具体失败原因。

#### `benchmark/LongBench/eval.py`

##### 功能、函数与调用关系

该文件读取 `pred.py` 生成的 JSONL，按数据集选择 `metrics.py` 中的指标并写出 `result.json`。

- 全局 `dataset2metric`：数据集到指标函数的分发表。
- `parse_args`：解析模型名、LongBench-E 开关和输出位置。
- `scorer_e`：逐样本取多个 ground truth 中最高分，再按上下文长度分桶。
- `scorer`：普通 LongBench 的整体平均分。
- 主入口：发现输出目录中所有 JSONL，聚合评分并写 JSON。

上游是 `run_longbench.sh`/`eval_longbench.sh` 和 `pred.py`，下游只有结果 JSON 文件。

##### 逐行/代码块解读

- 第 1-4 行：导入文件、JSON、CLI 和 NumPy。
- 第 6-16 行：从同目录 `metrics.py` 导入九类数据集指标。
- 第 18-40 行：定义 21 个数据集到指标的映射；例如 QA 用 token F1，摘要用 ROUGE-L，检索用编号指标，代码用模糊相似度。

`parse_args`（第 43-50 行）：

- `--e=1` 表示 LongBench-E；`--output-path` 非空时优先直接评估该目录，`--model` 和 `--t` 在当前评分逻辑中没有实际参与路径/计算。

`scorer_e`（第 53-74 行）：

- 第 54 行：初始化 `<4K`、`4K-8K`、`>=8K` 三个分桶。
- 第 55-65 行：逐预测遍历多个可接受答案，取最大指标分；四个任务只保留预测第一行。
- 第 66-71 行：按原样本 `length` 分桶。
- 第 72-74 行：每桶求均值、乘 100、保留两位小数。

`scorer`（第 77-91 行）：

- 逻辑与 `scorer_e` 的逐样本部分一致，但不分长度桶，最后用预测数作为分母返回百分制均值。

主入口（第 94-132 行）：

- 第 95-105 行：路径优先级为显式 `output_path`，其次 LongBench-E 的 `pred_e`，最后普通 `pred`。
- 第 107-111 行：列目录并仅处理 `.jsonl` 文件。
- 第 112-121 行：数据集名取文件名第一个 `-` 之前的前缀；逐行收集 `pred/answers/all_classes` 和可选 `length`。
- 第 122-126 行：根据 `--e` 调用分桶或整体 scorer，以完整文件名作为结果 key。
- 第 127-132 行：两个分支实际都写 `{dir}/result.json`；`ensure_ascii=False` 保留中文。

##### 注意与当前实现边界

- `all_classes` 在读每一行时被覆盖，评分时使用最后一条记录的值；这依赖同一数据集文件内候选类别恒定。
- LongBench-E 某个长度桶若没有样本，`np.mean([])` 会产生 `NaN` 和警告，最终 JSON 可能包含非标准的 `NaN` 值。
- 普通 scorer 在空预测文件上会除以 0。
- `dir` 覆盖了 Python 内建函数名，但在本脚本局部使用中不影响功能。

#### `benchmark/LongBench/pred.py`

##### 功能、函数与调用关系

这是 LongBench 推理主程序：加载数据集/模型/算法配置，调用 `enable_sparse_attention` 给 Llama/Mistral attention 打补丁，手写 prefill + token-by-token greedy decode，记录预测、稀疏预算与保留概率质量，最终写 JSONL。

- `parse_args`：解析模型、任务、数据/配置/输出路径和稀疏算法配置。
- `build_chat`：按模型系列包装聊天模板。
- `post_process`：对 XGen/InternLM 输出做简单清理。
- `get_pred`：逐样本构造 prompt、截断、拆分 context/question、解码并收集统计。
- `seed_everything`：设置 Python/NumPy/PyTorch 随机性。
- `load_model_and_tokenizer`：加载 HF 模型和 tokenizer，并启用 Twilight Python 稀疏 attention。
- 主入口：读取配置、取得数据集、调用 `get_pred` 并命名输出文件。

核心调用链：

```text
run_longbench.sh
  -> pred.py::load_model_and_tokenizer
    -> twilight/pyimpl/__init__.py::enable_sparse_attention
      -> twilight/pyimpl/attention.py monkey-patch 每个 Attention.forward
  -> pred.py::get_pred
    -> 模型 prefill/decode
    -> HistoryBudgetInfo / HistoryAccumulatedScoreInfo
  -> pred/*.jsonl
  -> eval.py -> metrics.py
```

##### 逐行/代码块解读

- 第 1 行：注明源自 Quest LongBench 评测脚本。
- 第 3-17 行：导入 Hugging Face datasets/transformers、Torch、JSON、进度条和随机数；`AutoConfig`、`LlamaTokenizer`、`LlamaForCausalLM` 在当前代码未使用。
- 第 19 行：从 `twilight.pyimpl` 公共入口导入 sparse-attention patch 函数。

`parse_args`（第 22-59 行）：

- 第 24-40 行：`--model` 限定为列出的 10 种模型标识。
- 第 41-57 行：本地模型、LongBench-E、单任务、输出、数据、LongBench 配置和算法 JSON 路径。
- 第 59 行：支持传入自定义 args，便于测试。

`build_chat`（第 63-87 行）：

- 第 64-67 行：ChatGLM3/ChatGLM 使用 tokenizer 自带构造器。
- 第 68-74 行：LongChat/Vicuna 使用 FastChat 的 Vicuna conversation template。
- 第 75-86 行：Llama2、XGen、InternLM、Llama 3.1 分别拼接各自特殊模板/token 文本。
- 第 87 行：未匹配模型原样返回。

`post_process`（第 90-95 行）：

- XGen 去首尾空白和字面 `Assistant:`；InternLM 截断 `<eoa>` 之后内容。

`get_pred`（第 98-257 行）：

- 第 111-118 行：创建结果列表，逐数据样本清 CUDA allocator cache并按数据集模板格式化 prompt。
- 第 119-126 行：先无截断 tokenize；ChatGLM3 关闭 special tokens 后重做一次。
- 第 127-131 行：超出模型最大长度时保留 token 序列前半和后半，再 decode 回文本，避免只截掉开头或结尾关键信息。
- 第 132-140 行：对若干分类/短答/代码任务不套 chat template，其余调用 `build_chat`。
- 第 142-152 行：按任务寻找问题/指令开始标志；未列出任务设为 `-1`。
- 第 154-159 行：最多只把末尾约 100 个**字符**作为模拟 decode 的 question；context 与 question 分开。
- 第 161-170 行：普通模型分别 tokenize context/question 到 CUDA，去掉 question 的首个特殊 token；代码不使用传入的 `device` 参数。
- 第 172 行：context token 数与 question token 数相加。
- 第 174-189 行：`samsum` 特例调用 `model.generate`，增加换行 token 作为 EOS 并设置最小长度。
- 第 190-205 行：其他任务先对 context 做一次 prefill，随后逐个输入 question token、更新 KV cache，模拟 question-stage decode。
- 第 206-221 行：从 question 最后一个输出取首个贪心 token，再最多生成 `max_gen-1` 个 token，遇 tokenizer EOS 停止。
- 第 223-229 行：原来的 `model.generate` 普通路径保留为注释。
- 第 231-233 行：解码 `generated_content` 并按模型清理。
- 第 238-242 行：从 Twilight 状态统计整个样本的平均预算/保留概率质量，然后 reset，避免样本间累积。
- 第 245-256 行：每条结果保存 prediction、参考答案、类别、长度、平均预算和 score sum。
- 第 257 行：返回整个数据集预测列表。

`seed_everything`（第 260-267 行）：

- 同时设置 CPU CUDA、NumPy、Python random；关闭 cuDNN benchmark，启用 deterministic，并设置所有 GPU seed。

`load_model_and_tokenizer`（第 270-287 行）：

- 第 271-275 行：tokenizer/model 都允许 remote code；模型以 BF16、`device_map=auto` 加载并切 eval。
- 第 277-278 行：读取算法 JSON。
- 第 280-285 行：启用稀疏 attention，同时打开 budget/score 记录，获得两个状态统计器。
- 第 287 行：把模型、tokenizer、算法配置和统计器一起返回。

主入口（第 290-373 行）：

- 第 291-307 行：固定 seed，读取模型最大长度；模型路径可来自 `model2path.json` 或 CLI 覆盖。
- 第 308-312 行：任务只有 `args.task` 一个；读取 dataset prompt 与最大生成长度配置。
- 第 314-321 行：按普通/E 模式创建 `pred` 或 `pred_e` 输出目录。
- 第 323-339 行：若 `--dataset-path` 在本地不存在，则从 `THUDM/LongBench` 下载；否则从本地 JSONL 载入。E 模式使用 `{dataset}_e`。
- 第 341-345 行：文件名前缀去掉任务名 `-` 后缀，例如 `repobench-p -> repobench`；取得 prompt 与生成长度。
- 第 346-358 行：运行预测。
- 第 360-367 行：为了文件名不过长，删除 selector 的 `config_path`，把算法配置与时间拼到名字里，移除空格/单引号并截到约 245 字符。
- 第 368-373 行：逐条写 UTF-8 JSONL。

##### 注意与当前实现边界

- `samsum` 分支得到变量 `output`，但第 231 行统一解码 `generated_content`；该变量只在非 `samsum` 分支创建，因此当前 `samsum` 路径会出现未定义变量。并且此特例只把 context `input` 传给 `generate`，没有合并拆出的 `q_input`。
- ChatGLM3 的 `build_chat_input` 返回值通常不是普通字符串，而第 142-159 行继续执行字符串的 `rfind`、切片；第 163 行又假定 `prompt` 有 `.to()`。该路径的数据类型契约与普通路径不一致，需要结合目标 Transformers 版本验证。
- 第 157 行判断 `q_pos != None`，但 `q_pos` 在前面始终是整数；该分支总会执行。这不一定导致错误，但判断表达的语义不是“是否找到分隔符”。
- `device` 参数和 CPU fallback 实际被第 163、166、168 行硬编码的 CUDA 覆盖，脚本当前本质上要求 GPU。
- `budget_info/score_info` 在关闭统计时会是 `None`，但本加载函数固定打开二者，所以当前入口安全；外部复用 `get_pred` 时需遵守这一前提。
- 第 360-361 行会原地修改 `algo_config`；当前每进程每次只写一个任务，因此影响主要体现在输出文件名，不会重新配置已 patch 的模型。

### `benchmark/scripts`

#### `benchmark/scripts/eval_longbench.sh`

##### 功能、函数与调用关系

这是只运行 LongBench 评分的简短脚本，调用 `benchmark/LongBench/eval.py` 并允许指定一个已经整理好的结果目录。

- 没有 Shell 函数。
- 当前路径写法 `../LongBench/eval.py` 表明预期工作目录是 `benchmark/scripts`。
- 下游 `eval.py` 再调用 `metrics.py` 并在 `OUTPUT_PATH` 写 `result.json`。

##### 逐行/代码块解读

- 第 1-2 行：连续给 `MODEL` 赋值，第二行覆盖第一行，所以实际模型名是 `Meta-Llama-3.1-8B-Instruct`。
- 第 4-5 行：普通 LongBench 模式，未启用 E。
- 第 7 行：`time=0` 会传给 eval，但当前 `eval.py` 不使用该参数做筛选。
- 第 9-10 行：设置通用输出目录和显式待评估目录 `final-twi`；显式 `output-path` 优先级最高。
- 第 12 行：用 unbuffered Python 启动评估。变量都未加双引号，当前值无空格时可用。

#### `benchmark/scripts/run_longbench.sh`

##### 功能、函数与调用关系

这是 LongBench 端到端调度脚本：选择普通/E 数据集列表，对每个任务调用 `LongBench/pred.py`，全部完成后调用 `LongBench/eval.py`。

- 没有 Shell 函数。
- 预期从 `benchmark` 目录运行，因为使用 `LongBench/...`、`LongBench/config` 和相对结果目录。
- 第一个位置参数 `$1` 必须是算法配置 JSON 路径，通常对应 `benchmark/configs/config_*.json`。

##### 逐行/代码块解读

- 第 1-4 行：固定模型标识、本地模型路径和输出目录。
- 第 6-9 行：`DATASET_PATH=HF`；只要当前目录不存在名为 `HF` 的路径，`pred.py` 就会走 Hugging Face `load_dataset`。配置路径指向 LongBench JSON 配置。
- 第 11 行：创建输出目录。
- 第 13-17 行：注释列出可选算法配置；实际使用脚本第一个参数。
- 第 19-20 行：普通 LongBench，`enable_e=0`。
- 第 22-29 行：定义普通和 LongBench-E 数据集数组；普通列表主动忽略注释所列的一批任务。
- 第 31-35 行：按 `enable_e` 复制对应数组到 `benchmark_dataset`。
- 第 37-41 行：用月日时分生成本次运行标识，分钟内重复启动可能同名。
- 第 43-56 行：遍历数据集；中间大量注释是常用的手工子集选择方案。
- 第 57-64 行：逐任务打印名称，设置 expandable CUDA allocator，然后传模型、任务、算法、数据/输出/配置路径给 `pred.py`。
- 第 65-74 行：旧的多 budget 循环模板，全部注释，不执行。
- 第 75 行：结束任务循环。
- 第 77 行：评估本次输出目录中的**所有 JSONL**；`eval.py` 不按传入时间筛选，结果会包含目录中先前运行文件。

##### 注意与当前实现边界

- 脚本没有检查 `$1` 是否为空；漏传算法配置会在 `pred.py` 打开配置时失败。
- 路径和变量展开大多未加双引号，模型/配置路径含空格时会被 Shell 拆词。
- 未设置 `set -e`，某个预测任务失败后循环仍可能继续，最后仍会启动评估。

#### `benchmark/scripts/run_passkey.sh`

##### 功能、函数与调用关系

这是 Passkey 准确率评测入口，调用 `benchmark/passkey/passkey.py`。虽然 `passkey.py` 不在本文逐行分析范围内，但它会读取算法 JSON、调用 `twilight.pyimpl.enable_sparse_attention`，因此最终进入与 LongBench 相同的 Python sparse-attention 主链。

- 没有 Shell 函数。
- 预期从 `benchmark` 目录运行，因为使用 `./passkey/passkey.py` 和 `./configs/...`。

##### 逐行/代码块解读

- 第 1 行：提示下面参数需按环境修改。
- 第 2-3 行：模型标识和本地模型目录。
- 第 5-7 行：按模型名构造结果目录并创建。
- 第 9 行：固定长度为 13000；`passkey.py` 内部会再按字符/token 逻辑处理。
- 第 10 行：使用 Quest、初始预算 1024 的配置。
- 第 12-15 行：运行 100 次 fixed-length 测试，把算法配置和输出路径传给 `passkey.py`。

##### 注意与当前实现边界

- 变量和路径未加双引号，含空格路径不安全。
- 当前文件在生成本文档之前已经存在用户修改（新增第 1 行提示）；本文档生成过程没有改动该 Shell 文件。

---

## `twilight`

### `twilight/pyimpl`

该目录是面向准确率验证的纯 PyTorch/朴素实现。它刻意允许在 decode 阶段先计算完整 QK 权重，再模拟不同 selector/estimator/pruner 的稀疏结果，因此不能把这里的运行速度视为最终 kernel 性能。

#### `twilight/pyimpl/__init__.py`

##### 功能、函数与调用关系

这是 `twilight.pyimpl` 的公共导出入口，没有自定义函数。它从 `attention.py` 暴露：

- `enable_sparse_attention`：给模型安装稀疏 attention monkey patch；
- `reset_sparse_config`：允许全局 `LocalState` 接受下一次配置。

`benchmark/LongBench/pred.py`、`benchmark/passkey/passkey.py` 和 RULER wrapper 都通过本文件导入入口，而不是直接依赖 `attention.py`。

##### 逐行/代码块解读

- 第 1 行：模块说明。
- 第 3 行：重导出两个 API。导入本包时会加载 `attention.py` 以及它依赖的各策略和 kernel wrapper。

#### `twilight/pyimpl/state.py`

##### 功能、函数与调用关系

该文件定义稀疏 attention 的配置枚举、共享状态和评测统计结构。

- `CompressorType.from_str`：把配置字符串转成 `NONE/SNAP_KV`。
- `IndexSelectorType.from_str`：转换 Selector 类型，枚举含 Streaming、Quest、Oracle TopK、Tidal Decode、SparQ、Double Sparse、PQCache。
- `WeightEstimatorType.from_str`：转换估计器类型，枚举含量化与 threshold 类型。
- `WeightPrunerType.from_str`：转换 pruner 类型，枚举含 threshold 与 Top-p。
- `LocalState.set_args`：一次性/动态地挂载全部配置和统计器。
- `HistoryAccumulatedScoreInfo`：按 query/layer/head 记录 mask 保留的真实 softmax 概率质量。
- `HistoryBudgetInfo`：按 query/layer/head 记录实际 token budget，以及可选初始预算 B0。

`attention.py` 创建全局 `LocalState`，并在每层 decode 时更新两个 History 对象；`LongBench/pred.py` 在每个样本结束后读取总平均值并 reset。

##### 逐行/代码块解读

- 第 1-9 行：导入 Enum、Torch/NN、NumPy 和列表类型。

四组枚举（第 12-63 行）：

- `CompressorType`（第 12-19 行）：0 表示不压缩，1 表示 SnapKV；`from_str` 大写后用枚举名字索引，未知字符串会抛 `KeyError`。
- `IndexSelectorType`（第 22-35 行）：值 0-7 列出 NONE、STREAMING、QUEST、ORACLE_TOPK、TIDAL_DECODE、SPARQ、DS、PQCACHE；转换逻辑同上。
- `WeightEstimatorType`（第 38-50 行）：NONE、MIN_MAX_QUANT、MAX_QUANT，以及数值 5/6 的 THRESHOLD/H_THRESHOLD。
- `WeightPrunerType`（第 53-63 行）：NONE、THRESHOLD、TOP_P。

`LocalState`（第 66-79 行）：

- 第 67-71 行：作为所有 attention layer 共享的可变对象；注释强调通过全局状态共享配置。
- 第 73-74 行：初始只设 `arg_set=False`。
- 第 76-79 行：遍历任意关键字并 `setattr`，最后标记配置已设置。没有字段 schema/type 校验。

`HistoryAccumulatedScoreInfo`（第 82-133 行）：

- 第 83-86 行：构造时立即 reset。
- 第 87-94 行：`_score_sum` 和 `_min_sum` 都按 query 存字典，字典键为 layer；`_last_layer_id=-999` 用于检测新 query。
- 第 95-100 行：`update` 对真实 logits 做 FP32 softmax，把 mask 外概率清零，按 token 维求每个 head 的保留概率质量。
- 第 104-110 行：若当前 layer 不是上一 layer+1，则开始一个新 query；保存逐 head score 和其中最小值，再更新最后 layer。
- 第 112-120 行：两个打印函数分别展示某 query 的层最小值和完整逐 head 值。
- 第 122-130 行：对每个 query，只统计 `layer_id>1` 的层，先对层的最小 head 保留质量求均值。
- 第 132-133 行：再对所有 query 均值求总平均并转 Python float。

`HistoryBudgetInfo`（第 136-234 行）：

- 第 139-149 行：构造/reset；`_budget_num`、`_B0` 按 query 存层字典。
- 第 151-159 行：对布尔 mask 在 token 维求和，得到逐 head 实际预算；通过 layer 连续性判断新 query。
- 第 161-167 行：记录某层标量 B0。
- 第 169-186 行：打印实际逐 head budget、B0、head 平均；层号大于 1 才进入总和，末尾固定用 30 层和 32 heads 求均值。
- 第 188-195 行：不含 B0 的逐层打印。
- 第 197-203 行：返回指定 query 每层的 head 平均预算列表。
- 第 205-211 行：对最后一个 query 计算相同列表；虽然注解写 `float`，实际返回 `list`。
- 第 213-222 行：每个 query 只取 `layer_id>1`，先对 head 求均值，再对层求均值。
- 第 224-228 行：B0 统计丢弃每个 query 字典的第一个值后求均值。
- 第 230-234 行：分别对 query 级 B1/B0 均值再求全局平均。

##### 注意与当前实现边界

- 枚举中 `IndexSelectorType.PQCACHE`、`WeightEstimatorType.THRESHOLD/H_THRESHOLD` 在当前 `attention.py` 没有对应执行分支；配置为这些值会让 `mask` 或 `estimated_weights` 未赋值。
- `HistoryBudgetInfo.update_B0` 不更新 `_last_layer_id`，却使用它判断新 query；若单独调用或调用顺序改变，`_B0` 的 query 分组可能与 `_budget_num` 不一致。
- `print_budget_info_single_query_with_B0` 固定假设 32 层中跳过前 2 层（30）和 32 heads，不适用于其他模型结构。
- 若 query 没有 `layer_id>1` 的数据，NumPy mean 会产生 NaN。

#### `twilight/pyimpl/attention.py`

##### 功能、函数与调用关系

这是准确率路径的核心调度器。它实现 Llama/Mistral attention 替代 `forward`，并通过 monkey patch 把压缩器、selector、weight estimator 和 pruner 组合到同一执行框架中。

- `attention_forward`：prefill 使用 FlashAttention并可压缩写入 cache；decode 计算完整权重后执行 Select-then-Prune。
- `enable_sparse_attention`：解析字典配置、创建统计器并调用内部安装函数。
- 内部 `_split_type_args`：从一个策略配置中拆出 `type` 和剩余参数。
- `_enable_sparse_attention`：初始化全局状态、可选加载 Double Sparse channel 配置、遍历模型并替换 Attention.forward。
- `reset_sparse_config`：只把全局状态的 `arg_set` 复位。

文件依赖关系：

```text
attention.py
  prefill compressor -> snap_kv.py
  selectors          -> streaming_llm.py / quest.py / top_k.py /
                        tidal_decode.py / sparq.py / double_sparse.py
  estimators         -> quantize.py
  pruners            -> elementwise_threshold.py / top_p.py
  state/statistics   -> state.py
  base attention     -> Transformers Llama/Mistral + FlashAttention helper

上游入口：pyimpl/__init__.py -> LongBench / Passkey / RULER
```

##### `attention_forward` 逐行/代码块解读

- 第 60-69 行：签名兼容 Transformers attention forward，返回 `(attn_output, attn_weights, past_key_value)`。
- 第 70 行：每次 attention 调用都清空 CUDA allocator cache；这可能降低准确率脚本的稳态性能，但本目录本就不是效率实现。
- 第 72-78 行：读取 batch/query length；`bsz` 与 `bs` 重复表示同一维度。
- 第 82-96 行：分别经 q/k/v projection，reshape/transpose 成 Q `[B,Hq,Q,D]`、K/V `[B,Hkv,Q,D]`。
- 第 98-116 行：估算 KV 长度。存在 past cache 时要求 `layer_idx`；SnapKV 路径优先用模块自带 `kv_seq_len`，否则询问 cache 的 usable length。
- 第 118-122 行：由 rotary embedding 得到 cos/sin，对 Q/K 应用 RoPE。
- 第 124-126 行：`repeat_kv` 把 KV heads 扩展到 query head 数，提前把 GQA 变成 MHA 形状。
- 第 128-136 行：`use_cache` 且 decode (`q_len==1`) 时，把当前 K/V 更新进 cache并取回完整历史 K/V。
- 第 137-147 行：prefill 时保留原 K/V 作为默认 cache；若 compressor 是 SnapKV，就先压缩，再写 cache。当前用于本次 prefill attention 的 `key_states/value_states` 仍是未压缩版本，压缩只影响后续 decode cache。
- 第 148-154 行：旧 tuple cache 兼容代码已注释。
- 第 156-174 行：prefill 把布局改为 `[B,Q,H,D]`，调用 Transformers `_flash_attention_forward`；传入 causal/sliding-window/position 等信息。
- 第 176-182 行：reshape、`o_proj`；不要求输出 attention 时设为 None并立即返回。
- 第 184-185 行：decode 当前强制不接受 attention mask。
- 第 187-190 行：即使后续做稀疏，也先计算完整 `QK^T/sqrt(D)`，所以这是准确率模拟。
- 第 195-197 行：默认跳过前两层；未跳过时从 layer 0 开始应用稀疏逻辑。

Selector 分发（第 198-242 行）：

- `NONE`（第 199-200 行）：全 True mask。
- `STREAMING`（第 201-206 行）：保留 sink 与最近窗口。
- `QUEST`（第 207-213 行）：用 query-aware chunk 上界估分选择 chunk。
- `ORACLE_TOPK`（第 214-218 行）：直接对真实完整 logits 取 TopK，属于 oracle baseline。
- `TIDAL_DECODE`（第 219-226 行）：传共享 state、层号和 reselection layers。
- `SPARQ`（第 227-232 行）：用 query 大幅度 channel 的部分 QK 估分后 TopK。
- `DS`（第 233-242 行）：复制配置并删除仅初始化用的 `config_path/selected_channel`，再用模块上的 `sorted_channel` 做 Double Sparse。

Weight Estimator（第 244-265 行）：

- `NONE`：估计权重就是完整真实权重。
- `MIN_MAX_QUANT`：逐 token 对 K min-max 伪量化/反量化，再算 QK；selector mask 外先置 dtype 最小值。
- `MAX_QUANT`：对 K 做 abs-max 对称伪量化，可选 smooth，再算 QK并屏蔽 selector 外位置。

Pruner、统计和输出（第 267-339 行）：

- 第 267-278 行：THRESHOLD 调相对 token-0 阈值；TOP_P 对未归一化 estimated weights 先 softmax 后调用 CUDA top-p mask。
- 第 280-281 行：存在 pruner 时与 selector mask 取交集，构成 Select-then-Prune。
- 第 283-293 行：可选记录实际逐 head budget，以及该 mask 在真实 softmax 下保留的概率质量。
- 第 295-296 行：可选择让最终 attention 使用估计 logits；否则估计值只用于 pruning。
- 第 298-299 行：mask 外 logits 写为 dtype 最小有限值。
- 第 301-306 行：以 FP32 softmax 再转回 Q dtype，与 V 相乘。
- 第 308-323 行：与完整 attention 输出做 cosine similarity 的调试代码已注释。
- 第 325-329 行：严格检查输出形状 `[B,H,Q,D]`。
- 第 331-339 行：转回 `[B,Q,hidden]`、执行 `o_proj`、按需丢弃权重并返回。

##### 配置与 monkey patch 逐行/代码块解读

- 第 342-343 行：模块级唯一 `local_state=LocalState()`，所有被 patch 的层共享同一个对象。
- 第 351-377 行：`enable_sparse_attention` 的接口和配置模板，包含 compressor、selector、estimator、pruner、是否使用估计权重、是否跳过前两层。
- 第 379-383 行：内部 `_split_type_args` 浅拷贝单个策略字典，取出并删除 `type`，返回类型名与剩余参数。
- 第 385-392 行：分别解析四个策略部分；这些 key 在配置中必须存在。
- 第 394-395 行：按开关创建预算/概率质量统计器。
- 第 397-400 行：两个全局布尔项提供默认值 False/True。
- 第 402-416 行：字符串经各 Enum 的 `from_str` 转换，所有对象/参数交给 `_enable_sparse_attention`。
- 第 418 行：返回统计器给评测脚本。
- 第 421-441 行：内部安装函数签名；多个参数默认值是空字典，但函数内没有修改这些默认字典。
- 第 442-445 行：文档说明同一个 local state 未 reset 时不应重复配置。
- 第 447-461 行：只有首次/已 reset 时才把配置和统计器挂到共享状态。
- 第 463-474 行：Double Sparse 额外读取 JSON channel profile，通过 `init_model_channel_config` 给每个 attention 层加 `sorted_channel`。
- 第 476 行：打印完整状态字典。
- 第 478-482 行：SnapKV 的 `prepare_inputs_for_generation` monkey patch 当前被注释，没有生效。
- 第 484-494 行：扫描 `model.named_modules()`；匹配 Llama/Mistral Attention 后从模块名第三段提取 layer id，用 `types.MethodType` 绑定替代 forward，并挂共享 state。
- 第 497-499 行：reset 只把 `arg_set=False`，不会恢复原 attention forward，也不会删除其他状态字段或已加载 channel。

##### 与用户示例对应的递归/扫描关系

本文件用的是 `named_modules()` 的扁平递归迭代，而不是显式递归函数；等价过程可理解为：

```text
_enable_sparse_attention(LlamaForCausalLM, ...)
    -> named_modules 遍历 LlamaForCausalLM 自身：不是 Attention，跳过
    -> 遍历 model(LlamaModel)：不是 Attention，跳过
    -> 遍历 model.layers.0(LlamaDecoderLayer)：不是 Attention，跳过
    -> 遍历 model.layers.0.self_attn(LlamaAttention)
         -> layer_id = 0
         -> self_attn.forward = MethodType(attention_forward, self_attn)  ✅
         -> self_attn.state = 全局 local_state                        ✅
    -> 遍历 model.layers.0.mlp：不是 Attention，跳过
    -> ...
    -> 遍历 model.layers.N.self_attn：逐层重复 patch
```

Double Sparse 的附加扫描类似：

```text
init_model_channel_config(model, channel_config, ...)
    -> 遍历所有 named_modules
    -> 遇到 LlamaAttention / MistralAttention
         -> 从名称提取 layer_idx
         -> 读取 model.layers.{idx}.self_attn.{k|q}_proj 的 channel 排名
         -> 截取前 r 个，转 tensor、contiguous、搬到 CUDA
         -> module.sorted_channel = tensor  ✅
    -> 其他模块跳过
```

##### 注意与当前实现边界

- `TIDAL_DECODE` selector 当前返回 `(None, mask)`，而第 220 行把整个返回值赋给 `mask`；后续 `~mask`、逻辑与和统计都期望 Tensor，因此接口不一致。
- 状态枚举中的 PQCACHE、THRESHOLD/H_THRESHOLD estimator 没有分支。配置命中时局部变量可能未定义。
- `_enable_sparse_attention` 通过 `name.split('.')[2]` 提取 layer id，强依赖模块名形如 `model.layers.N.self_attn`；不同模型包装层级会解析失败或取错段。
- 第二次直接调用 `enable_sparse_attention` 时，若没有先 reset，新的配置/统计器不会写入 state，但遍历 patch 仍执行；调用方拿到的新统计器可能没有被 attention 使用。
- `reset_sparse_config` 不恢复原 forward。它只允许下一次覆盖共享参数，不是“卸载 sparse attention”。
- decode 强制 `attention_mask is None`，不支持 padding/额外 mask 场景。
- 每层每 token 调用 `torch.cuda.empty_cache()` 会显著扰动性能，印证该实现仅用于准确率。

#### `twilight/pyimpl/top_k.py`

##### 功能、函数与调用关系

该文件实现基于真实/给定 attention logits 的布尔 TopK mask。

- `top_k`：每行选最多 `token_budget` 个最大元素并返回同形状 bool mask。
- `oracle_topk_selector`：语义化 wrapper，直接调用 `top_k`。

`attention.py` 用 `oracle_topk_selector` 做 Oracle baseline；`sparq.py` 和 `double_sparse.py` 复用 `top_k` 对估计权重选择 token。

##### 逐行/代码块解读

- 第 1-4 行：说明这是 Oracle TopK Python 实现并导入 Torch。
- 第 7-11 行：`k=min(token_budget, seq_len)`，避免预算大于序列。
- 第 12 行：沿最后一维取 top-k indices，丢弃 values。
- 第 13-14 行：创建全 False mask并原地 scatter 为 True。
- 第 16 行：返回形状与 logits 相同的 bool tensor。
- 第 19-29 行：`oracle_topk_selector` 仅提供 selector 文档语义并转调 `top_k`。

##### 注意与当前实现边界

- 没有把 `token_budget` 下界限制为 0；负值会由 `Tensor.topk` 抛错。
- “Oracle”表示直接看真实完整 attention logits，不能作为高效 selector 实现。

#### `twilight/pyimpl/top_p.py`

##### 功能、函数与调用关系

这是未归一化 attention logits 到 Twilight CUDA Top-p kernel 的高层桥接层。

- `top_p_unnormalized`：FP32 softmax 后返回 bool mask。
- `top_p_unnormalized_return_indices`：FP32 softmax 后返回紧凑 indices 和逐行 counts，可选映射输入 indices/限制输出容量。
- `top_p_unnormalized_return_indices_out`：同上，但写入调用方预分配 buffer。

`attention.py` 的 TOP_P pruner、`model/llama_patch.py`、`model/wrapper.py` 和三个 efficiency benchmark 都调用本文件。底层接口由 `twilight/kernel/__init__.py` 从 `kernel/cuda/sampling.py` 导出。

##### 逐行/代码块解读

- 第 1-5 行：导入 Torch/NN。
- 第 7-11 行：导入 FP32 mask、indices、indices-out CUDA wrapper；当前高层没有 FP16 softmax输出路径。

`top_p_unnormalized`（第 14-19 行）：

- 第 15-17 行：沿 token 维用 FP32 softmax，把 logits 变成概率。
- 第 18-19 行：调用底层 threshold kernel并返回 bool mask。

`top_p_unnormalized_return_indices`（第 22-33 行）：

- 第 28-30 行：相同 FP32 softmax。
- 第 31-33 行：传递阈值、可选 `input_indices` 和最大输出长度；若给 input indices，底层应输出映射后的原始索引而非局部位置。

`top_p_unnormalized_return_indices_out`（第 36-48 行）：

- 第 43-45 行：softmax。
- 第 46-48 行：把预分配 indices/counts 交给 raw CUDA 模块原地写入并原样返回。

##### 注意与当前实现边界

- 高层每种接口都包含 FP32 softmax；若输入已经是概率（如 `model/wrapper.py::TopPWrapper` 当前生成的数据），会发生第二次 softmax，语义改变。
- Top-p 的具体选择顺序/阈值边界由 CUDA `sampling.cu` 决定，本文件不排序也不做参数范围校验。

#### `twilight/pyimpl/elementwise_threshold.py`

##### 功能、函数与调用关系

该文件提供按 logits 差值阈值裁剪以及一个实验性的分层阈值预算估算。

- `elementwise_threshold`：保留权重大于“token 0 权重减 threshold”的 token。
- `hierarchical_elementwise_threshold`：尝试在多个固定阈值 mask 中按目标 rate 为每个 head 选一层。

`attention.py` 的 `WeightPrunerType.THRESHOLD` 调用第一个函数。第二个函数当前没有仓库调用方。

##### 逐行/代码块解读

- 第 1-6 行：导入 Torch/NN/NumPy；`nn` 当前未使用。

`elementwise_threshold`（第 9-18 行）：

- 第 15 行：取序列 token 0 的 logit 作为每个 batch/head/query 的基准。
- 第 17 行：保留 `weight > first_weight-threshold`；使用严格大于。
- 第 18 行：返回 bool mask。

`hierarchical_elementwise_threshold`（第 21-62 行）：

- 第 22-30 行：读取 head/seq，固定阈值 `[5,7,9,10,11,12]`，分配所有候选 mask 和最终 mask。
- 第 32-36 行：为各阈值保存 token 数差分与按 head 权重。
- 第 38-41 行：意图为每个阈值计算底层 mask并统计各 head token 数。
- 第 43-45 行：把相邻阈值选中数做差，最后一段是仍未选中的 token。
- 第 47-50 行：差分 token 数乘 `exp(-threshold)`，倒序写入并累积。
- 第 51 行：根据累计权重与 `1-rate` 比较，决定每个 head 选择哪一级阈值。
- 第 58-60 行：逐 head 从预计算 mask 中复制选中的一级。
- 第 62 行：返回最终 mask。

##### 注意与当前实现边界

- 第 39 行调用 `twi(attn_weights, t)`，但本文件和导入中都没有名为 `twi` 的函数；当前执行该函数会 `NameError`，很可能原意是 `elementwise_threshold`。
- 第 60 行用 `weights_pos[i]` 索引长度为 6 的第一维；若计算结果为 6 会越界，当前没有 clamp。
- `elementwise_threshold` 以 token 0 而非行最大值为基准；注释中保留了曾考虑 `max` 的写法。这依赖 attention sink 假设。

#### `twilight/pyimpl/quantize.py`

##### 功能、函数与调用关系

该文件实现量化后立即反量化的“伪量化”估计器，用于模拟低比特 K 对权重排序的影响，而不是节省实际存储/算力。

- `min_max_per_token_quant_kv`：逐最后一维做非对称 min-max qbit 量化/反量化。
- `max_per_token_quant_kv`：逐最后一维做对称 abs-max 量化/反量化；可先做跨 token 平移 smooth。
- 主入口：打印一个 4-bit min-max 示例。

`attention.py` 调用两种量化；`double_sparse.py` 调用 min-max 对 K label 量化。

##### 逐行/代码块解读

`min_max_per_token_quant_kv`（第 4-13 行）：

- 第 5-6 行：每个最后维向量的最大/最小值。
- 第 8-10 行：无符号整数范围 `[0,2^qbit-1]`；range 至少 clamp 到 `1e-9`，计算 scale。
- 第 11 行：`zeros=-min_val`，即先把最小值平移到 0。
- 第 13 行：平移/除 scale/round/clamp，再乘 scale并撤销平移，返回浮点反量化值。

`max_per_token_quant_kv`（第 16-25 行）：

- 第 17-19 行：smooth 时对倒数第二维取最小值并从所有 token 中减去，使每 channel 沿 token 的最小值变 0。
- 第 20-21 行：1-bit 特例直接取 sign。
- 第 22-24 行：取每个最后维向量的绝对最大值，除以 `2^qbit-1` 得 scale，并尝试下限为 `1e-6`。
- 第 25 行：round 后反量化。

示例（第 28-49 行）：构造 15 个浮点数并打印 4-bit min-max 结果。

##### 注意与当前实现边界

- 第 24 行新建的 `torch.tensor(1e-6)` 未显式跟随 `kv` 的 device/dtype；GPU/低精度输入下依赖 PyTorch 对标量 tensor 的设备/类型处理，使用 tensor 自身的 `clamp_min` 会更稳健。
- `qbit` 没有正数校验；0 或负数会令整数范围/scale 不合理。

#### `twilight/pyimpl/streaming_llm.py`

##### 功能、函数与调用关系

该文件实现 StreamingLLM 风格 selector。

- `streaming_selector`：保留序列开头的 attention sink token，以及结尾最近 `token_budget` 个 token。

仅由 `attention.py` 的 STREAMING 分支调用。

##### 逐行/代码块解读

- 第 1-4 行：模块说明和 Torch 导入。
- 第 7-16 行：函数接收完整权重只为复用其形状/设备，实际不查看权重值。
- 第 18 行：创建全 False mask。
- 第 19-21 行：要求四维 `[B,H,1,S]` 形状，但实际只检查维数，没有检查 query length 是否为 1。
- 第 23 行：开头 `num_sinks` 个位置设 True。
- 第 24 行：最后 `token_budget` 个位置设 True；与 sinks 重叠时 union 后总数小于两者之和。
- 第 25 行：返回 mask。

##### 注意与当前实现边界

- `token_budget` 在这里表示“最近窗口大小”，不包含 sink 数；实际最大保留数约为 `num_sinks+token_budget`。
- 负的 `token_budget` 会产生意外的 Python slice 语义，函数没有参数校验。

#### `twilight/pyimpl/tidal_decode.py`

##### 功能、函数与调用关系

该文件尝试实现 Tidal Decode 的跨层 mask 复用。

- `tidal_decode_selector`：在 reselection layer 计算 TopK并缓存 mask，同时让该层用 full attention；其他层复用最近缓存的 mask。

只由 `attention.py` 的 TIDAL_DECODE 分支调用，并通过共享 `LocalState` 保存 `td_mask_buffer`。

##### 逐行/代码块解读

- 第 1-3 行：导入 Torch 和列表类型。
- 第 6-17 行：函数签名带 state、真实权重、预算、层号和重选层列表；文档写返回 mask。
- 第 19-24 行：若当前是重选层，对真实权重 TopK，scatter 成 bool mask并挂到 `state.td_mask_buffer`。
- 第 25-27 行：返回 `(None, 全True mask)`，意图是重选层只负责产生供后层使用的 mask，自身仍做 full attention。
- 第 29-30 行：非重选层要求 buffer 已存在，返回 `(None, cached mask)`。

##### 注意与当前实现边界

- 实际返回的是二元 tuple，不是文档所说的单 mask；`attention.py` 也没有解包，导致当前主链接口不兼容。
- 缓存只有一个全局 mask，不按 batch/query/layer组区分；它依赖模型按层顺序串行执行，并在下一次 reselection 时覆盖。

#### `twilight/pyimpl/sparq.py`

##### 功能、函数与调用关系

该文件是 SparQ Attention 的朴素准确率实现。

- `sparq`：按 query 绝对值选择 top-r channel，只用这些 channel 估计所有 token 的 attention logits。
- `sparq_selector`：对估计 logits 再做 token TopK。

`attention.py` 的 SPARQ selector 调用 `sparq_selector`；本文件复用 `top_k.py`。

##### 逐行/代码块解读

- 第 1-7 行：说明论文来源以及该实现只验证准确率、仍会计算全序列估分。
- 第 9-13 行：导入 Torch/math 和本地 TopK。

`sparq`（第 16-44 行）：

- 第 21-29 行：`r` 通常是 head dim 乘 channel 压缩率；返回的是 estimated weights，文档里的“mask”字样不准确。
- 第 31-33 行：对 query 取绝对值，沿 channel 选 top-r indices。
- 第 35 行：gather partial query。
- 第 36-37 行：把 channel indices 沿 KV token 维重复，gather 每个 token 的相同 channel。
- 第 39-43 行：partial QK matmul，但仍除以原始 `sqrt(head_dim)` 而不是 `sqrt(r)`。
- 第 44 行：返回 `[B,H,Q,S]` 估计 logits。

`sparq_selector`（第 47-59 行）：先调用 `sparq`，再用统一 `top_k` 返回 bool token mask。

##### 注意与当前实现边界

- `r` 大于 head dim 或小于 0 会由 `torch.topk` 报错；没有显式校验。
- `repeat` 会实际复制 channel index tensor，符合朴素准确率定位，但不是高效实现。

#### `twilight/pyimpl/double_sparse.py`

##### 功能、函数与调用关系

该文件实现 Double Sparsity 的 channel 配置注入、重排辅助、部分 channel QK 估分和 TopK selector。

- `init_model_channel_config`：遍历 Llama/Mistral Attention，按层加载预分析的 heavy-channel 排名。
- `permute_channel_config`：对 channel id 做偶/奇交错位置变换。
- `double_sparse`：用 Triton gather kernel提取 Q/K 的 r 个 heavy channels，可选量化 K label，再计算估计 logits。
- `double_sparse_selector`：支持固定预算或按序列比例预算，估分后 TopK。
- `double_sparse_selector1`：固定预算版本，可先给估计 logits 加 attention mask。

`attention.py` 在安装 DS 时调用 `init_model_channel_config`，decode 时调用 `double_sparse_selector`。本文件调用 `quantize.py`、`top_k.py` 和经 `twilight.kernel` 导出的 Triton `get_label_tensor`。

##### 逐行/代码块解读

- 第 1-5 行：说明这是准确率用朴素 Double Sparsity。
- 第 7-19 行：导入 Torch/math/Optional、Transformers 两种 Attention、本地量化/TopK 和 Triton gather wrapper。

`init_model_channel_config`（第 22-38 行）：

- 第 25 行：把 `selected_channel='k'` 转成 `.k_proj`（或 q/v 对应名字）。
- 第 26-28 行：遍历全部 named modules，只处理 Llama/Mistral Attention。
- 第 30-31 行：从模块名第三段解析层号，拼出 channel-config JSON key。
- 第 32-36 行：把每个 head 排名前 `heavy_channel_num` 的 channel 转 tensor、contiguous并搬到默认 CUDA，挂为 `module.sorted_channel`。
- 第 38 行：返回原模型对象（已原地修改）。

`permute_channel_config`（第 41-44 行）：

- 读取 head 数/维度；映射公式 `(i*2)%D + floor(i*2/D)` 在偶数 D 下把前半映射到偶数位、后半映射到奇数位，适配某种 interleaved channel 布局。

`double_sparse`（第 47-95 行）：

- 第 64 行：K 预期形状 `[B,H,S,D]`。
- 第 65-72 行：分配 Q label `[B,H,r]`，把 decode Q reshape 成 `[B,H,D]`，按各 head `sorted_channel` 用 Triton gather。
- 第 73-83 行：把 K 转成 `[B*S,H,D]`，提取相同 heavy channels 到 `[B*S,H,r]`。
- 第 84-85 行：恢复 Q `[B,H,1,r]` 和 K `[B,H,S,r]`。
- 第 86-87 行：`quant_bit<32` 时对 K label 做 min-max 伪量化；32 被约定为不量化。
- 第 89-93 行：partial QK 估分，仍除以原始 head dim 的平方根。
- 第 95 行：返回 estimated weights。

`double_sparse_selector`（第 98-121 行）：

- 第 112 行：读取序列长度。
- 第 114-116 行：若 `sparsity_rate!=-1`，用 `int(seq_len*rate)` 覆盖固定 token budget。
- 第 118-121 行：估分并调用统一 TopK。

`double_sparse_selector1`（第 124-148 行）：

- 第 138-140 行：同样先估分。
- 第 142-146 行：存在 attention mask 时相加，再尝试把结果下限限制到 dtype 最小值。
- 第 148 行：固定预算 TopK。

##### 注意与当前实现边界

- channel 配置 key 和层号解析强依赖 `model.layers.N.self_attn.{k|q}_proj` 命名；`.cuda()` 也忽略模型实际 device，可能在多卡/非默认卡下错位。
- `get_label_tensor_kernel` 假设 `sorted_channel` 最后一维 contiguous；初始化路径显式 `.contiguous()` 满足该假设，外部直接调用需自行保证。
- `double_sparse_selector1` 当前没有仓库调用方；其第 145 行新建 CPU 标量 tensor，与 CUDA estimated weights 的设备兼容性需结合 PyTorch 版本验证。
- `sparsity_rate` 没有限制在 `[0,1]`；很小正比率可得到预算 0。

#### `twilight/pyimpl/quest.py`

##### 功能、函数与调用关系

该文件实现 Quest 的 query-aware chunk 上界估分。对每个 query channel，根据正负号选择每个 chunk 中能使点积最大的 K channel 值，从而用 chunk 分数选 token 块。

- `quest_selector`：支持固定 token budget 或按序列比例计算预算。
- `quest_selector1`：只支持固定 token budget；主体与前者近乎重复。

`attention.py` 调用 `quest_selector`；`quest_selector1` 当前没有仓库调用方。传入的 `model` 主要用于 `num_heads` 和调试注释中的 layer id。

##### `quest_selector` 逐行/代码块解读

- 第 1-10 行：论文、准确率实现定位和参考实现链接。
- 第 13-14 行：导入 Torch/Optional。
- 第 17-30 行：签名支持 model、Q/K、chunk size、token budget、sparsity rate 和 attention mask；返回 bool mask。
- 第 32-34 行：注释要求 decode 且较后层，取 batch size；函数本身没有 assert q_len=1/layer>=2，约束由主调度保证。
- 第 36-39 行：`sign` 在 Q 正时为 +1，否则 -1；`postive_query=abs(Q)`，K 同 channel 乘该符号。这样 chunk max 选择对当前 Q 符号最有利的 K 值。
- 第 41-46 行：读取序列长度；非 -1 sparsity rate 覆盖 token budget。
- 第 48-65 行：若长度不能整除 chunk size，在 token 维补 dtype 最小值，避免 padding 被 chunk max 选中。
- 第 67-74 行：reshape 为 `[B,H,num_chunks,chunk_size,D]` 并沿 chunk token 维取 max。
- 第 76-77 行：K max 为 FP32 时也把正 query 转 FP32。
- 第 79-84 行：`[B,H,1,D] @ [B,H,D,num_chunks]` 得 chunk estimated weights。
- 第 88-92 行：预算/整除 assert 被注释。
- 第 93-95 行：token budget 向上取整成 chunk budget，并至少为 1。
- 第 97-99 行：不超过实际 chunk 数地选 TopK chunks。
- 第 103-108 行：每个 chunk id 展开为连续 `chunk_size` 个 token id并压平。
- 第 110-118 行：按 padding 后长度创建 mask并 scatter selected token ids。
- 第 120-122 行：裁掉 padding，返回原序列长度 mask。

##### `quest_selector1` 逐行/代码块解读

- 第 125-137 行：签名去掉 sparsity rate，保留未使用的 attention mask。
- 第 139-186 行：与第一函数的符号变换、padding、chunk max 和估分相同。
- 第 190-201 行：同样把 token budget 向上取整为 chunk budget并 TopK。
- 第 205-224 行：恢复 token ids、scatter mask、裁 padding并返回。

##### 注意与当前实现边界

- 两个函数都接收 `attention_mask` 但完全没有使用；主 attention 又强制 decode mask 为 None。
- 预算按 chunk 向上取整，所以实际选中 token 数可比 `token_budget` 多至 `chunk_size-1`，末尾 padding裁掉时也可能少一些。
- 构造 padding的 `torch.ones` 没显式 dtype，但乘 FP16/BF16 最小值后与 `max_key` 拼接时会发生类型提升/转换，需注意实际 dtype；更直接的 full_like 可减少歧义。
- `quest_selector` 与 `quest_selector1` 大量重复，未来修改一处时容易行为漂移。

#### `twilight/pyimpl/snap_kv.py`

##### 功能、函数与调用关系

该文件实现 SnapKV prefill cache 压缩，并保留一份面向 Llama generation 输入准备的替代函数。

- `snap_kv_compression`：用 prompt 末尾 observation window 对更早 token 的注意力聚合打分，选出历史重要 K/V，再与最近窗口拼接。
- `prepare_inputs_for_generation_llama_snapkv`：按 compressed cache 的逻辑长度裁剪新输入并构造 position ids/model inputs。

`attention.py` prefill 在 compressor 为 SNAP_KV 时调用第一个函数。第二个函数已被 `attention.py` 导入，但安装 monkey patch 的代码目前注释掉。

##### `snap_kv_compression` 逐行/代码块解读

- 第 1-7 行：论文来源与准确率实现说明。
- 第 9-15 行：导入 Torch、池化/NN/math 和 Transformers cache 类型；`time`、`DynamicCache` 当前未使用。
- 第 18-26 行：默认 observation window 64，总 cache 容量 320，池化 kernel 5，avg pooling。
- 第 27-31 行：只允许 prefill Q/K 等长；prompt 比容量短时不压缩直接返回原 K/V。
- 第 32-35 行：只用最后 `window_size` 个 query 对全部 K 计算 scaled dot-product logits。
- 第 36-45 行：构造 observation window 内的 causal mask；未来 token位置为 dtype 最小值，合法位置为 0。
- 第 46 行：只把该 mask 加到 logits 的“末尾 query × 末尾 key”子块；历史 key 对所有 observation query 可见。
- 第 48-50 行：FP32 softmax后转回 Q dtype。
- 第 51 行：对 observation query 维求和，得到每个较早 token 的累计受关注度；排除最近 window。
- 第 52-65 行：对时间维做 avg/max pool平滑；其他字符串抛 ValueError。
- 第 68-69 行：选择 `max_capacity_prompt-window_size` 个历史 token，并扩展 index 到 head dim。
- 第 70-75 行：在原 K/V 的历史区间按相同 indices gather。
- 第 76-79 行：保留完整最近 window，与压缩历史拼接，总长度等于 `max_capacity_prompt`。
- 第 80 行：返回压缩 K/V。

##### `prepare_inputs_for_generation_llama_snapkv` 逐行/代码块解读

- 第 83-90 行：签名仿 Transformers generation hook。
- 第 91-93 行：新 generation 开始时，把每层 attention 的 `kv_seq_len` 复位为 0。
- 第 94-103 行：有 past cache 时，Cache 对象从 API 读取长度；legacy/其他格式从第 0 层自定义 `kv_seq_len` 读取。
- 第 104-114 行：三种输入情形下只保留尚未 cache 的 input ids。
- 第 116-122 行：若将超最大 cache长度，裁 attention mask 左侧。
- 第 124-130 行：没有 position ids 时由 attention mask 累积构造；有 past 时只保留新 token 对应尾部。
- 第 132-136 行：inputs_embeds 仅首次使用，之后使用 input ids。
- 第 138-146 行：组装 position、past、use_cache、attention_mask并返回。

##### 注意与当前实现边界

- `window_size` 必须不大于 q_len，且 `max_capacity_prompt>window_size`；函数没有显式检查这些参数组合。
- `topk(max_capacity_prompt-window_size)` 默认假设历史 token 数足够；进入压缩分支时 `q_len>=capacity` 基本满足，但非默认负/异常参数仍可破坏。
- generation hook 当前没有安装，因此正常 `enable_sparse_attention` 路径主要依赖 `attention_forward` 中 `kv_seq_len`/cache API 的处理。
- 不同 Transformers 版本的 Cache 属性（如 `seen_tokens/get_max_length`）可能变化，这段函数带明显版本耦合。

#### `twilight/pyimpl/h2o.py`

##### 功能、函数与调用关系

这是一个 0 字节占位文件，没有函数、类、导入或调用关系。文件名暗示计划容纳 H2O attention 策略，但当前 `state.py` 枚举和 `attention.py` 分发中也没有 H2O 分支。

##### 逐行/代码块解读

- 文件为空，没有可执行代码。

### `twilight/kernel`

该目录是 Python 到自定义 CUDA/Triton kernel 的接口层。CUDA Top-p 通过 FlashInfer JIT 机制加载本仓库 `csrc/src/sampling.cu`；Triton 子目录提供 channel gather、INT8 batched GEMV 和实验性 per-block INT8 QK/softmax kernel。

#### `twilight/kernel/__init__.py`

##### 功能、函数与调用关系

这是 `twilight.kernel` 的公共导出入口，没有自定义函数。

- 显式导出 CUDA sampling 的 FP16/FP32 mask、indices、indices-out 六个接口。
- 用 `from .triton import *` 导出 `get_label_tensor`、`bgemv_int8`、`qk_int8_per_block`。

`pyimpl/top_p.py` 从这里取得 sampling；`pyimpl/double_sparse.py` 从这里取得 channel gather。

##### 逐行/代码块解读

- 第 1-8 行：从 `cuda/sampling.py` 重导出六个 Top-p API。
- 第 10 行：导入 `triton/__init__.py` 的非下划线名字；由于后者未定义 `__all__`，三个显式导入的 wrapper 会暴露到本包。

#### `twilight/kernel/cuda/env.py`

##### 功能、函数与调用关系

该文件只计算自定义 CUDA 扩展的仓库路径常量，供 `cuda/sampling.py` 的 JIT 编译使用。

- `TWILIGHT_HOME`：仓库根。
- `TWILIGHT_CSRC_DIR`：`csrc/src`。
- `TWILIGHT_INCLUDE_DIR`：`csrc/include`。

##### 逐行/代码块解读

- 第 1 行：导入 pathlib。
- 第 3 行：从当前 `twilight/kernel/cuda/env.py` 向上三级得到 Twilight 根目录。
- 第 4-5 行：拼出源文件和头文件目录的 Path 对象。

#### `twilight/kernel/cuda/sampling.py`

##### 功能、函数与调用关系

该文件是 Top-p CUDA 扩展的延迟加载、FlashInfer custom-op 注册和公共 Python wrapper。

- `get_sampling_module`：首次调用时选择预编译 `_kernels` 或 JIT 编译 `sampling.cu`，注册四个 custom op并缓存 namespace。
- `_to_tensor_scalar_tuple`：把 `top_p` 统一拆成“可选逐行 tensor + 标量 fallback”。
- `top_p_fp16/32_return_mask`：返回同形状 bool mask。
- `top_p_fp16/32_return_indices`：分配紧凑 indices/counts，再调用 out 版本。
- `top_p_fp16/32_return_indices_out`：直接调用 raw 扩展写入预分配输出。

调用链：

```text
pyimpl/top_p.py
  -> kernel/__init__.py
    -> cuda/sampling.py
      -> flashinfer.jit.load_cuda_ops("twilight_sampling", csrc/src/sampling.cu)
      -> C++/CUDA 导出的 top_p_fp{16,32}_return_{mask,indices}
```

##### 模块加载与 custom op 逐行/代码块解读

- 第 1-15 行：FlashInfer 版权/许可证声明。
- 第 17-20 行：导入 namespace、类型和 Torch。
- 第 22-27 行：从 FlashInfer JIT 取得预编译判断、加载器和其 include/source 路径。
- 第 29-32 行：导入当前 CUDA stream 获取与 custom-op 注册工具。
- 第 34 行：导入本仓库 csrc/include 路径。
- 第 37 行：模块级 cache 初始为 None。

`get_sampling_module`（第 40-153 行）：

- 第 42-48 行：只在第一次调用加载；有预编译 op 时导入同包 `_kernels`，否则走 JIT。
- 第 49-59 行：JIT 模块名 `twilight_sampling`，编译 `sampling.cu`，include 同时包含 FlashInfer include/source 与 Twilight include。
- 第 62-80 行：注册 FP16 mask custom op。它声明不修改输入，按 probs 创建 bool mask，调用 raw module并传当前 CUDA stream，返回 mask。
- 第 82-100 行：FP32 mask 版本，流程相同。
- 第 102-121 行：注册 FP16 indices custom op；按 probs 全长分配 int32 indices，按去掉最后维的形状分配 counts，再调用 raw op。
- 第 123-142 行：FP32 indices custom op，流程相同。
- 第 144-151 行：用 `SimpleNamespace` 同时保存 raw module 与四个注册后的 callable。
- 第 153 行：返回缓存，后续调用不重新编译/注册。

##### 公共 wrapper 逐行/代码块解读

`_to_tensor_scalar_tuple`（第 156-160 行）：

- 若 top-p 是 Tensor，返回 `(tensor, 0)`；否则返回 `(None, scalar)`。raw kernel 可据此支持逐行阈值或全局阈值。

mask 接口（第 163-178 行）：

- FP16/FP32 两个函数取得 lazy module，并把统一拆分后的两个阈值参数展开给注册 custom op。

动态 indices 接口（第 181-216 行）：

- 第 187-191、206-210 行：文档强调 counts 不会被 `max_output_len` 裁剪；若需要完整 indices，容量必须不小于最大 count。
- 第 192-197 行：FP16 版本按显式容量或完整序列长分配 indices/counts，然后转调 out 版本。
- 第 211-216 行：FP32 同理。

预分配 out 接口（第 219-273 行）：

- 第 226-229、255-258 行：再次声明 count 不受输出容量裁剪。
- 第 230-235 行：取得 lazy module、拆阈值，访问 `_raw_module`；没有 raw 模块就抛 RuntimeError。
- 第 236-244 行：FP16 raw op接收 probs、预分配输出、可选 input indices、阈值 tensor/标量和当前 stream。
- 第 245 行：原样返回两个输出 buffer。
- 第 259-273 行：FP32 版本相同。

##### 注意与当前实现边界

- public `return_indices` 当前主动转调 raw `*_out`，而不是 namespace 中注册的 allocating custom op；因此动态接口虽“分配输出”，实际 kernel 路径仍是 raw out 调用。
- out 接口不会检查输出 shape/dtype/device/capacity；约束由调用方和 C++/CUDA 实现承担。
- `_sampling_module` 是进程级全局 cache；首次 JIT 失败会直接向上抛异常。

#### `twilight/kernel/triton/__init__.py`

##### 功能、函数与调用关系

这是 Triton kernel 的聚合入口，没有自定义函数。

- 导出 `channel.py::get_label_tensor`；
- 导出 `bgemv_int8.py::bgemv_int8`；
- 导出 `qk_int8_per_block.py::qk_int8_per_block`。

上层 `kernel/__init__.py` 再把它们暴露为 `twilight.kernel.*`。

##### 逐行/代码块解读

- 第 1 行：模块说明。
- 第 3-6 行：导入三个 Python launch wrapper；导入模块时也会加载 Triton 和定义相应 JIT kernel。

#### `twilight/kernel/triton/channel.py`

##### 功能、函数与调用关系

该文件按每个 attention head 的 channel index，从三维 tensor 中 gather heavy-channel label。

- `get_label_tensor_kernel`：每个 Triton program处理一个 `(token, head)`，载入该 head 的若干 channel。
- `get_label_tensor`：提取 stride/shape并以 `(L,H)` grid 启动 kernel。
- `test_get_label_tensor`：构造随机 channel并逐 token/head 与 PyTorch indexing 比较。

`pyimpl/double_sparse.py` 用它提取 Q/K label。

##### 逐行/代码块解读

- 第 1 行：来源说明。
- 第 3-8 行：导入 Torch/Triton；`math/random` 当前未使用。

`get_label_tensor_kernel`（第 11-53 行）：

- 第 12-25 行：接收 X/channel/out 指针、各 stride和编译期 heavy channel 数。
- 第 27-29 行：二维 grid 的 program id 分别是 token 与 head。
- 第 31-35 行：从该 head 的 channel 行读取 `[0,r)` 个索引。
- 第 37-44 行：用 token/head/channel stride算 X 地址，载入 r 个值。
- 第 46-53 行：按输出 stride写到 `[token,head,:]`。

`get_label_tensor`（第 56-79 行）：

- 第 58-64 行：读取 `L,H`、三个 tensor stride，grid 为 `(L,H)`。
- 第 66-79 行：传指针、stride和编译期 r 启动 kernel；输出原地写入，没有显式 return。

测试（第 82-110 行）：

- 第 84-93 行：构造 `[10,3,8]` X，每 head 随机选 4 channel并分配输出。
- 第 95 行：运行 kernel。
- 第 97-106 行：搬回 CPU，双循环与 `X[i,j,channel[j]]` 比较；发现差异返回 False，否则 True。
- 第 109-110 行：直接运行文件时启动测试。

##### 注意与当前实现边界

- kernel 接收 `stride_channel_c`，但第 33 行地址计算没有乘它，隐含 channel 最后一维 stride 为 1；Double Sparse 初始化使用 `.contiguous()` 满足此条件。
- 没有对 channel id 范围做 mask；越界索引会读非法地址。

#### `twilight/kernel/triton/bgemv_int8.py`

##### 功能、函数与调用关系

该文件计算 FP query label 与 INT8 K label 的 batched multi-head GEMV，并按每个 token/head 的 K scale 反量化输出。

- `bgemv_int8_kernel`：一个 program负责一个 `(batch,head)`，一次算该 head 对全序列的点积。
- `bgemv_int8`：推导形状/stride并 launch。
- `torch_bgemv_int8`：PyTorch reference。
- `test_bgemv_int8`：warmup、计时和数值比较。

它由 `triton/__init__.py` 导出，但在本文目标 Python 范围内没有生产调用方；当前主要是独立实验 kernel。

##### 逐行/代码块解读

- 第 1 行：来源说明。
- 第 3-9 行：导入计时、Torch/Triton；`math/random` 当前未使用。

`bgemv_int8_kernel`（第 12-50 行）：

- 第 13-21 行：传 Q/K/scale/out 指针、stride，以及编译期 head 数、heavy channel数、序列长。
- 第 23-24 行：二维 grid 选 batch/head。
- 第 26-27 行：Q 地址是当前 batch/head 的 r 个 channel。
- 第 29-30 行：K 地址覆盖当前 batch 的所有 `N_CTX` token、当前 head、r 个 channel；`BLOCK_HMODEL` 实际传 H，用于跨 token 跳过所有 heads。
- 第 32-33 行：K scale 地址覆盖每个 token 的当前 head。
- 第 35-40 行：加载 Q/K/scale。
- 第 42-46 行：沿 channel 求点积，再乘 K scale。
- 第 48-50 行：写 `[B,H,N_CTX]`。

`bgemv_int8`（第 53-75 行）：

- 第 55-61 行：Q 形状给出 B/H/r，K 第一维为 `B*N_CTX`；读取全部 stride。
- 第 63-73 行：以 `(B,H)` launch并传 H/r/N_CTX 编译常量。
- 第 75 行：返回调用方预分配 Out。

`torch_bgemv_int8`（第 78-93 行）：

- 第 83-84 行：用 scale 反量化 K，Q 转 FP16。
- 第 86-89 行：reshape 成 `[B,H,1,r] @ [B,H,r,S]`并 squeeze query 维。

测试（第 96-137 行）：

- 第 98-108 行：构造 B=32/H=32/S=2048/r=8，按 abs-max/127 量化 K 到 int8。
- 第 110-113 行：Q 量化尝试被注释，并注明尚不能工作。
- 第 116-128 行：warmup 10 次，同步后运行 1000 次并用 wall-clock 求平均。
- 第 130-134 行：与 PyTorch reference 打印最大/平均误差并断言容差。
- 第 136-137 行：脚本入口。

##### 注意与当前实现边界

- Triton kernel没有显式 boundary mask，依赖 `N_CTX`、`HEAVY_CHANNEL_NUM` 作为编译期范围且实际内存完全匹配。
- reference 的 K scale 广播和布局与 wrapper强绑定；传入非预期 stride/shape虽有 stride参数，也未必满足地址公式中的扁平假设。

#### `twilight/kernel/triton/qk_int8_per_block.py`

##### 功能、函数与调用关系

该文件来源于 SageAttention，名字和参数意图是对 per-block INT8 Q/K 计算在线 softmax/LSE。当前实现没有 V 输入，且 accumulator 没有累积概率加权值，因此更像未完成的实验骨架。

- `_attn_fwd_inner`：按 K block 扫描，计算缩放 QK并在线更新 softmax max/denominator。
- `_attn_fwd`：为 batch/head/query block建立指针、调用 inner、写 output/LSE。
- `qk_int8_per_block`：处理 HND/NHD layout，分配输出并 launch。

它经 `triton/__init__.py` 导出，但在仓库 Python 调用搜索中没有调用方。

##### 逐行/代码块解读

- 第 1-17 行：来源和许可证。
- 第 19-21 行：导入 Torch/math/Triton；Python `math` 当前未使用。

`_attn_fwd_inner`（第 23-48 行）：

- 第 30-33 行：从 0 到 `kv_len` 按 `BLOCK_N` 扫描，尾 block 用 mask。
- 第 34-36 行：加载 K 与该 K block 的 scale；`tl.dot(q,k)` 后乘 Q/K scale，转 FP32。
- 第 37-45 行：在线 softmax：更新行最大 `m_i`、用 `exp2` 得概率、累加分母 `l_i` 并重标旧分母。
- 第 46-47 行：K 指针前进一个 token block，scale 指针前进一个 block。
- 第 48 行：原样返回 `acc` 以及更新后的分母/最大值；函数中从未修改 `acc`。

`_attn_fwd`（第 50-96 行）：

- 第 62-68 行：program id 选择 query block/head/batch，计算 Q/K scale block offset；GQA 用 `head // num_kv_groups` 映射 KV head。
- 第 70-77 行：建立 Q、K、scale、output 指针。
- 第 79-81 行：初始化在线 softmax状态；`l_i` 初始为 1，`acc` 全 0。
- 第 83-89 行：加载 Q，调用 inner 扫完整 K。
- 第 90-91 行：`acc/l_i` 仍为全 0，写入 q 同形状 output。
- 第 93-96 行：可选写 `log2(l_i)+m_i` 作为 base-2 LSE。

`qk_int8_per_block`（第 98-143 行）：

- 第 99-103 行：固定 Q/K block 为 128/64，输出 shape 与 Q 相同。
- 第 105-118 行：分别从 HND `[B,H,N,D]` 或 NHD `[B,N,H,D]` 读取形状和 stride。
- 第 119-120 行：其他 layout 抛 ValueError。
- 第 122-123 行：head dim 与 GQA group 数。
- 第 125-128 行：按需分配 GPU LSE，否则创建 CPU 空 tensor作为占位参数。
- 第 130-141 行：grid 是 query blocks × query heads × batch；按 head dim选择 warp/stage数启动 kernel。
- 第 143 行：返回 `(o,lse)`。

##### 注意与当前实现边界

- `acc` 从头到尾没有使用 V 或其他值更新，所以输出 `o` 按当前代码恒为 0（忽略异常数值情况）。只有 LSE 计算有实际信息。
- 使用 `exp2(qk)`，但没有看到把自然指数 logits 乘 `log2(e)` 的转换；如果目标是标准 softmax，数值尺度与 `exp` 不同。
- `l_i` 初始化为 1 而常见在线 softmax初始化为 0；这也会改变 LSE。
- `STAGE/start_m/acc` 等参数保留了来源 attention kernel 的结构但未实际发挥完整作用，说明文件仍是实验/裁剪状态。

### `twilight/model`

该目录是另一条偏高性能/原型化的模型集成路径，与 `pyimpl/attention.py` 的准确率 patch 相互独立。它用 FlashInfer paged attention 和 FTKA GEMV 构造 wrapper，或直接替换 Llama Attention；当前没有在 LongBench/Passkey 脚本中启用。

#### `twilight/model/flashinfer_utils.py`

##### 功能、函数与调用关系

该文件封装 FlashInfer RMSNorm 和 RoPE。

- `flashinfer_rmsnorm_forward`：把 `[B,S,H]` 展平为二维，调用 FlashInfer RMSNorm，再恢复形状。
- `enable_flashinfer_rmsnorm`：遍历模型并 monkey-patch Llama/Mistral RMSNorm.forward。
- `apply_rope_inplace`：把定长 batch Q/K 转成 FlashInfer ragged 表示，原地应用 RoPE。

`llama_patch.py` 导入 `apply_rope_inplace` 并在 decode 调用；同时导入但当前没有调用 `enable_flashinfer_rmsnorm`。

##### 逐行/代码块解读

- 第 1-6 行：导入两种 Transformers RMSNorm、Torch、FlashInfer、MethodType 和 Optional。

`flashinfer_rmsnorm_forward`（第 9-16 行）：

- 第 10 行：读取 batch、sequence、hidden size。
- 第 11-15 行：展平前两维，传权重和模块 epsilon 给 `flashinfer.norm.rmsnorm`。
- 第 16 行：恢复三维输出。

`enable_flashinfer_rmsnorm`（第 19-26 行）：

- 第 20 行：打印替换提示。
- 第 21-25 行：扫描 `named_modules`；Llama/Mistral RMSNorm 都用 `types.MethodType` 绑定同一个替代 forward。
- 第 26 行：返回原地修改后的模型。

`apply_rope_inplace`（第 29-59 行）：

- 第 37-41 行：输入预期 `[B,S,H,D]`，展平为 `[B*S,H,D]`。
- 第 42-45 行：没有传 ragged `indptr` 时，按等长 batch构造 `[0,S,2S,...]` int32 指针。
- 第 46-47 行：单个 offset 扩展成每 batch 一个连续 tensor。
- 第 48-56 行：调用 FlashInfer 原地 RoPE，明确 `interleave=False` 并传 scale/theta。
- 第 57-59 行：恢复 Q/K 形状并返回；底层已经原地修改，返回值便于链式使用。

##### 注意与当前实现边界

- `offsets` 既非单元素也非 batch size 时，本函数没有显式 shape校验，交给 FlashInfer 报错。
- `indptr` 默认在每次调用新建 GPU tensor；高频 decode 若追求性能应复用元数据。

#### `twilight/model/llama_patch.py`

##### 功能、函数与调用关系

这是实验性的高性能 Llama attention patch：预分配随机 KV/label/量化 cache和 FlashInfer wrapper，decode 时可计 Quest、Twilight INT4估分/Top-p，再运行 paged attention。

- `attention_forward`：替代 LlamaAttention.forward；prefill 当前直接跳过，decode 调 FTKA/Twilight op 和 FlashInfer。
- `enable_twilight`：创建合成 cache/page metadata，给每层挂属性并 monkey-patch forward。

它调用 `flashinfer_utils.apply_rope_inplace`、`twilight.ops` 的 GEMV/label op、`pyimpl/top_p.py` 和 FlashInfer。本文目标范围内没有上游调用 `enable_twilight`，所以它不是当前 LongBench/Passkey 生效路径。

##### `attention_forward` 逐行/代码块解读

- 第 1-24 行：导入类型、系统/Torch/FlashInfer、Transformers Llama 类型，以及本地 RoPE/RMSNorm helper；大量导入（如 GC、time、多个 model output 类型）在当前有效代码中未使用。
- 第 27-32 行：从编译扩展 `twilight.ops` 导入 paged wrapper别名、FP/INT8 sparse GEMV 和 label gather；其中 TwilightWrapper/get_label_tensor 当前未使用。
- 第 34 行：导入高层 Top-p。
- 第 36-45 行：替代 forward 的 Transformers兼容签名。
- 第 47-51 行：读取 batch/query length并说明 Q/K/V 目标形状。
- 第 53-66 行：prefill 路径没有运行原 `flash_forward`，而是直接返回 `hidden_states,None,None`；原调用留在注释中。
- 第 70-78 行：仅 decode 时投影 Q/K/V，布局 `[B,1,H,D]`；后面 K/V 实际没有写入预分配 cache。
- 第 80-83 行：调用原地 RoPE；返回值未接收，但函数底层原地修改 Q/K。
- 第 85-89 行：强制无 attention mask，自定义 KV 长度加 1。
- 第 91 行：Q reshape 成 `[B,H,D]`。
- 第 93-106 行：CUDA event计时代码注释。
- 第 98 行：按 16-token chunk向下取整 page 数。
- 第 108-118 行：`use_quest` 时用 `k_label` 做 sparse GEMV，TopK 选 `B0/16` 个 label index，并把局部 `kv_seq_len` 改为 B0。
- 第 122-133 行：`use_twilight` 时在固定 `B0_indices` 上做 INT8/INT4风格估分，调用 `top_p_unnormalized(...,0.85)`，再把局部长度改成 B1。
- 第 138-143 行：重新构造 `[0,kv_seq_len)` 的 page table，设备硬编码 `cuda:0`。
- 第 145-156 行：每 token调用 FlashInfer `plan`；query/KV heads 都传 `self.num_heads`，page size 1，无位置编码。
- 第 157 行：运行 attention并 reshape `[B,1,hidden]`。
- 第 164 行：输出投影。
- 第 170-173 行：按需置 attention weights 为 None并返回；函数没有任何路径实际定义非 None `attn_weights`，但 `output_attentions=True` 时第 173 行会引用未定义变量。

##### `enable_twilight` 逐行/代码块解读

- 第 176-186 行：接收模型、三个开关、最大 page、初始长度、B0/B1和 batch。
- 第 187-190 行：从模型参数取 device/dtype，但 head 数/维度硬编码 32/128。
- 第 193-201 行：创建**随机** `[pages,2,1,H,D]` KV cache，不来自模型 prefill。
- 第 203-209 行：创建 16-token page/chunk 的随机 K label。
- 第 213-225 行：创建 packed 4-bit风格 int8 cache（最后维 D/2）。
- 第 227-241 行：创建随机 scale/zero。
- 第 243-248 行：创建 Quest chunk page table；设备硬编码 `cuda:0`。
- 第 250-253 行：创建 B0 token page table。
- 第 255 行：分配 128 MiB共享 workspace。
- 第 257-261 行：遍历 `model.model.layers` 并强制 self-attn 是 LlamaAttention。
- 第 263-280 行：给每层挂 KV 长度、开关、共享随机 cache、FlashInfer wrapper、B0/B1。
- 第 281-286 行：挂 Quest 与 B0 page-table tuple。
- 第 288-289 行：保存原 forward 到 `flash_forward`，再绑定实验 forward。

##### 注意与当前实现边界

- 这是原型/微基准性质代码，不是正确的端到端模型实现：prefill直接返回 hidden states；投影出的新 K/V 没有写入 cache；cache/label/scale/zero 全为随机合成数据。
- Quest 算出的 `label_index` 没有用于后续 `B0_indices`；Top-p 返回 mask也被丢弃；最终 attention读取 `[0,kv_seq_len)` 固定前缀。因此 selector/pruner 与 attention 数据流尚未接通。
- `use_flash_infer` 只保存为属性，forward中没有分支使用。
- `output_attentions=True` 会使 `attn_weights` 未定义。
- 每个 decode token重新 `plan` 并重新分配 page metadata，不能代表优化后的稳态集成。
- 硬编码 32 heads、128 dim、`cuda:0`，多卡/GQA/其他 Llama 配置不通用。
- `enable_twilight` 结尾没有 `return model`；它只原地修改。

#### `twilight/model/wrapper.py`

##### 功能、函数与调用关系

该文件提供独立算子计时用的轻量 wrapper，与 `benchmark/efficiency/bench_breakdown.py` 中的更新版 wrapper 高度相似。

- `get_indices`：等长、page size 1 的 page table。
- `SpGEMVINT4Wrapper`：FTKA packed INT4 sparse GEMV。
- `QuestSelectorWrapper`：FTKA Quest GEMV。
- `TopKWrapper`：RAFT TopK。
- `TopPWrapper`：Twilight Top-p mask。
- `AttentionWrapper`：等长 FlashInfer paged decode attention。
- `VarlenAttentionWrapper`：不同长度的 ragged FlashInfer attention。

本文件调用 `ftka.cuda_ops`、`pyimpl/top_p.py` 和 FlashInfer。目标仓库 Python 搜索中没有其他文件导入这些类；`bench_breakdown.py` 自己复制/演进了一套实现。

##### 逐行/代码块解读

- 第 1-10 行：导入 Torch、类型、FTKA GEMV/TopK、Twilight Top-p 和 FlashInfer。

`get_indices`（第 14-23 行）：

- 为每个 batch 重复 `[0,seq_len)` page id；`indptr` 步长为 seq_len，last page len全 1；设备固定 CUDA。

`SpGEMVINT4Wrapper`（第 26-81 行）：

- 第 33-42 行：从 packed cache推导 B/page/head/head dim，D乘 2。
- 第 44-64 行：构造 Q、输出、随机 scale/zero。
- 第 66-68 行：构造 page metadata。
- 第 71-81 行：调用 INT4 K sparse GEMV写预分配输出。

`QuestSelectorWrapper`（第 84-122 行）：

- 第 91-100 行：读取 label cache并要求 page size 1。
- 第 102-110 行：构造 Q、输出和 metadata。
- 第 113-122 行：同一 K label同时作为 Quest kernel的 K/V输入，仅做 selector算子包装。

`TopKWrapper`（第 125-160 行）：

- 第 132-149 行：创建 input values/indices、output values/indices和固定公式临时 buffer。
- 第 152-160 行：调用 RAFT TopK原地写输出。

`TopPWrapper`（第 163-196 行）：

- 第 171-175 行：保存 FP16/CUDA/阈值并生成 `weight`。
- 第 178-193 行：固定 seed，构造 attention-sink风格 tensor，然后第 192 行已经对它做 softmax。
- 第 195-196 行：又调用 `top_p_unnormalized`，后者会再做一次 softmax。

`AttentionWrapper`（第 199-244 行）：

- 第 207-219 行：从 `[pages,2,page_size,Hkv,D]` cache读取元数据，可选 query head数，构造 Q并保存 cache。
- 第 221-228 行：分配 workspace、page table和 FlashInfer wrapper。
- 第 230-241 行：计时前 plan；参数第 234-235 行都传 `self.num_heads`，没有使用可选 `self.num_qo_heads`。
- 第 243-244 行：运行 decode attention。

`VarlenAttentionWrapper`（第 247-299 行）：

- 第 254-267 行：每个 `seq_lens` 元素是一个请求；读取 query/KV head、最大长度并创建 Q/cache。
- 第 269-277 行：拼接每请求 `[0,len)` page id，逐请求累积 indptr，last page len全 1。
- 第 279-296 行：创建 wrapper并用正确的 `num_qo_heads/kv_num_heads` plan。
- 第 298-299 行：运行 ragged decode attention。

##### 注意与当前实现边界

- `TopPWrapper` 对已经归一化的概率再次 softmax，和 `bench_breakdown.py`/`bench_top_p.py` 的 logits 输入不一致。
- `AttentionWrapper` 可接 `num_qo_heads`，但 plan仍把 query head数写成 KV head数；GQA override没有真正生效。`VarlenAttentionWrapper` 已正确区分二者。
- 多个 `self.o` 只为形状/旧接口预留，FlashInfer `run` 的返回值没有写入或保存到这些字段。
- 全部设备硬编码 CUDA，定位为 GPU benchmark helper。

---

## 关键配置到文件的映射

| 配置字段/类型 | 安装或分发位置 | 实际实现文件 | 关键参数 |
|---|---|---|---|
| `compressor.type=none` | `pyimpl/attention.py` | 无压缩 | — |
| `compressor.type=snap_kv` | prefill cache 写入前 | `pyimpl/snap_kv.py` | `window_size`、`max_capacity_prompt`、`kernel_size`、`pooling` |
| `selector.type=streaming` | decode selector | `pyimpl/streaming_llm.py` | `token_budget`、`num_sinks` |
| `selector.type=quest` | decode selector | `pyimpl/quest.py` | `chunk_size`、`token_budget`/`sparsity_rate` |
| `selector.type=oracle_topk` | decode selector | `pyimpl/top_k.py` | `token_budget` |
| `selector.type=tidal_decode` | decode selector | `pyimpl/tidal_decode.py` | `token_budget`、`reselection_layers` |
| `selector.type=sparq` | decode selector | `pyimpl/sparq.py` | `r`、`token_budget` |
| `selector.type=ds` | 安装时载入 channel，decode 估分 | `pyimpl/double_sparse.py` + `kernel/triton/channel.py` | `config_path`、`selected_channel`、`r`、`quant_bit`、预算 |
| `weight_estimator.type=min_max_quant` | selector 之后 | `pyimpl/quantize.py` | `quant_bit` |
| `weight_estimator.type=max_quant` | selector 之后 | `pyimpl/quantize.py` | `quant_bit`、`smooth_k` |
| `weight_pruner.type=threshold` | estimator 之后 | `pyimpl/elementwise_threshold.py` | `threshold` |
| `weight_pruner.type=top_p` | estimator 之后 | `pyimpl/top_p.py` -> `kernel/cuda/sampling.py` | `threshold` |

## 最重要的文件阅读顺序

若要快速理解项目，建议按以下顺序：

1. `README.md` 的 Select-then-Prune 设计说明；
2. `twilight/pyimpl/state.py`：先认识四组策略类型和共享状态；
3. `twilight/pyimpl/attention.py`：理解 patch、prefill/decode 与四阶段调度；
4. 按实验配置阅读对应的 selector/estimator/pruner 文件；
5. `twilight/pyimpl/top_p.py` -> `twilight/kernel/cuda/sampling.py`：理解高层 pruning 到 CUDA 的边界；
6. `benchmark/LongBench/pred.py` -> `eval.py` -> `metrics.py`：理解准确率评测闭环；
7. `benchmark/efficiency/bench_breakdown.py`：理解高性能算子如何被拆开测量；
8. `twilight/model/*`：作为尚在演进的高性能模型集成原型阅读，不要与当前 LongBench 生效的 `pyimpl` patch 混淆。

## 当前实现中应优先留意的接口断点

以下是跨文件关系中最容易误判的点，集中列出便于排查：

1. `tidal_decode_selector` 返回 tuple，而 `attention_forward` 期望单个 bool Tensor。
2. state 枚举声明 PQCACHE、THRESHOLD/H_THRESHOLD estimator，但 `attention_forward` 没有相应分支。
3. `hierarchical_elementwise_threshold` 调用未定义的 `twi`。
4. `reset_sparse_config` 不卸载 monkey patch，只复位“是否接受新参数”的标志。
5. `model/llama_patch.py` 的 Quest/Top-p 选择结果未接到最终 page indices，且 prefill/cache 是合成原型。
6. `LongBench/pred.py` 的 `samsum` 分支没有创建随后使用的 `generated_content`。
7. `bench_gemv.py` 的 INT8 correctness 分支比较未定义的 `out_int8`。
8. `qk_int8_per_block.py` 没有更新 accumulator，当前 output 路径为全零实验骨架。
