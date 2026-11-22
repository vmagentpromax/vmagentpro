# S1–S4 碎片整理 benchmark 接入主引擎（M9 桥接）

时间：2026-11-22
转换器：`tools/convert_benchmark_instances.py`
跨引擎核对：`tools/compare_benchmark_engines.py`
对象：`benchmark任务/三指标数据评测工作文件_20261108/01_场景数据与构建/candidates/`（7 组候选 × S1–S4 = 28 个实例）

## 1. 这一节解决什么问题

主线诉求是"不要再弄两套东西了"。要合并，前提是**两边能读同一份数据、并核对算出的是不是同一个数**。
她那套 benchmark 吃 `servers.parquet + initial_pods.parquet + arrivals.parquet + config.json`，主引擎吃
`nodes.csv + jobs.csv`，此前没有任何转换关系。M9 就是这座桥，M10 的验收要用它。

## 2. 转换结果（28 个实例全部通过，0 处矛盾）

| 项 | 值 |
|---|---|
| 实例数 | 28（7 组候选 × S1–S4） |
| 节点 | 每组 100 台 H20，8 GPU / 192 核 = 800 GPU / 19,200 核 |
| 任务 | 初始 284–294 个 + 到达 120–160 个 = 每实例 404–454 个 |
| 转换产物 | `outputs/benchmark_csv/<候选>/<Sx>/inputs/{nodes.csv, jobs.csv, initial_placement.csv}` + `conversion_manifest.json` |
| 校验 | 无 pod_id 重复、无跨表重叠、初始布局不超容量、`simulation_start_sec` 全为 0 |

产物在 `outputs/`（已 gitignore），因为**完全可由 `parquet + 转换器` 再生**；转换器的 `conversion_manifest.json`
里记录了源 parquet 的 sha256，源文件本身在仓库内。

字段映射的关键一条：**初始 Pod 用 `remaining_duration_sec`，不用 `duration_sec`**。主引擎是"开始放置时才起算服务时间"，
这些 Pod 在 t=0 被放到指定节点上，它们还欠的是剩余时长；用 `duration_sec` 会把每个初始 Pod 的寿命人为拉长它已经跑过的那一段。

## 3. ★ 主引擎需要一个"指定初始布局"的能力

她的 benchmark 的**全部意义**在于起始状态是人为构造的碎片。如果让主引擎的 first_fit 去放这 294 个初始 Pod，
它们会被压实，实验就变成另一件事了。所以本次给主引擎新增：

- `contract.load_initial_placement()` 读 `{pod_id, server_id}`
- `GpuSchedEnv(..., initial_placement={...})`：这些 Pod **在 t=0 直接落到指定节点，不经过任何策略**；
  与 workload 矛盾（submit_time≠0、server 不存在、超容量、pod 未知）一律抛错而不是悄悄重放
- 命令行 `--initial-placement <文件>`

不传该参数时行为与之前完全一致：64 项测试通过，场景 F 归档结果仍逐字段复现。

## 4. 跨引擎核对结果：first_fit 28/28 完全一致，best_fit 19/28 不一致

`tools/compare_benchmark_engines.py` 用**她引擎的指标定义**（等待右删失到 horizon、p95 用 numpy linear 分位、
deadline 率按全部到达算、预留率按 `[0, allocation_window]` 积分）从主引擎的 records 重算，再与实例自带的
`baseline_results.csv` 比对。只比 `migration=none` 的格子——主引擎还没有迁移能力，其余 4 个迁移策略列不可比。

| 放置策略 | 一致 | 不一致 |
|---|---:|---:|
| `first_fit` | **28 / 28** | 0 |
| `best_fit` | 9 / 28 | **19 / 28** |

不一致时的量级：

| 指标 | 最大偏差 | 平均偏差 |
|---|---:|---:|
| `gpu_reserved_allocation_rate` | 0.0109（1.09 个百分点） | 0.0047 |
| `p95_wait_censored_sec` | 303.9 秒 | 148.8 秒 |
| `mean_wait_censored_sec` | 71.2 秒 | 14.8 秒 |
| `start_within_deadline_rate` | 0.0188 | 0.0125 |

**原因已经定位，不是数值噪声，是两套 best_fit 不是同一个算法：**

```python
# 她的 engine.py:51  —— 字典序：先看 GPU 剩余，再看 CPU 剩余，再看 server_id
min(fit, key=lambda s: (free_gpu - gpu_request, free_cpu - cpu_request, s))

# 主引擎 placement.py —— 归一化求和：GPU 剩余比例 + CPU 剩余比例，再看节点下标
min(valid, key=lambda j: ((free_gpu-gpu)/gpu_capacity + (free_cpu-cpu)/cpu_capacity, j))
```

举例：A 节点剩 1 GPU / 100 核，B 节点剩 0.5 GPU / 200 核。字典序选 B（GPU 剩余更少），
归一化求和选 A（0.646 < 1.104）。两者都是"剩余最少"的合理实现，但**不是同一个策略**。

`first_fit` 之所以 28/28 完全一致：两边都是"按 server_id 升序取第一个能放的"，转换后的 `nodes.csv`
正好是 server_id 升序，所以候选顺序也一致。这说明**账本、事件顺序、容量判定、队列规则这四层是对得上的**，
差异只在放置打分这一层。

## 5. 这件事要求谁做决定

M10 合并前必须先定一件事：**哪个 best_fit 是权威定义**。三个选项：

1. 以主引擎为准（归一化求和）——她那 160 组 best_fit 结果需要重跑；
2. 以 benchmark 为准（GPU 优先字典序）——主引擎的 best_fit 语义变化，仓库内所有 best_fit 结果需要重跑；
3. 两个都留，改名为 `best_fit_sum` / `best_fit_gpu_first`——不动历史数字，但策略矩阵要扩列。

我倾向 **3**：这两条本来回答的是不同的问题（"整体剩余最少" vs "GPU 剩余最少"），而 GPU 碎片是这个项目的
研究对象，把 GPU 优先那条单独立名反而更有解释力；且不需要重跑任何已有结果。

## 6. M10/M11：她的响应式碎片整理已经进主引擎，实测对齐

`vmagent/gpu/migration.py` 把她 `engine.py` 的核心机制移植进主引擎，跑在主引擎的**增量账本、事件堆、队列索引与指标层**上，
不再有自己的 `st.used` 全量重算。触发条件用 `queues.py` 的分类器（只对 `fragmentation` 反应，不对容量短缺反应）。
四个开关即 M11，默认值 = 她的现行为：

| 开关 | 默认 | 她的 config 字段 |
|---|---|---|
| `migration` | `none` | 策略参数 `one_move/cheap_multi/first_multi` |
| `reserve_destination` | `True` | `destination_reserved_before_source_release` |
| `release_source_when` | `"end"` | （隐含：源在迁移落地时才释放） |
| `pause_service` | `True` | `pause_service` |
| `block_admission_during_plan` | `True` | `admission_paused_during_plan` |

### 对齐结果（112 格 = 28 实例 × 4 个迁移模式，只比 first_fit）

| 指标 | 结果 |
|---|---|
| `migration_count` / `migration_completed_count` / `gpu_units_moved` / `migration_gpu_pause_sec` | **全部逐格完全一致** |
| `started` / `completed` / `unstarted` / `p95_wait_censored_sec` / `mean_wait_censored_sec` / `start_within_deadline_rate` / `gpu_reserved_allocation_rate` | **全部逐格完全一致** |

**112/112 格、0 个字段差异**（28 实例 × 4 个迁移模式，first_fit；相对容差 1e-9）。

### 曾经存在的 31 格残差是怎么定位并修掉的

第一轮核对是 81/112 一致，31 格 `gpu_reserved_allocation_rate` 差 ≤2.2e-4，且只在迁移格出现，`none` 模式零差异。定位工具 `tools/diff_migration_integral.py` 的做法是：从**两边各自的事件日志重建"已分配 GPU 随时间的阶梯函数"**，再逐区间比对。用它重建她的日志能精确复算出她 `summary` 里的数字（0.856133310105783 对 0.856133310105783），重建主引擎的事件也能复算出主引擎的数字，所以差异被压到具体的时间区间：

```text
S1 / cheap_multi：3 个差异区间，每个恰好差 1.0 GPU，合计 28.8000 GPU·秒
  [1539.448387, 1545.974937)   6.53 s   benchmark=740  engine=739
  [1545.974937, 1567.428328)  21.45 s   benchmark=739  engine=738
  [1567.428328, 1568.248387)   0.82 s   benchmark=739  engine=738
```

再往下查发现：两边**移动次数相同、被移动的 Pod 及各自次数相同、到达与启动时刻全部相同**，只有 3 个 Pod 的**完成时刻**分别晚了 28.8 / 28.8 / 31.2 秒——正好等于这三个 1-GPU Pod 各自的迁移暂停时长（`20 + 8×1 + 0.1×核数`）。

根因：主引擎在迁移落地时是**用 `running_start + service − preserved` 重算完成时刻**，而不是在 Pod 当前的完成时刻上加暂停。第一次移动两者相同；**同一个 Pod 被移动第二次时，重算会把第一次的暂停丢掉**。她的引擎写的是 `p['finish'] += pause`，累加在当前完成时刻上。

修法：引擎现在维护 `finish_at[pid]`（`_place` 与初始布局写入，迁移在其上累加暂停），迁移不再重算。修完 112/112 全一致。

顺带修掉一个事件日志污染：`migration_complete` 被记录**两次**（`_complete_migration` 内部记一次，`_process_due` 弹出该事件后又记一次），而且已经失效的迁移事件也会被记成"某次迁移完成了"其实什么都没完成。现在 `_complete_migration` 返回是否真的落地，调用方据此跳过记录，与 `complete` 事件用 `completion_counter` 判旧的处理方式一致。这个不影响占用积分（`_record` 写的是当时的真实状态），但影响事件日志可信度。

回归测试：`test_two_moves_of_one_pod_pay_two_pauses`（一个 Pod 两次迁移必须付两次暂停）、`test_a_stale_move_event_is_recorded_only_once`。

### 四个开关确实改变结论（S1/first_fit/cheap_multi 实测）

| `release_source_when` | `reserve_destination` | GPU 预留率 |
|---|---|---:|
| end | True（她的默认，make-before-break 双占） | 0.856123 |
| end | False | 0.855598 |
| start | True | 0.855598 |
| start | False（暂停期两头都不占） | 0.855073 |

中间两行相等不是巧合：一头占与另一头占在总量上等价，这本身就是对实现的一次交叉校验。
四个组合给出四个不同结果，说明 M11 要暴露的敏感性是真实的，而不是文档上的假设。

## 6.5 主引擎现在能直接产出她那张结果表（2026-11-22 补充）

`tools/benchmark_runner.py` 用主引擎跑 28 实例 × 2 放置 × 4 迁移模式 = 224 格，
输出**与她 `baseline_results.csv` 完全相同的表头和列序**，落在
`outputs/benchmark_main_engine/<候选>/<Sx>/main_engine_results.csv`，旁边一份
`main_engine_results.notes.json` 说明哪些列是故意留空的。

这样做的目的只有一个：**让她那份 `engine.py` 变成可以指向主引擎的引用，而不是唯一能重出现有数字的地方。**
删不删她的代码是第二步（要她签字），但先把"删了也能重算"这件事做实。

为此主引擎补了两个新指标（`4701249` 与本次提交）：

| 新指标 | 含义 |
|---|---|
| `effective_gpu_ratio` | 她的 A_GPU：正在迁移的 Pod 在暂停期间**不计入有效分配**（它占着资源但没在算） |
| `effective_gpu_ratio_full_horizon` | 同一积分按**整个 horizon** 而不是分配窗做分母 |
| `migration.gpu_pause_accrued_sec` | 按时间积分的实际暂停量（与按计划登记的 `gpu_pause_sec` 只在"结束时仍有迁移在途"时不同） |

`effective_gpu_ratio` 第一次实现时是错的，两个错误都是**靠比对发现的、不是靠看代码发现的**：
只扣了 1 份 GPU（make-before-break 下账本其实计了 2 份），以及全 horizon 版用了"最后仿真时刻"而不是 horizon 做分母。
修完后 `tools/compare_benchmark_engines.py`：**112 格、0 字段差异**（first_fit × 4 迁移模式 × 28 实例）。

`tools/benchmark_runner.py` 全量跑完后与 28 份归档表逐格对比：

| 放置 | 差异 |
|---|---|
| `first_fit` × 4 迁移模式 × 28 实例 = 112 格 | **0 个字段差异** |
| `best_fit` × 同上 = 112 格 | **83 格有差异，共 643 个字段**；另外 29 格相同（两种规则在那格上恰好选了同一个节点）。差异范围从 `mean_wait` 几秒到 `migration_count` 31 对 35，原因见第 4 节：两套 best_fit 不是同一个算法 |

也就是说：**在 best_fit 的定义定下来之前，主引擎可以完全替代她的引擎跑 first_fit 那一半；best_fit 那一半替代不了。** 这正是第 5 节那个决定为什么必须先做。

## 7. 复跑命令

```bash
# 1) 转换全部 28 个实例（需要 pandas + pyarrow）
C:/Users/trw/anaconda3/python.exe tools/convert_benchmark_instances.py --all

# 2) 跨引擎核对（只比 migration=none 的格子）
C:/Users/trw/anaconda3/python.exe tools/compare_benchmark_engines.py

# 3) 单个实例在主引擎上跑
python -m vmagent.gpu.run --clock fast --policy first_fit --queue-policy backfill ^
  --max-sim-time 10800 --measurement-seconds 3600 ^
  --nodes outputs/benchmark_csv/seed20261107_interval10/S2/inputs/nodes.csv ^
  --tasks outputs/benchmark_csv/seed20261107_interval10/S2/inputs/jobs.csv ^
  --initial-placement outputs/benchmark_csv/seed20261107_interval10/S2/inputs/initial_placement.csv ^
  --output outputs/gpu/S2_first_fit
```

## 8. 仍然不可比的部分（2026-11-22 按当前代码逐条重核）

这一节原先写于 M10 之前，**其中三条已经过期**，过期内容会把人往回带，所以整节按代码现状重写。
判据是"仓库里现在有没有这段代码 + 实测跑没跑过"，不是记忆。

| 项 | 原状态 | 现在的真实状态 |
|---|---|---|
| 迁移四个指标（`migration_count` / `gpu_units_moved` / `migration_gpu_pause_sec` / 扣除迁移暂停的 A_GPU） | "主引擎尚无迁移能力" | **已可比**。M10/M11 之后主引擎有迁移与四个预留开关，112 格跨引擎核对**逐格完全一致**（见第 6 节），`tools/benchmark_runner.py` 直接输出这些列 |
| `start_deadline_sec=300` 的达成率 | "主引擎 summary() 里没有，核对脚本用 records 现算" | **仍在脚本里算，不在 `summary()` 里**。`benchmark_runner.py:79` 按 config 的 deadline 现算 `start_within_deadline_rate`，28 个实例可复现；但它不是引擎原生指标，换别人调引擎拿不到 |
| `decision_interval_sec=30`（她的引擎每 30 秒决策一次，主引擎事件驱动） | 会影响迁移触发时机 | **仍然是语义差异，但在这些实例上没有产生任何可观测差异**：112 格（含开启迁移的格子）迁移次数、搬运 GPU 数、暂停 GPU·秒全部逐格相同。所以这是一个"潜在差异、实测未触发"，不是"已知偏差" |
| `migratable` 列被丢弃 | 丢弃不改变现状 | **不再丢弃**。转换器写出该列（`convert_benchmark_instances.py:102`），引擎从 `task.metadata` 读取并解析文本 `"False"`。**注意**：这些实例里所有 Pod 仍是 `migratable=True`（她的 config 里 `all_pods_assumed_migratable`），所以这条修复目前不改变任何数字，它改变的是"引入部分可迁移时必须回头改转换器"这件事 |
| `asw_id` 全为空，拓扑/熵指标覆盖率 0 | 不能评价拓扑感知策略 | **仍然成立**，未变 |
| 她的 `engine.py` 本体 | 归档在 `benchmark任务/…` | **仍未处置**，删除需要她签字。但"删了会丢能力"这个理由已经不成立：`tools/benchmark_runner.py` 能在 her 的表头、her 的列序上产出 first_fit 112 格 0 差异、best_fit 83 格有差异（差异来源是两套 best_fit 定义不同，见第 4 节，不是能力缺失） |

一句话结论：**能力差异已经清零，剩下的全是"定义还没统一"和"数字对不对得上"两类。**
前者是 best_fit 的权威定义（需她定），后者是 `runs/old_gpu_only_20261025/` 与十场景表这类
已汇报数字是否要用主引擎重出（需你点头）。
