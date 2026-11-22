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

## 6. 复跑命令

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

## 7. 已知不可比的部分（写清楚，别当成已对齐）

- **迁移相关四个指标**（`migration_count` / `gpu_units_moved` / `migration_gpu_pause_sec` /
  `gpu_effective_allocation_rate` 的"扣除迁移暂停"口径）：主引擎尚无迁移能力，M10/M11/M12 之后才可比。
- **`start_deadline_sec=300` 的达成率**：主引擎 `summary()` 里没有这个指标，上面的核对脚本是用 records 现算的。
- **`decision_interval_sec=30`**：她的引擎每 30 秒做一次决策，主引擎是事件驱动。这不影响 `none` 列的结果
  （实测完全一致），但会影响迁移触发时机。
- **`migratable` 列被丢弃**：她的实例里所有 Pod 都标 `migratable=True`（config 里
  `all_pods_assumed_migratable`），所以丢弃不改变现状；一旦引入"部分可迁移"就必须补回来。
- **`asw_id` 全为空**：拓扑/熵指标在这些实例上覆盖率为 0，不能用来评价拓扑感知策略。
