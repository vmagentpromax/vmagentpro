# 场景 F：异构 8 天高负载窗口（day109–116）

收录时间：2026-11-12
数据来源：本机 `D:\vmagent_scenarios_hetero_8d\`（未入库，见第 6 节）
上游 trace：`alibaba/clusterdata` 的 `cluster-trace-gpu-v2026`（ASI，OSDI'26）

本目录是**场景 F 的仿真就绪输入与指标结论**。运行明细（events / resource_history / tasks，约 1.2 GB）未入库，可由本目录的 `inputs/` 加第 5 节的命令复现。

---

## 1. 场景定义

| 项 | 值 |
|---|---|
| 角色 | 8 天整体高负载窗口，用于调度策略横向对比 |
| 时间块 | `day=109 hour=00` 至 `day=117 hour=00`，连续 192 小时 |
| 预热 | `day=108 hour=23`（1,070 个 warmup 任务，warmup GPU/CPU 负载 0.6793 / 0.5596） |
| 源分区 | `asi_opensource_pod_hourly` day=108 hour=23 + day=109..116 全部小时 |
| cluster_id | `1f28e063c4e5e1bc48ff55269617dd2f`（30 台同属一个 cluster，已核实唯一） |
| asw_id | `7b39f1498b7a4c42b46c440960e863b9`（**单一 ASW 域**，池内无跨 ASW 拓扑差异） |
| 筛选 | 30 台池内服务器；`gpu_request>0 OR cpu_request_cores>0` |
| 规模 | 339,034 条 pod-hour 原始行 → **119,742 个去重 pod** → 119,495 个入仿真任务 |
| 任务构成 | **119,495 个**：GPU 任务 5,910、CPU-only 任务 113,585 |
| 达成负载 | GPU **96.11%** / CPU **143.89%**（CPU 为超载场景） |
| 构造种子 | `20261101` |
| 最长服务时长 | 2,322,400 秒 |
| 推荐仿真上限 | 4,000,000 秒（约 46.30 个仿真日，非墙钟） |

### 节点池

**30 台，总容量 240 GPU / 5,760 CPU 核**，实测构成：

| 机型 | 台数 | 每台 GPU | 每台 CPU 核 |
|---|---:|---:|---:|
| XPU-A | 20 | 8 | 192 |
| H20 | 10 | 8 | 192 |

GPU 任务受机型约束，只能放到 `allowed_gpu_types` / `gpu_spec_public` 匹配的节点。任务机型分布实测：CPU-only（`[]`）113,585、XPU-A 4,720、H20 1,190。

### 优先级（SpotGPU 依赖此字段）

| 口径 | HP | LP | Other |
|---|---:|---:|---:|
| **全部 119,495 个任务** | 9,487 | 5,317 | 104,691 |
| **仅 5,910 个 GPU 任务** | **447** | **5,215** | **248** |

第二行与 `vmagentpro准备工作/异构场景支持修改说明.md:48` 记录的数字逐个吻合（该文档用的是 GPU 任务口径）。**这是仓库内唯一带真实 `priority_class` 的场景**——默认场景 A 的 50 个任务该字段全为空，触发不了抢占路径。

### 作业类型分布（实测）

`unknown` 113,613、`offline_inference` 5,341、`online_inference` 461、`training` 73、`other` 4、`dev` 3。

`unknown` 占 95% 与 CPU-only 任务数（113,585）基本重合，即**作业类型信息实际只对 GPU 任务有效**。按类型分桶报指标时应注明这一点。

### 构造假设（来自 `inputs/scenario_manifest.json`，逐条照录）

1. 异构池：ASW `7b39f149...`（cluster `1f28e063...`）的 20 台 XPU-A + 10 台 H20；机型约束经 `allowed_gpu_types` 表达
2. 单个 192 小时连续窗口 `day109 h0 .. day117 h0`；预热取自 `day108 h23`
3. **到达时刻**：任务在窗口内首次出现的那个小时内**均匀分布**（合成，非 trace 记录）
4. **服务时长**：GPU 任务取 `job_execution_summary` 的 `duration_hours` 中位数；未匹配到的与 CPU-only 任务取观测 presence span（`last-first+1` 小时）
5. 窗口末尾附近首次出现的 CPU-only 任务，其 presence span 存在**右删失**（`right_censored_first_seen_last_hour` 487、`right_censored_first_seen_last_day` 15,125）
6. 无合成 CPU 背景；CPU 负载全部来自真实 pod 请求
7. **`gpu_mem_request` 与主机 RAM 不建模**
8. `asw_id` 在 day ≥ 109 全部存在，无需代理

第 3、4、5 条意味着**到达顺序与服务时长含有合成成分**，不能说"精确还原了历史执行实例"。引用本场景结果时须保留这个声明。

---

## 2. 目录内容

```text
runs/hetero8d_20261101/
├── README.md                          本文
├── inputs/
│   ├── nodes.csv              2,417 B   30 台，引擎格式（server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores）
│   ├── jobs_F.csv        15,910,384 B   119,495 行，引擎格式（10 字段，见下）
│   ├── scenario_manifest.json 4,799 B   场景构造口径与假设（原 D 盘 manifest.json，改名以区别于 conversion_manifest）
│   ├── cpu_background.csv     1,682 B   CPU 背景负载表（本批十组运行 background=0，未使用）
│   └── nodes_raw_with_asw.csv 3,656 B   节点原始字段层，含 source_asw_id / snapshot_day / snapshot_hour（选型证据）
└── metrics/
    ├── 策略对比结果.md        13,640 B   十组策略对比的结论表与口径（2026-11-02 既有交付文档，保留原名）
    ├── provenance.json         3,393 B   本次收录生成：输入 sha256、实测统计、运行参数、收录与未收录清单、上游缺口
    └── summaries/                       14 个运行组，每组 summary.json + run_manifest.json（共 28 个文件）
```

`jobs_F.csv` 的 10 个字段：

```text
pod_id, gpu_spec_public, gpu_request, cpu_request_cores, submit_time_sec,
service_sec, cluster_id, duration_match_status, job_type_public, priority_class
```

**该表头与 `runs/allpods_20261025/inputs/jobs_A.csv` 逐字相同**，`nodes.csv` 表头也与 `runs/allpods_20261025/inputs/nodes.csv` 逐字相同——两套场景输入是同一格式，可共用加载与评价代码。

### 关于 sha256 核对（跨平台必读）

本机 `core.autocrlf=true`，所以**工作副本是 CRLF，而 git blob 里存的是归一化后的 LF**。`metrics/provenance.json` 的 `input_sha256` 记录的是**工作副本（CRLF）**口径——这与仓库既有的 `runs/allpods_20261025/metrics/provenance.json` 口径一致（我核对过，那份记录的三个哈希全部匹配工作副本而非 blob）。

后果：在 Linux/macOS 上 clone，或在 Windows 上把 `core.autocrlf` 设为 `input`/`false` 时检出，得到的是 LF 版本，`sha256_working_copy_crlf` **不会匹配**。这不是数据损坏。`provenance.json` 的 `hash_convention.files` 段为每个输入同时记录了两种哈希：

| 文件 | CRLF 字节 | LF 字节 | 除行尾外内容相同 |
|---|---:|---:|:---:|
| `jobs_F.csv` | 15,910,384 | 15,790,888 | ✅ |
| `scenario_manifest.json` | 4,799 | 4,654 | ✅ |
| `nodes_raw_with_asw.csv` | 3,656 | 3,625 | ✅ |
| `nodes.csv` | 2,417 | 2,386 | ✅ |
| `cpu_background.csv` | 1,682 | 1,651 | ✅ |

五个文件均已逐个验证 `identical_apart_from_line_endings = true`。CSV/JSON 解析对行尾不敏感，两种版本都能被引擎正确加载（第 5 节的复现校验就是在 CRLF 工作副本上跑的）。**跨平台核对时请用 `sha256_git_blob_lf`。**

### 收录的 14 个运行组

`F-{first_fit,best_fit,worst_fit,random,spotgpu}-{fcfs,backfill}` 共 10 组，另有 4 个 `F-{first_fit,best_fit,worst_fit,random}-fcfs-rerun`（性能优化后的复跑，用于核验既有结果）。

**未收录**：每组的 `events.csv`（约 30 MB）、`resource_history.csv`（约 30 MB）、`tasks.csv`（约 21 MB）、`nodes_final.csv`（2 KB），单个运行目录约 80 MB、14 组合计约 1.2 GB。`nodes_final.csv` 体积小但同样未收，因为它与 events/tasks 属同一层运行明细；如需核对资源释放，复跑一次即可。

---

## 3. ★ 三个节点池不要混淆

仓库里现在存在三个都叫"20 台/30 台 H20"的节点池，**容量各不相同**，实测如下：

| 节点表 | cluster_id | 构成 | 每台核数 | 总 GPU | 总 CPU 核 |
|---|---|---|---:|---:|---:|
| `docs/server_selection/servers_20_H20.csv` | `1f28e063...` | 20 × H20 | **192** | 160 | **3,840** |
| `runs/allpods_20261025/inputs/nodes.csv` | `9089d80b...` | 20 × H20 | **224** | 160 | **4,480** |
| `runs/hetero8d_20261101/inputs/nodes.csv`（场景 F） | `1f28e063...` | 20 × XPU-A + 10 × H20 | 192 | **240** | **5,760** |

前两行**都是"20 台 H20、160 GPU"，但 cluster 不同、每台核数不同（192 vs 224）、总 CPU 容量差 640 核**。这不是笔误，是两个不同的节点池：前者来自 `docs/server_selection/` 的首轮选型（论文 §3.3 举例的 8-GPU H20 / 192 核），后者来自五场景（allpods）工作，`conversion_manifest.json` 里记的是 `capacity_cpu: 4480`。

**已经产生的后果**：`docs/server_selection/run_request_scenes.py:123,125` 把容量写死成 `160` 和 `3840`，只适用于第一个池；`runs/allpods_20261025/metrics/README.md` 写的是"160 GPU、4,480 CPU 核"。**用错池会让 CPU 预留率算错约 17%。**

引用任何 CPU 相关指标前，先确认用的是哪个 `nodes.csv`。

---

## 4. 与仓库内其他工作的关系

| 关系 | 说明 |
|---|---|
| **是十场景抢占实验的数据源** | `experiments/preemption_20261107/build_scenarios.py:13` 的 `SOURCE` 原本指向 `D:/vmagent_scenarios_hetero_8d/sim_inputs`；十场景中的 `P0_real_F8d` 就是场景 F 本身 |
| **是性能优化说明的数据来源** | `vmagentpro准备工作/调度器性能优化说明.md` 里"119,495 任务 / 30 节点"的十组墙钟数字，与 `metrics/策略对比结果.md` 逐个一致 |
| **驱动了引擎的异构与抢占支持** | 提交 `e3181b4`（异构 GPU 支持、backfill 扫描游标、分机型指标、64 MiB 上传上限、8 天场景预设 UI）与 `d20f8b1`（SpotGPU 抢占、CPU 线段树 backfill）都是为跑通本场景而做 |
| **前端有对应预设按钮** | `vmagent/gpu/static/index.html:44` 的"异构 8 天场景 F 参数"按钮、`app.js:83` 的提示文案（测量窗 691200 秒 / 上限 4,000,000 秒 / 快速模式） |

---

## 5. 复跑命令

从仓库根目录执行。

```bash
# 单组：First Fit + FCFS（约 42 秒）
python -m vmagent.gpu.run --clock fast --seed 42 --policy first_fit --queue-policy fcfs ^
  --measurement-seconds 691200 --max-sim-time 4000000 --cpu-background-fraction 0 ^
  --nodes runs/hetero8d_20261101/inputs/nodes.csv ^
  --tasks runs/hetero8d_20261101/inputs/jobs_F.csv ^
  --output outputs/hetero8d/F-first_fit-fcfs

# SpotGPU 抢占（checkpoint 间隔 3600 秒）
python -m vmagent.gpu.run --clock fast --seed 42 --policy spotgpu --queue-policy fcfs ^
  --checkpoint-interval 3600 --measurement-seconds 691200 --max-sim-time 4000000 ^
  --nodes runs/hetero8d_20261101/inputs/nodes.csv ^
  --tasks runs/hetero8d_20261101/inputs/jobs_F.csv ^
  --output outputs/hetero8d/F-spotgpu-fcfs
```

（`^` 是 Windows cmd 的续行符；PowerShell 用反引号，bash 用反斜杠。）

**网页方式**：`打开前端.bat` → 点"异构 8 天场景 F 参数" → 上传本目录 `inputs/nodes.csv` 与 `inputs/jobs_F.csv`（任务表约 15.9 MB，上传上限 64 MiB）→ 选策略与队列规则 → 开始实验。

完整参数见 `python -m vmagent.gpu.run --help`。

### 复跑时的预期

`metrics/策略对比结果.md` 记录的十组结果（`fast` 时钟、seed 42、background 0、测量窗 691,200 秒、上限 4,000,000 秒）：

| 放置 | 队列 | 墙钟秒 | 完成 | 未完成 | 结束原因 |
|---|---|---:|---:|---:|---|
| first_fit | fcfs | 42.1 | 119,495 | 0 | drained |
| best_fit | fcfs | 30.4 | 119,495 | 0 | drained |
| worst_fit | fcfs | 25.4 | 119,489 | 6 | max_sim_time |
| random | fcfs | 24.8 | 119,490 | 5 | max_sim_time |
| spotgpu | fcfs | 33.0 | 119,495 | 0 | drained |
| first_fit | backfill | 276.7 | 119,495 | 0 | drained |
| best_fit | backfill | 122.8 | 119,495 | 0 | drained |
| worst_fit | backfill | 120.9 | 119,495 | 0 | drained |
| random | backfill | 128.9 | 119,495 | 0 | drained |

墙钟时间受本机负载影响，只描述当次实验成本，**不用于评判策略好坏**。完成数、未完成数、结束原因是可复现的判定项。

### 收录后的复现校验（2026-11-12，本机实测）

从**仓库内路径**跑 `first_fit + fcfs`，与 D 盘原结果 `metrics/summaries/F-first_fit-fcfs/summary.json` 逐字段对比：

| 项 | 结果 |
|---|---|
| 一致字段 | **30 / 30**（不一致 0） |
| 跳过字段 | 2（`wall_time_sec`、`sim_time_sec` 属墙钟/时钟类，按口径不参与判定） |
| `sim_time_sec` | 3,628,230.0 —— 两侧相同 |
| `reserved_gpu_ratio` | 0.7447563657407585 —— 到浮点末位相同 |
| `reserved_cpu_ratio` | 0.7896641038180335 —— 同上 |
| `offered_gpu_ratio` / `offered_cpu_ratio` | 0.961098173605083 / 1.4389450130270514 —— 同上 |
| `mean_wait_sec_started` / `p95_wait_sec_started` | 673,580.8773264866 / 1,050,225.11 —— 同上 |
| `peak_queue_length` | 89,127 —— 相同 |
| `completed` / `rejected` / `unfinished` / `end_reason` | 119,495 / 0 / 0 / `drained` —— 相同 |
| 墙钟 | 本次 37.7 秒，原记录 42.1 秒 |

加载校验：`make_environment()` 读仓库内 `inputs/` 耗时 3.2 秒，解析出 30 节点 / 240 GPU / 5,760 CPU 核 / 119,495 任务，优先级分布 `Other 104,691 / HP 9,487 / LP 5,317`，机型 `XPU-A 20 / H20 10`——全部与第 1 节一致。

**结论：本目录的 16 MB 输入足以逐位复现原结果，不需要那 1.2 GB 运行明细。** 校验产物写在 `outputs/`（已 gitignore）并已清理。

---

## 6. 上游原始层与已知缺口

`inputs/` 收的是**转换后的引擎格式**。转换前的原始层仍在 `D:\vmagent_scenarios_hetero_8d\`，未入库：

| 文件 | 大小 | 表头 | 说明 |
|---|---:|---|---|
| `jobs_scenario_F.csv` | 14,135,696 B | `job_id, submit_time_sec, gpu_request, cpu_request_cores, service_sec, allowed_gpu_types, job_type_public, priority_class, source_pod_id, service_source, pod_kind` | 比引擎格式**多三列溯源信息**：`source_pod_id`（回查 trace 的原始 pod）、`service_source`（服务时长的推导方式）、`pod_kind` |
| `nodes.csv` | 3,656 B | `node_id, source_cluster_id, source_asw_id, gpu_type, gpu_count, cpu_capacity_cores, snapshot_day, snapshot_hour` | 已作为 `inputs/nodes_raw_with_asw.csv` 入库 |

两者的 sha256 记录在 `metrics/provenance.json` 的 `upstream_not_in_repo` 段，可用于日后核对。

**已知缺口（三条，都不要掩盖）：**

1. **转换脚本未保留。** raw 层 → `sim_inputs/` 层的转换脚本在 D 盘不存在（`ls *.py` 为空）。对照 `runs/allpods_20261025/prepare_inputs.py` 的做法，这一步本应有脚本入库。因此 `jobs_F.csv` 目前**不能从 raw 层自动重建**，只能靠 sha256 核对身份。
2. **`allowed_gpu_types` 在转换中丢失。** raw 层用的是引擎推荐的 `allowed_gpu_types`（JSON 数组，见 `异构场景支持修改说明.md:150`），引擎格式却退回单机型字符串 `gpu_spec_public`。CPU-only 行在 raw 层是 `[]`，转换后成为字符串 `"[]"`（实测 113,585 行取值为字面量 `[]`）。功能上等价，但**丢失了"一个任务允许多种机型"的表达能力**。
3. **`outputs/` 明细只在 D 盘单机。** 1.2 GB 运行明细无备份、无版本控制。`metrics/summaries/` 里的 14 组 `summary.json` 是本次收录后仓库内唯一的数值留存。

---

## 7. 使用边界

- 达成负载 GPU 96.11% / CPU **143.89%**——**CPU 是超载的**，这不是一个"双低"温和场景。CPU-only 任务占 95%，CPU 争用主导排队行为。
- 到达时刻与服务时长含合成成分（第 1 节假设 3、4、5），不可宣称还原真实执行实例。
- 单一 ASW 域，**不能用于研究跨 ASW 拓扑约束**。
- `gpu_mem_request` 与主机 RAM 未建模；显存不是约束维度。
- 预留率指标是**资源分配比例**，不是 GPU SM 利用率，也不是 CPU 执行利用率。
- 未实现跨节点 gang placement。`workload_id` 在 trace 中非空率约 63%，无法对所有任务判定组归属。
