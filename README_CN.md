# VMAgent GPU/CPU 调度版（中文说明）

本仓库是 [mail-ecnu/VMAgent](https://github.com/mail-ecnu/VMAgent) 的一个改造分支，仓库地址：

- https://github.com/vmagentpromax/vmagentpro
- 初始版本基于 commit：`cb828fdd153afae208558d006acbc59ddb2e64c5`；后续中文文档与前端改版在 `main` 分支。

它保留原版 VMAgent 的代码结构，同时新增了一套面向 **H20 GPU + CPU** 的定时资源调度环境。与只做 CPU/内存/NUMA 的原始模拟器不同，这套环境按任务真实的 `submit_time_sec` 到达，按 `service_sec` 完成，并在同一台服务器上同时检查 GPU 与 CPU 余量。

> 一句话：这是“给定 20 台 H20 节点和一份带到达/执行时间的任务序列，按 First Fit / Best Fit 等策略做 GPU+CPU 联合放置”的可运行模拟器，不是 GPU 性能模拟器，也不会在 H20 上执行真实模型。

相关文档：

- [README.md](README.md)：英文原版说明
- [README_GPU.md](README_GPU.md)：GPU/CPU 改造的详细中文说明（字段、网页、命令行、验证）
- [docs/server_selection/README.md](docs/server_selection/README.md)：仓库内置 20 台 H20 与场景数据的来源说明

---

## 一、这套环境做了什么

新增的 GPU 环境位于 `vmagent/gpu/`，主要能力：

1. **读取真实节点和任务**：节点 CSV、任务 CSV/JSON；保留小数 GPU 请求与 CPU 核数。
2. **整机资源账本**：每台节点同时维护 GPU 总量/剩余量和 CPU 总量/剩余量；一个任务必须在同一台节点上同时满足两项请求。
3. **时间推进**：处理到达事件、开始放置、完成释放；任务等待期间不预占资源；完成时刻 = 实际开始时刻 + `service_sec`。
4. **等待与拒绝**：暂时放不下的任务进入等待队列；型号不匹配、单节点容量绝对放不下的任务标记 `rejected` 并记录原因。
5. **放置策略**：`first_fit`、`best_fit`、`worst_fit`、`random`；队列策略 `backfill` 或 `fcfs`。
6. **两种时钟**：`fast` 直接推进模拟时间，适合功能检查和批量实验；`realtime` 按墙钟节奏推进，适合观察真实到达过程，可用 `--speed` 加速。
7. **网页入口**：本机 HTTP 服务，支持上传节点/任务文件、运行、查看状态、下载结果。
8. **统计输出**：提供负载、实际资源预留率、等待时间、队列长度、峰值资源、任务明细等。

### 与原版 VMAgent 的边界

- 原版的 NUMA、CPU 配额、内存、虚拟机生命周期和 RL 训练网络仍然是原代码；新 GPU 环境是独立入口。
- 新环境**不建模** GPU 显存、主机 RAM、NUMA、ASW 拓扑和跨节点 gang placement。
- 新环境是“资源请求账本”，不是 GPU SM 利用率或真实执行性能的模拟。
- 原版的 DQN/PPO/SAC 模型输入维度（2N 个 NUMA 动作）与新环境的 N 个服务器动作不兼容，不能直接加载旧权重；需要重新适配和训练。

---

## 二、目录结构

```text
vmagentpro/
├─ README.md                 英文原版说明
├─ README_GPU.md             GPU/CPU 改造的详细中文说明
├─ README_CN.md              本文件
├─ docs/server_selection/    内置 20 台 H20 节点、场景 A 计时数据、来源审计
├─ vmagent/gpu/              GPU 环境、命令行入口、网页入口、静态资源
│  ├─ environment.py         数据读取、资源账本、时钟、队列、完成释放、统计
│  ├─ run.py                 命令行入口
│  ├─ dashboard.py           本机网页服务
│  └─ static/                网页前端
├─ schedgym/gpu_env.py       新环境导入入口
├─ tests/test_gpu_*.py       GPU 环境与网页测试
├─ conda_env.yml             原版 VMAgent 的训练环境（不是 GPU 入口必需）
└─ runs/allpods_20261025/    五场景输入与转换脚本（运行输出本地生成）
```

---

## 三、环境要求

新 GPU 入口**只使用 Python 标准库**，推荐 Python 3.11，最低 Python 3.8。本机已在 **Python 3.12.7 / Windows** 上验证。

不需要安装 TensorFlow、PyTorch、Gym、stable-baselines 就能运行 GPU 页面或命令行入口。

`conda_env.yml` 和 `require_sched.txt` 是原版 VMAgent 训练环境的依赖（其中 `numpy==1.16.5`、`gym==0.10.9` 等版本很旧），与本次 GPU 入口无关；不要为了运行 GPU 环境去强行安装它们。

---

## 四、快速开始

以下命令都在仓库根目录执行。

### 1. 运行测试

```bash
python -m unittest discover -s tests -p "test_gpu*.py" -v
```

本机结果：**15 个测试全部通过**，覆盖到达/等待/完成、CPU 阻塞、型号拒绝、单节点超限、FCFS/回填差异、背景、时间上限、真实场景、网页上传/运行/下载等。

### 2. 命令行运行内置场景 A

```bash
python -m vmagent.gpu.run --clock fast --policy first_fit --output outputs/gpu/A-first-fit
```

默认使用：

- 节点：`docs/server_selection/servers_20_H20.csv`（20 台 H20，8 卡/192 核）
- 任务：`docs/server_selection/real_pod_sequences/scene_A_timing/scene_A_50_pods_timed.json`（50 个带时间的 Pod）
- 时钟：`realtime`（命令行默认；上面显式改成 `fast` 便于快速检查）

查看全部参数：

```bash
python -m vmagent.gpu.run --help
```

### 3. 启动网页界面

最简方式：双击项目根目录的 `打开前端.bat`。它会检查服务是否已在运行；没有就启动，然后自动用默认浏览器打开 `http://127.0.0.1:8765/`。启动后任务栏会出现一个最小化的 `VMAgent Dashboard` 命令行窗口，**不要关闭它**，关掉服务就停止。

也可以手动启动：

```bash
python -m vmagent.gpu.dashboard --port 8765
```

然后打开：

```text
http://127.0.0.1:8765/
```

页面支持：

- 使用内置场景 A；
- 分别上传节点 CSV 和任务 CSV/JSON；
- 设置策略、时钟、队列规则、测量窗口、模拟上限、CPU 背景比例；
- 查看运行中状态、资源曲线、队列长度、任务明细；
- 下载 `summary.json`、`tasks.csv`、`events.csv`、`resource_history.csv`、`nodes_final.csv`、`run_manifest.json`。

网页服务只监听本机，不需要账号，也不会上传数据到远程。

---

## 五、输入格式

### 1. 节点 CSV

必需字段：

```csv
server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores
server_01,cluster_demo,H20,8,192
server_02,cluster_demo,H20,8,192
```

约束：

- `server_id` 不可重复；
- 所有节点必须属于同一个 `cluster_id`；
- `gpu_count`、`cpu_capacity_cores` 必须为正数；
- `gpu_spec_public` 是 GPU 型号标签（如 H20）；
- `asw_id`、快照日期等额外列可以存在，但不进入调度约束。

### 2. 任务 CSV / JSON

必需字段：

```csv
pod_id,gpu_spec_public,gpu_request,cpu_request_cores,submit_time_sec,service_sec
pod_01,H20,0.19,5,0,68.214
pod_02,H20,2,8,0.3,12.5
```

约束：

- `pod_id` 在本次输入中唯一；同一个 Pod 的跨小时重复观测必须先在上游去重；
- `submit_time_sec` 为相对实验开始的提交时刻，允许 0，不可为负；
- `service_sec` 必须大于 0；它是“实际开始执行后需要的服务时长”，不是“提交到完成”的墙钟跨度；
- `gpu_request`、`cpu_request_cores` 可为小数；
- `gpu_request>0` 的任务必须提供 `gpu_spec_public` 或 JSON 的 `allowed_gpu_types`；
- 如果任务带 `cluster_id`，必须与节点池的 `cluster_id` 一致；
- 原始的 `server_id` 只作为来源记录，不绑定新实验的目标节点；
- 不能直接把“只有资源需求、没有 `submit_time_sec`/`service_sec`”的清单当作可运行任务表。

JSON 可以是任务数组，也可以是 `{"tasks": [...], "measurement_window_sec": [0, 120], "gpu_capacity": 160, "cpu_capacity_cores": 3840}` 这种对象。若 JSON 中声明了容量，必须与节点表一致。

---

## 六、调度模型与规则

### 1. 资源账本

- 每台节点维护 `free_gpu` 和 `free_cpu`；
- GPU 请求按“等价份额”扣减，支持小数；
- 任务必须同一台节点同时满足 GPU 与 CPU 请求；
- 每次分配/释放都做资源守恒检查；
- 完成后只释放该任务实际占用的资源。

### 2. 时间与队列

- 任务在 `submit_time_sec` 到达；
- 到达时若暂时没有可用节点，进入等待队列，不预占资源；
- 资源释放后重试等待队列；
- 同一模拟时刻先处理完成释放，再处理新到达，然后放置；
- `backfill`：按等待顺序寻找可放任务，允许后来的小任务越过暂时放不下的任务；
- `fcfs`：严格等待队首，但永久不可能的任务仍会被拒绝，避免堵死队列。

### 3. 拒绝条件

任务在空节点上也无法满足时，标记 `rejected`，原因包括：

- `gpu_model_unavailable`：没有匹配型号的节点；
- `gpu_request_exceeds_single_node`：GPU 请求超过单节点容量；
- `cpu_request_exceeds_single_node_after_background`：扣除 CPU 背景后超过单节点容量；
- `joint_request_never_fits_one_node`：GPU/CPU 单独都够但联合放不下。

本版不做跨节点 gang placement。

### 4. 放置策略

- `first_fit`：第一个可用节点；
- `best_fit`：分配后归一化剩余 GPU+CPU 之和最小的节点；
- `worst_fit`：剩余资源最多的节点；
- `random`：带种子的随机可用节点。

### 5. 时钟

- `fast`：直接推进模拟时间，不等待墙钟；适合功能检查、批量比较。
- `realtime`：按墙钟节奏推进，`--speed 10` 表示 1 秒墙钟推进 10 秒模拟时间；系统唤醒和调度计算会带来少量时间差。

### 6. 测量窗口

`--measurement-seconds` 设置 `[0, seconds]` 的测量窗口；也可在任务 JSON 中提供 `measurement_window_sec`。提供负载和资源预留率都按该窗口积分。

- `offered_gpu_ratio` / `offered_cpu_ratio`：按输入的提交时间与服务时间计算的“理想提供负载”；
- `reserved_gpu_ratio` / `reserved_cpu_ratio`：实际分配/释放事件的积分；
- 两者都不是 GPU SM 利用率或 CPU 实际执行利用率。

---

## 七、本次接入的五个全真实 Pod 场景

本次部署把本地场景目录 `vmagent_scenarios_warmup_allpods` 的五套场景接入了本仓库。

### 1. 数据位置

```text
runs/allpods_20261025/
├─ prepare_inputs.py          场景转换脚本
├─ inputs/
│  ├─ nodes.csv               20 台 H20，8 卡/224 核，共 160 GPU/4,480 核
│  ├─ jobs_A.csv ... jobs_E.csv
│  └─ conversion_manifest.json
└─ outputs/                    本地生成，不提交
   ├─ A-first_fit/ ... E-first_fit/
```

源场景说明见该目录下的 `README.md`。这些场景与仓库内置场景 A 不是同一组节点：

- 内置场景 A：cluster `1f28e063...`，20 台 H20，8 卡/192 核，共 3,840 核；
- 本次五场景：cluster `9089d80...`，20 台 H20，8 卡/224 核，共 4,480 核。

环境从 `nodes.csv` 读取容量，因此 224 核/节点可以正常运行；`README_GPU.md` 里写的 192 核是内置示例的口径。

### 2. 运行参数

以 A 为例：

```bash
python -m vmagent.gpu.run \
  --nodes runs/allpods_20261025/inputs/nodes.csv \
  --tasks runs/allpods_20261025/inputs/jobs_A.csv \
  --clock fast \
  --policy first_fit \
  --queue-policy backfill \
  --measurement-seconds 21600 \
  --max-sim-time 150000 \
  --output runs/allpods_20261025/outputs/A-first_fit
```

B–E 只需把 `jobs_A.csv` / `A-first_fit` 换成对应场景。

- `--measurement-seconds 21600`：这五个场景的测量窗口都是 6 小时（21,600 秒）；
- `--max-sim-time 150000`：任务最长完成时刻约 101,760 秒（B 场景），默认 600 秒会截断，必须放大；
- `--clock fast`：功能验证用快速时钟；`realtime` 在 `--speed 1` 下 B 场景约需 28 小时墙钟。

### 3. 本机运行结果

| 场景 | 任务数 | 完成 | 拒绝 | 未完成 | 提供GPU | 提供CPU | 实际GPU预留 | 实际CPU预留 | 结束状态 |
| ---- | ------ | ---- | ---- | ------ | ------- | ------- | ----------- | ----------- | -------- |
| A    | 78     | 78   | 0    | 0      | 29.4%   | 31.3%   | 29.4%       | 31.3%       | drained  |
| B    | 223    | 223  | 0    | 0      | 74.0%   | 53.8%   | 74.0%       | 53.8%       | drained  |
| C    | 533    | 533  | 0    | 0      | 28.4%   | 41.3%   | 28.4%       | 41.3%       | drained  |
| D    | 505    | 505  | 0    | 0      | 91.6%   | 89.3%   | 91.6%       | 89.3%       | drained  |
| E    | 115    | 115  | 0    | 0      | 52.3%   | 44.4%   | 52.3%       | 44.4%       | drained  |

五个场景均正常排空，结束后 20 台节点的 GPU/CPU 全部释放；任务明细全部为 `completed`。所有任务平均等待 0 秒——当前负载下 First Fit 可以立即放置，瞬时队列峰值来自同一时刻到达的预热任务。

### 4. 场景备注

- A、D、E 分别对应“低GPU/低CPU、高GPU/高CPU、中GPU/中CPU”；
- B、C 受真实数据限制只能近似（B 中/中，C 低/中），原因见场景 README 的“真实数据限制”一节；
- 场景 C/D 的 CPU-only 任务较多，逐 Pod 仿真会产生较多事件；如果只关心 CPU 压力，也可以在上游把 CPU-only 聚合为逐节点背景。

---

## 八、输出文件怎么看

每次运行输出 6 个文件：

| 文件                     | 内容                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------ |
| `summary.json`         | 任务计数、完成/拒绝/未完成、模拟与墙钟时间、提供负载、资源预留率、等待统计、峰值队列 |
| `tasks.csv`            | 每个 Pod 的最终状态、分配节点、提交/开始/完成时间、等待时间、拒绝原因                |
| `events.csv`           | 到达、开始、完成、拒绝、结束事件                                                     |
| `resource_history.csv` | 事件时间线上的 GPU/CPU 预留、等待队列、完成数                                        |
| `nodes_final.csv`      | 运行结束时各节点剩余 GPU/CPU；正常排空应恢复满容量                                   |
| `run_manifest.json`    | 本次输入文件、策略、队列规则、随机种子、任务元数据                                   |

常见判断：

- `end_reason = drained` 且 `unfinished = 0`：所有任务已完成或已明确拒绝，正常排空；
- `end_reason = max_sim_time`：达到模拟时间上限，仍有未完成任务；
- `rejected > 0`：查看 `tasks.csv` 的 `rejection_reason`；
- `reserved_*` 是资源预留率，不是实际利用率；
- `offered_*` 是输入的理想提供负载，可能高于实际预留率（有等待时）。

---

## 九、常见问题

**No module named vmagent**
请从仓库根目录运行 `python -m vmagent.gpu.run ...`，不要进入 `vmagent/` 子目录运行。

**默认 600 秒不够**
这批场景的完成时刻远超 600 秒；用 `--max-sim-time 150000` 或按最长 `submit+service` 设置。

**网页打不开**
确认启动终端仍在运行，端口与 URL 一致；默认 `http://127.0.0.1:8765/`。如果端口被占用，换 `--port 8766`。

**任务被拒绝**
看 `tasks.csv` 的 `rejection_reason`。单任务必须能放进一台节点，不能只看 20 台的总容量。

**B–E 能直接跑吗**
本仓库的 `runs/allpods_20261025/inputs` 已经为五套场景生成好 `submit_time_sec` 和 `service_sec`，可以直接跑。仓库内置的 B–E 原始需求清单没有这两个字段，不能直接运行。

**为什么 fast 和 realtime 结果略有差异**
fast 使用精确事件时刻；realtime 包含线程唤醒、调度计算和系统计时粒度，短任务会有毫秒级差异。

**旧 DQN/PPO/SAC 能直接用吗**
不能。旧模型的动作空间是 2N 个 NUMA 动作，新环境是 N 个服务器动作；需要适配网络、掩码、终止转移和奖励后重新训练。

---

## 十、测试与复现

```bash
# GPU 环境、网页和真实场景测试
python -m unittest discover -s tests -p "test_gpu*.py" -v

# 重新生成五场景输入
python runs/allpods_20261025/prepare_inputs.py

# 运行五场景（示例：A）
python -m vmagent.gpu.run \
  --nodes runs/allpods_20261025/inputs/nodes.csv \
  --tasks runs/allpods_20261025/inputs/jobs_A.csv \
  --clock fast --policy first_fit --queue-policy backfill \
  --measurement-seconds 21600 --max-sim-time 150000 \
  --output runs/allpods_20261025/outputs/A-first_fit
```

---

## 十一、已知限制

1. **不是 GPU 执行模拟器**：GPU 请求只是资源账本中的份额，不模拟 SM、显存带宽、算子性能或多卡通信。
2. **不建模显存、RAM、NUMA、ASW**：这些字段即使出现在输入中，也不进入约束。
3. **不做 gang scheduling**：多机任务不会被拆成跨节点 gang；单个 Pod 必须能放进一台节点。
4. **CPU-only 任务时长可能是近似值**：本次五场景的 CPU-only Pod 不在执行汇总表，使用观测出现时段作为服务时长；详见场景 README。
5. **fast 与 realtime 的时间语义不同**：fast 用于逻辑验证；realtime 用于观察真实节奏，不能保证亚毫秒精度。
6. **旧 RL 训练代码未适配新动作空间**：需要重新训练，不能直接加载原权重。
7. **场景负载是请求口径**：`offered_*` 不是实测利用率；>100% 表示需求超容，不表示实际运行到 100% 以上。

---

## 十二、来源与许可

- 原始项目：https://github.com/mail-ecnu/VMAgent
- 本分支：https://github.com/vmagentpromax/vmagentpro
- 初始版本基于 commit：`cb828fdd153afae208558d006acbc59ddb2e64c5`；后续中文文档与前端改版在 `main` 分支。
- 许可证：MIT（见 [LICENSE](LICENSE)）

本中文 README 由本次部署补充；`runs/allpods_20261025/` 中的 inputs 与转换脚本随仓库提供，outputs 为本地生成、未提交。原始 `README.md`、`README_GPU.md` 和 `docs/server_selection/` 中的说明仍然有效。
