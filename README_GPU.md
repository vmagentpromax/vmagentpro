# VMAgent H20 / CPU 调度版：修改步骤与详细使用说明

新增两组连续窗口及预热实验：见 [两个GPU场景跑通说明](docs/server_selection/two_gpu_scenes/README.md)。支持原20台节点上的 FirstFit / BestFit；网页现可设置非零测量开始时间，并在上传 JSON 时读取其测量窗口。

这版接入了已选的 20 台 H20 服务器和场景 A 的 50 个 Pod，支持按提交时间到达、按 CPU/GPU 需求放置、等待资源、按服务时间完成并释放资源。提供命令行和本地网页两种入口；网页可以上传数据、查看运行过程和下载结果。

## 1. 本次主要修改了什么

### 第一步：读取真实服务器与任务数据

新增 `vmagent/gpu/environment.py`，提供 `load_nodes()` 和 `load_workload()`：

- 按字段名读取 UTF-8 CSV / JSON，允许 CSV 带 BOM；保留小数 GPU 与 CPU 请求。
- 服务器必须属于同一 `cluster_id`，每个 `server_id` 唯一；任务 `pod_id` 必须唯一。
- 缺字段、重复 ID、负数、非有限数字、非正服务时长直接报错，不静默丢行或补值。
- GPU 型号作为节点资格检查，不把不同型号视作可互换的卡。
- 从任务文件中的 `submit_time_sec` 和 `service_sec` 运行，不在调度环境里再次生成随机到达或缩放时长。

### 第二步：改为整机 CPU + GPU 的资源账本

每个节点单独维护 GPU 总量、CPU 整机核数、剩余 GPU、剩余 CPU。一个任务须在同一台合资格服务器上同时满足两项请求才能运行。

当前每台 H20 为 8 GPU、192 CPU 核，20 台合计 160 GPU、3,840 CPU 核。CPU 不按 NUMA 复制计算；GPU 小数按“等价份额”扣减。显存、RAM、NUMA、ASW 不进入本版约束，也不假设真实设备的 GPU 共享隔离机制已经被模拟。

每次分配和释放都会核对资源守恒。任务结束时只释放它实际分配的资源。

### 第三步：加入时间推进、任务完成与等待队列

新增到达事件和完成事件。任务开始之前只等待，不预占资源；完成时刻按 **实际开始时刻 + service_sec** 计算。提交后因为资源不足而等待的任务，不会在“提交时刻 + 服务时间”被提前释放。

默认使用 `realtime`：一秒模拟时间对应一秒墙钟时间；等待未来事件时程序会按单调时钟等待，调度计算与系统唤醒延迟会体现在开始和完成记录里。它是按时间节奏运行的资源模拟，不会在 H20 上执行真实模型代码。

另有 `fast`：使用同一套到达、分配和释放逻辑，直接推进模拟时间，用于功能检查和批量比较。它不会修改输入文件的时间。

### 第四步：处理暂时等待与永久不可满足

- 暂时没有资源：保留在等待队列，资源释放后重试。
- 型号不存在、请求超过单节点容量、扣除固定背景后永远无法放置：标记 `rejected` 并记录原因，继续处理后续任务。
- 默认队列为 `backfill`：按等待次序查找可放任务，允许后来的小任务越过暂时放不下的任务。
- 可切换 `fcfs`：严格等待队首。永久不可能的任务仍会被拒绝，避免永久堵住队列。
- 同一模拟时刻先处理完成释放，再处理新到达，随后开始放置。
- 达到时间上限或用户停止后，保留尚未到达、等待中、运行中的状态，计入未完成任务。

### 第五步：增加放置策略与策略接口

内置 First Fit、Best Fit 和带种子的 Random。First Fit 取第一个可用节点；Best Fit 在可用节点中选择分配后归一化剩余 GPU 与 CPU 之和最小的节点。

环境每次动作选择一台服务器，20 台节点对应 20 个动作。`observation()` 提供节点剩余资源比例、当前任务的 GPU/CPU 请求、节点型号、可行动作掩码和队列长度，不向策略输入未来真实服务时长或完成时间。

`schedgym/gpu_env.py` 提供新环境导入入口。原有训练网络的 `2N` 个 NUMA 动作、CPU/内存状态形状和 CUDA 依赖与本环境不相同；当前完成了可运行的 GPU 环境、三种启发式策略、数据接口和可视化，旧 DQN/PPO/SAC 模型不能直接加载到新动作空间。

### 第六步：增加网页导入、可视化与结果导出

`vmagent/gpu/dashboard.py` 提供仅本机访问的 HTTP 服务。`vmagent/gpu/static/` 包含网页、样式、刷新与绘图逻辑，不依赖 Node 构建、外部 CDN 或数据库。

上传后的数据保存到本机 `outputs/gpu/dashboard/uploads/`。每次网页实验使用独立目录保存结果；运行中每 0.5 秒刷新一次状态。一次只允许一个网页实验运行。

## 2. 启动前准备

新入口仅使用 Python 标准库，推荐使用已测试的 Python 3.11；最低 Python 3.8。不需要安装原版训练环境的 TensorFlow、Torch、Gym 或 stable-baselines 才能运行本次 GPU 页面。

在 PowerShell 中进入项目根目录：

```powershell
Set-Location -LiteralPath 'D:\一般文件\VMAgent'
python --version
```

如果 `python` 不是你希望使用的解释器，可使用本机已验证的解释器：

```powershell
& 'C:\Users\11935\anaconda3\python.exe' -m vmagent.gpu.dashboard
```

以下命令都应从项目根目录运行。

## 3. 最快开始：打开可视化界面

### 3.1 启动服务

```powershell
python -m vmagent.gpu.dashboard --port 8765
```

终端出现 `Open http://127.0.0.1:8765` 后，在浏览器打开：

**http://127.0.0.1:8765/**

保持启动服务的终端开启。关闭网页不会停止服务器或正在运行的实验；网页重新打开后可以继续查看当前实验。终端按 `Ctrl+C` 关闭网页服务前，建议先在页面点“停止”，等结果保存完成。

如果 8765 端口已被已有服务使用，可以直接打开已有页面，或另用端口：

```powershell
python -m vmagent.gpu.dashboard --port 8766
```

随后访问 `http://127.0.0.1:8766/`。服务仅监听本机，不需要账号，也不会上传到远程平台。

### 3.2 先用内置数据运行一次

1. 左侧点击“使用内置场景 A 数据”。
2. 保留 First Fit、真实时间节奏、允许后续可放任务先运行。
3. 高级设置保留：随机种子 42、测量窗口 120 秒、模拟时间上限 600 秒、CPU 背景比例 0。
4. 点击“开始实验”。
5. 默认场景的最后一个任务约在 16.23 秒到达，无排队理想完成时刻约 105.74 秒；真实时间模式会有系统计时和调度开销，通常约两分钟。
6. 等待“已完成 / 总任务”为 `50 / 50`，运行中和等待中都为 0。然后下载运行汇总和任务明细。

内置数据实际路径：

```text
docs/server_selection/servers_20_H20.csv
docs/server_selection/real_pod_sequences/scene_A_timing/scene_A_50_pods_timed.json
```

## 4. 数据如何上传 / 导入

你可以直接使用内置数据，也可以在页面上传自己的服务器表和任务表。两份文件分别上传，不能把服务器文件放进任务上传框。

### 4.1 上传当前已准备好的数据

服务器选择：

```text
D:\一般文件\VMAgent\docs\server_selection\servers_20_H20.csv
```

任务选择 JSON 或对应的 timed CSV，两者都可以：

```text
D:\一般文件\VMAgent\docs\server_selection\real_pod_sequences\scene_A_timing\scene_A_50_pods_timed.json
D:\一般文件\VMAgent\docs\server_selection\real_pod_sequences\scene_A_timing\scene_A_50_pods_timed.csv
```

操作步骤：

1. 点击“上传服务器表”下面的“选择文件”，选中服务器 CSV。
2. 等待提示导入成功，并确认显示 20 条记录。
3. 点击“上传任务表”下面的“选择文件”，选中 timed JSON 或 timed CSV。
4. 等待显示 50 条记录。
5. 确认参数，再点击“开始实验”。

单个文件界面限制为 10 MiB，编码为 UTF-8；Excel 导出的“CSV UTF-8”可以使用。原始大型 Parquet 不直接上传到网页，先提取成节点表和带时间字段的任务表。上传文件只保存在本机；原始文件不会被修改。

页面中的上传选择在服务本次启动期间有效。重启服务后，可重新上传同一文件；已保存的运行结果和上传文件仍留在输出目录。页面默认数据按钮会同时恢复节点表和任务表。

### 4.2 自定义服务器 CSV 的必需字段

```csv
server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores
server_01,cluster_demo,H20,8,192
server_02,cluster_demo,H20,8,192
```

这是格式示例，不是额外生成的真实服务器。`cpu_capacity_cores` 为整机 CPU 核数。服务器 ID 不可重复，所有服务器须同集群。`asw_id`、快照日期等额外字段可以存在，但不进入调度约束。

JSON 可以使用相同字段的数组，或包在 `{"nodes": [...]}` 中。

### 4.3 自定义任务 CSV 的必需字段

```csv
pod_id,gpu_spec_public,gpu_request,cpu_request_cores,submit_time_sec,service_sec
pod_01,H20,0.19,5,0,68.21436403508773
pod_02,H20,2,8,0.3,12.5
```

这同样只是字段格式示例。实际实验应使用提取的真实 ID 和资源搭配。

- `submit_time_sec`：相对本次实验开始的提交时间，允许为 0，不可为负。
- `service_sec`：实际开始执行后需要的服务时间，必须大于 0。
- `gpu_request`、`cpu_request_cores`：原始资源需求，保留小数，不取整。
- `gpu_spec_public`：例如 H20。JSON 也可用 `allowed_gpu_types: ["H20"]`。
- `pod_id`：每个任务唯一，不把同一 Pod 的跨小时重复观测作为新任务。
- `cluster_id` 如存在，必须与所选节点集群一致；任务的原始 `server_id` 作为来源记录，不绑定新实验的目标节点。
- 其余来源和时长歧义字段可以保留，运行结果会保留 `duration_match_status`。

**不能直接上传 `scene_A_real_H20_pods.csv` 或 B～E 的原始需求清单作为可运行任务表**，因为这些文件没有 `submit_time_sec` 和 `service_sec`。当前只有场景 A 已生成时间；B～E 需先完成执行段匹配和时间构造，再导入同一环境。

### 4.4 当前 A 场景时间的含义

50 个 Pod 的 GPU/CPU 需求来自真实记录，执行时长来自真实执行汇总，统一压缩 36,480 倍。24 个 Pod 有多个执行段，采用较长段作为服务时间代理并明确标记；不能宣称精确还原了对应历史执行实例。

到达时间采用均值 0.3 秒的指数间隔，种子 42，是合成仿真输入。环境直接使用文件中已生成的时间，**不再次缩放或随机生成**。详情见 `docs/server_selection/real_pod_sequences/scene_A_timing/README.md`。

## 5. 可视化界面怎么看

### 5.1 顶部进度

- “已完成 / 总任务”：完成数量和全部输入数量。
- “运行中”：已分配资源、尚未完成的任务。
- “等待中”：已经到达但尚未分配资源的任务。
- “永久不可满足”：型号不匹配或在空节点上也永远放不下的请求。
- “模拟时间 / 墙钟时间”：前者是实验内部时钟；后者是本次运行真实消耗的时间，结束后停止增长。

达到时间上限或点击停止，不意味着所有任务完成。查看结束原因、未完成数量和任务状态，不能只看页面“实验结束”。

### 5.2 两张曲线

“节点池资源预留率”显示已分配 GPU 和 CPU 的数量占节点池总容量的比例；CPU 包含显式设置的固定背景。曲线不是 GPU SM 利用率，也不是 CPU 实际执行利用率。

“等待队列长度”显示已到达且待调度的 Pod 数；任务到达后立即安排时也可能产生很短的队列尖峰。

页面另列“提供负载”和“资源预留率”：提供负载按输入的理想到达/运行区间积分；实际预留按真实分配与释放事件积分。运行中预留率按已观测的窗口部分显示，正常排空后按完整测量窗口显示；被用户停止或时间上限截断的结果按已覆盖时段统计，不把未观测部分当成空闲。不同分母不要直接比较。

### 5.3 服务器资源与任务明细

服务器表列出各节点剩余 GPU、剩余 CPU 和固定背景 CPU。正常排空后应恢复到原始容量（若设置背景，CPU 可用量仍扣除背景）。

任务明细可按状态筛选，查看到达、服务、开始、结束和等待时间。`finish_time_sec` 是实际完成记录，排队期间为空；“多段：最长段代理”保留原始时长匹配的不确定性。悬停“永久不可满足”状态可查看拒绝原因。

最近事件列表按时间倒序，帮助检查任务是否按时到达、是否等待过资源、完成后是否释放。

## 6. 下载结果与本地目录

实验结束或停止后，左侧出现下载链接：

- “运行汇总” → `summary.json`：计数、时间、等待、提供负载、资源预留率等。
- “任务明细” → `tasks.csv`：每个 Pod 的最终状态、分配节点、提交/开始/完成时间、等待及拒绝原因。
- “资源曲线” → `resource_history.csv`：完整事件时间线上的资源与队列数据。
- “事件日志” → `events.csv`：到达、开始、完成、拒绝及结束事件。
- “最终节点状态” → `nodes_final.csv`：核对资源释放。
- “输入与配置” → `run_manifest.json`：本次输入文件位置、策略、参数及任务元数据。

网页实验分别保存到：

```text
D:\一般文件\VMAgent\outputs\gpu\dashboard\runs\<run_id>\
```

网页会显示当前结果的绝对目录。文件在关闭页面或服务后仍保留。`outputs/gpu/` 已加入 `.gitignore`，上传和运行结果不会默认进入 Git 变更。

## 7. 命令行运行

默认真实时间模式：

```powershell
python -m vmagent.gpu.run --policy first_fit --output outputs/gpu/A-first-fit
```

快速检查场景 A：

```powershell
python -m vmagent.gpu.run --clock fast --policy best_fit --output outputs/gpu/A-best-fit-fast
```

指定自己的文件，路径有空格或中文时加引号：

```powershell
python -m vmagent.gpu.run --nodes "D:\一般文件\VMAgent\docs\server_selection\servers_20_H20.csv" --tasks "D:\一般文件\VMAgent\docs\server_selection\real_pod_sequences\scene_A_timing\scene_A_50_pods_timed.csv" --clock realtime --policy first_fit --output "outputs/gpu/A-csv"
```

严格 FCFS：

```powershell
python -m vmagent.gpu.run --queue-policy fcfs --clock fast --output outputs/gpu/A-fcfs
```

完整参数：

```powershell
python -m vmagent.gpu.run --help
python -m vmagent.gpu.dashboard --help
```

`--speed` 仅用于真实时间模式，默认 1；例如 10 表示 1 秒墙钟推进 10 秒模拟时间，不修改任务输入秒数。比较算法时固定输入文件、节点、种子、时间模式、背景、队列规则与测量窗。网页每次自动使用独立结果目录；命令行重复使用同一个 `--output` 会覆盖该目录中的同名结果文件。

## 8. 程序中接入自己的策略

```python
from vmagent.gpu.run import make_environment
from vmagent.gpu.environment import DEFAULT_NODES, DEFAULT_TASKS

env, metadata = make_environment(DEFAULT_NODES, DEFAULT_TASKS, clock="fast")
observation = env.reset()
while not env.done:
    # 在这里调用自己的策略；只选择 avail=1 的服务器索引。
    action = next(i for i, available in enumerate(observation["avail"]) if available)
    observation, reward, done, info = env.step(action)
env.export("outputs/gpu/my-policy")
```

观察中的 `obs` 是 N×2 的 `[剩余GPU/总GPU, 剩余CPU/总CPU]`；`feat` 是 `[GPU请求, CPU请求]`；`avail` 是 N 个节点的动作掩码。环境自行等待并推进到下一个可放置决策，策略不需要输出虚假的“拒绝”动作来跳过暂时缺资源的任务。`reward` 当前是每次成功开始任务 +1，尚未设计等待时间等 RL 奖励。

旧 `vmagent/train.py` 的网络维度和经验回放依赖旧状态接口。若后续接入 DQN/PPO/SAC，需要适配网络输入、N 个动作、掩码、终止转移和奖励，并重新训练；不能用原 NUMA 模型权重直接评估 H20 场景。

## 9. 已做的验证和预期结果

运行针对新增环境的验证：

```powershell
python -m unittest discover -s tests -p "test_gpu*.py" -v
```

覆盖任务等待后才开始计服务时间、CPU 阻塞、型号拒绝、单节点超限、FCFS 与回填差异、同刻先释放、时间上限保留未完成任务、背景只计一次、非法动作、去重与数值校验、真实 A 场景 CSV/JSON、一致释放、真实时间停止和运行结束后的墙钟冻结。另通过实际 HTTP 请求检查网页、静态资源、文件上传校验、运行、结果下载及停止操作。

场景 A 的 First Fit、Best Fit、Random 在快速模式下均完成全部 50 个任务、拒绝 0、未完成 0，结束时所有节点资源释放。模拟完成时刻约 105.740805 秒；GPU/CPU 提供负载约 5.7416% / 3.8292%，无等待时资源预留率与此相符。真实时间模式也完成过完整的 50 个 Pod；系统唤醒与调度计算会带来小幅时间差。

这批任务属于双低场景，不是严格 30%/30% 时间平均负载；原清单 30%/30% 指所有请求总和与容量的比例。

## 10. 常见问题

**提示 No module named vmagent：**先进入 `D:\一般文件\VMAgent` 再执行 `python -m ...`，不要进入 `vmagent` 子目录运行新入口。

**网页打不开：**确认启动终端仍在运行、端口与地址一致；默认网址是 `http://127.0.0.1:8765/`。第一次用快速模式检查也可以。若在受限执行环境中无法访问监听端口，直接在普通本机 PowerShell 启动上述服务，不需要修改防火墙或关闭安全设置。

**Duplicate pod_id / server_id：**输入含重复身份，请先在数据准备阶段去重和核对资源变化；环境不会把重复观测自动当作新任务。

**Missing / must be finite / service_sec 报错：**检查字段名、UTF-8 编码、空值、负数、NaN，以及是否误选了尚未配时间的需求表。

**某个任务被拒绝：**看任务明细的 `rejection_reason`。每个任务须能放在一台服务器上，不能仅因 20 台合计容量足够，就认为 16 GPU 的单个任务能放进某台 8 GPU 节点。本版不实现跨节点 gang placement。

**B～E 还不能直接运行：**它们目前只有真实需求字段；需要按真实执行段配好 `service_sec`、明确生成 `submit_time_sec` 后再上传。

**为什么 fast 和 realtime 时间略有差异：**fast 使用精确事件时刻；realtime 包含线程唤醒、调度计算和操作系统计时粒度。这批输入中有亚毫秒服务时间，真实时间模式不能保证亚毫秒精确执行；程序保留原值，不擅自拉长短任务。

**终止后任务还在 running / waiting：**表示实验被停止或达到上限，不是假装任务已完成；查看 `unfinished` 和 `end_reason`。正常 `drained` 排空应为 0。

**修改网页后没更新：**刷新网页；如果修改了 Python 后端，先停止实验，再在启动终端按 Ctrl+C 后重启服务。

## 11. 代码位置

```text
vmagent/gpu/environment.py    数据读取、GPU/CPU模型、时钟、队列、完成释放、统计
vmagent/gpu/run.py            命令行实验入口
vmagent/gpu/dashboard.py      本地上传、后台运行、状态与下载接口
vmagent/gpu/static/           可视化网页
schedgym/gpu_env.py           新环境导入入口
tests/test_gpu_environment.py 关键行为验证
tests/test_gpu_dashboard.py   上传、运行、停止与下载验证
README_GPU.md                本文：修改步骤与详细使用说明
```
