# 三天真实 H20 Pod：五组资源需求清单

## 结论与解释边界

已从 `day=109～111`、`cluster_id=1f28e063c4e5e1bc48ff55269617dd2f` 的真实 H20 Pod 记录中选出五组请求，共 270 个不同 `pod_id`。每组都是某个小时快照的真实 Pod 子集；没有修改 GPU/CPU 请求，没有合成任务身份、到达时间、执行时长或 CPU 背景。

**这份交付确定的是五种资源需求组合，没有证明附件定义的五种时间平均提供负载已经存在于真实连续窗口。**清单中的行顺序也不是恢复出的真实提交顺序。仅凭 GPU/CPU 请求字段无法计算任务在时间窗口内消耗的 GPU 时间和 CPU 核时间。

附件中的提供负载需要到达时间、服务时长和窗口积分。公开小时表缺少准确提交时刻；执行汇总没有 CPU 请求，且身份与执行段的关联还需另行核验。因此这里没有把需求总和冒充时间平均负载，也没有给任务补假时间。

## 五组结果

节点池固定为先前选定的 20 台 H20/8GPU/192CPU，总容量 **160 GPU、3,840 CPU 核**。

- **A，GPU低、CPU低**：50 个 Pod，来自 day=109/hour=00。GPU 请求总和 48，CPU 请求总和 1,152 核，分别为总容量的 30% 和 30%。见 `scene_A_real_H20_pods.csv`。
- **B，GPU高、CPU低**：50 个 Pod，来自 day=109/hour=01。GPU 请求总和 144，CPU 请求总和 1,152 核，分别为总容量的 90% 和 30%。见 `scene_B_real_H20_pods.csv`。
- **C，GPU低、CPU高**：70 个 Pod，来自 day=109/hour=14。GPU 请求总和 48，CPU 请求总和 3,456 核，分别为总容量的 30% 和 90%。见 `scene_C_real_H20_pods.csv`。
- **D，GPU高、CPU高**：50 个 Pod，来自 day=110/hour=01。GPU 请求总和 144，CPU 请求总和 3,456 核，分别为总容量的 90% 和 90%。见 `scene_D_real_H20_pods.csv`。
- **E，GPU中、CPU中**：50 个 Pod，来自 day=111/hour=00。GPU 请求总和 96，CPU 请求总和 2,304 核，分别为总容量的 60% 和 60%。见 `scene_E_real_H20_pods.csv`。

上述百分比统一按 `Σgpu_request / 160`、`Σcpu_request_cores / 3840` 计算，是这批请求的总量相对节点池容量的比例，不是观测利用率，也不是任务依次提交后的平均资源占用。源表记录一小时内观测到的 Pod，不能仅凭同小时出现证明它们在某个瞬间同时运行。

## 主要文件和字段

- `all_scenes_real_H20_pods.csv`：全部 270 个 Pod 的总清单。
- `scene_A_real_H20_pods.csv` 至 `scene_E_real_H20_pods.csv`：分组清单。
- `scene_request_evidence.json`：每组计数、总量、请求形状分布和解释限制。
- `original_source_audit.json`：逐条回查原始 Parquet 的结果。

任务清单主要字段：

- `pod_id`：原始匿名 Pod ID，组内和组间均唯一。
- `gpu_spec_public`：原始型号标签，全部为 H20；它不是公开的完整 affinity 配置。
- `gpu_request`：原始 GPU 数量/份额，保留小数。
- `cpu_request_cores`：同一原始记录中成对的 CPU 核需求，保留小数。
- `source_day`、`source_hour`：该条记录的实际来源小时。
- `cluster_id`、`server_id`：该小时的原始集群及宿主，宿主不限定为先前抽出的 20 台。
- `job_type_public`、`priority_class`：原始业务类型和优先级。
- `scene`：本次筛选分组 A～E，是新增整理标签。
- `row_index`：本次清单行号，不是原始提交顺序。
- `first_observed_day`、`first_observed_hour_of_day`：该 Pod 在本次 72 小时 H20/GPU正请求记录中首次被观察到的位置，不是创建或提交时间。

## 筛选与去重方法

1. 扫描 day=109～111 共 72 个 Pod 分区，按 Pod 行自身的 cluster_id 筛选目标集群，再筛选 H20 且 GPU 请求大于 0 的原始行。
2. 共读取 429,168 条满足条件的 Pod 小时观测，对应 78,538 个不同 pod_id。
3. 在上述 H20/GPU正请求观测内，181 个 Pod 的 GPU 或 CPU 请求值不一致。本次排除这些 Pod，得到 78,357 个需求稳定的候选；这不表示核查过它们在其他机型、零 GPU 行或三天以外的所有记录。
4. 对每组指定来源小时，按真实 `(gpu_request,cpu_request_cores)` 联合形状求解非重复子集，使请求总量达到对应比例。为控制首轮规模，任务数限制在 50～150，优先选择较少任务；最终 A/B/D/E 各 50 个，C 为 70 个。
5. 在同形状候选内，按 `SHA256('42:' + pod_id)` 选定具体身份。按 C、B、D、A、E 顺序预留 ID，之后的分组不能再使用已选 ID。
6. 行顺序按窗口内首见小时排列，同小时以 pod_id 打破并列。这个顺序仅便于整理；没有恢复秒级到达时间或真实同小时先后次序。
7. 验证所有 ID、型号、资源配对、集群和宿主与来源小时记录一致，验证 270 个 ID 全局去重。另在允许分数资源的 CPU/GPU 数量模型中，为每组找到可放入 20 台空节点的假设分配；该分配只做容量可行性检查，不是历史真实调度。

这些是按目标资源结构定向筛选的真实请求子集，组成不代表原始集群的自然分布。由于用户要求同一任务不能重复，A/C、B/D 使用不同 Pod，不能把比较解释为只改变 CPU 背景的严格配对实验。

## 原始数据与复算

源路径：`E:\cluster-trace-gpu-v2026\data\asi_opensource_pod_hourly\asi_opensource_pod_hourly\day=<day>\hour=<hour>\*.parquet`。

复算：先运行上一级 `scan_real_pods.py`，再运行 `select_real_request_groups.py`。前者读取原始三天数据，后者选择请求集合；仅写入本文件夹，不修改原始数据或仿真器。

参考用户附件《按集群负载构造仿真场景_简洁版(5).md》的五场景定义，以及 [官方字段说明](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md)。
