# 首轮实验：同一 cluster 的 20 台 H20 服务器

本轮场景已按“先小时请求指标、后窗口、再时长和到达”的顺序重建；沿用本文件的原 20 台节点。方法见 [request_load_scenes/METHOD.md](request_load_scenes/METHOD.md)，结果与复现入口见 [request_load_scenes/README.md](request_load_scenes/README.md)。

选择结论：使用 `cluster_id=1f28e063c4e5e1bc48ff55269617dd2f` 中的 20 台 H20 服务器，每台 8 GPU、192 CPU 核，总容量 160 GPU、3,840 CPU 核。

本次已读取用户分享聊天。实验背景是先以 20 台服务器构建小规模 CPU/GPU 混合资源压力实验；分享聊天中的 A10/8卡/192核只是举例，不作为已核实配置。

## 为什么选 H20

- **A，论文依据**：本地论文 PDF 第 7 页（§3.3，CPU/GPU ratio）明确举例 8-GPU H20 / 192 CPU 核，以及 1-GPU A10 / 128 CPU 核。第 8 页（§4.1，Fig.13）讨论 CPU 不足和节点间 GPU 碎片，使用 1、4、8 GPU 等请求配置分析。这支持选择同时具有多卡容量和 CPU 约束的节点作为首轮实验对象；论文没有直接推荐这 20 台节点。
- **本地原始数据核实**：在 `day=109/hour=00`，目标 cluster 中有 1,186 台 H20/8卡/192核节点；其中 1,181 台在 `day=109/hour=00` 至 `day=111/hour=23` 的全部 72 个小时均被观测到且 cluster、GPU 型号、卡数、CPU 核数一致。
- **本地原始数据核实**：同一 cluster、同一小时的 H20 Pod 中，GPU 请求大于 0 的去重 Pod 有 5,441 个。其中 1卡 3,914 个、2卡 224 个、4卡 186 个、8卡 627 个；另有分数和其他卡数请求。同一组 H20 Pod 中还有 GPU 请求为 0 且 CPU 请求大于 0 的 Pod 12,331 个。
- **本地原始数据核实**：5,441 个 GPU Pod 中，在线推理 2,360 个、离线推理 1,947 个、训练 519 个、开发 430 个、other 51 个、unknown 134 个。这是单小时观测组成，不是新到达任务数，也不是六个月的任务池规模。
- **C，实验设计判断**：每台 8 卡可以容纳 1/2/4/8 卡任务，让少量多卡任务产生较高请求压力，并支持研究节点内放置与节点间碎片。每卡对应 24 核 CPU，相较常见 1卡/128核 A10 更容易在混合背景下观察 CPU 约束。H20 是针对本轮实验目标的选择，不代表对所有任务或硬件性能最优。

## 清单与可复现方法

完整机器清单见 [servers_20_H20.csv](servers_20_H20.csv)，保留 `server_id`、`cluster_id`、`gpu_spec_public`、`gpu_count`、`cpu_capacity_cores`、`asw_id`、快照时间与核验小时数。

候选集先按连续 72 小时配置一致性筛选，再按 `SHA256('42:' + server_id)` 升序取前 20 台。不以负载高低挑选节点。ID 是发布方提供的匿名标识，不能解释为真实主机名。

运行 `python docs/server_selection/select_servers.py` 可复算清单及 [selection_evidence.json](selection_evidence.json)。脚本仅需 pandas、pyarrow；可用 `--data` 指定数据根目录、`--out` 指定输出目录。

所选 20 台在基准小时上关联到 458 个去重 Pod，其中 118 个 GPU 请求大于 0；这仅验证节点和 Pod 关联，并未提取任务序列或推断实时占用。

## 使用边界

这 20 台属于同一 cluster，基准小时分布在 10 个 ASW 域，没有增加“同一 ASW”的筛选条件。若后续研究跨节点通信，需要另外处理拓扑约束。72 小时持续出现只说明库存观测及配置一致，不能证明节点一直健康、空闲或可立即分配。

后续各场景可以固定这组容量。若从空集群开始，这是使用真实配置构造仿真；若回放原有运行状态，还应重建初始存量。不得把 160 GPU 当成基准小时的空闲 GPU 数。

公开 `gpu_spec_public` 是型号标签，没有直接暴露 Pod 的硬性 affinity 规则；后续抽取 H20 任务模板时，应将“按 H20 标签过滤”与“仿真中要求 H20”的设定区分。CPU-only Pod 的执行时长不能直接从仅覆盖 GPU 请求的执行汇总表取得。

## 数据路径与一手资料

- 服务器：`E:\cluster-trace-gpu-v2026\data\asi_opensource_server_hourly\asi_opensource_server_hourly\day=<day>\hour=<hour>\*.parquet`
- Pod：`E:\cluster-trace-gpu-v2026\data\asi_opensource_pod_hourly\asi_opensource_pod_hourly\day=109\hour=00\*.parquet`；该小时按 `pod_id` 保留首条，去除 1 条重复观测。
- [分享聊天](https://chatgpt.com/share/6adcaa27-8234-83ec-9346-462744d200bc)
- [论文官方页面](https://www.usenix.org/conference/osdi26/presentation/li-suyi)
- [官方数据集](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026)
- [官方字段定义](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md)

时间是 trace 相对 day/hour，不能转换为真实日历日期。这里仅完成节点选择和有关选型的局部论文核实，没有开展整篇论文分析。
