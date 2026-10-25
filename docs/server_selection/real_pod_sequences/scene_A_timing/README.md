# 场景 A：50 个真实 Pod 的仿真到达与执行时间

已生成 `scene_A_50_pods_timed.csv` 和供程序加载的 `scene_A_50_pods_timed.json`。保留原场景 A 的全部 50 个不同 Pod、H20 型号及 GPU/CPU 请求配对，没有替换任务、重复提交或修改资源需求。

## 执行时间来源与匹配限制

原始执行汇总路径：`E:\cluster-trace-gpu-v2026\data\asi_opensource_job_execution_summary\asi_opensource_job_execution_summary\part-000.parquet`。

50 个 Pod 在执行汇总中共匹配到 74 条记录。所有记录的 `pod_id`、`gpu_spec_public`、`gpu_request`、`server_id`、`job_type_public`、`priority_class` 与所选 Pod 行一致。其中 26 个 Pod 只匹配到一条，24 个 Pod 各有两条不同时长的记录。

对于多条记录的 Pod，本次**取较长的真实执行段**，不求和、不平均、不为了短时实验取最短段。每个选定时长都真实存在于原始表，并记录 `execution_summary_row_index_0based` 以供精确追溯。但是，公开表缺少相对起止时间和执行实例 ID，无法证明这条执行段就是 day=109/hour=00 对应的那次执行。因此 24 个 Pod 标为 `ambiguous_longest_span_proxy`；26 个单匹配 Pod 标为 `single_matching_span`。单匹配也不等于获得了精确历史起止时间。

这是一组**源自真实执行段的仿真服务时间代理**，不能宣称恢复了 50 个 Pod 的真实完整生命周期。原始候选记录保存在 `matched_original_execution_spans.csv`，每个 Pod 在最终任务输入中仍只出现一次。

维护者确认：duration_hours 是由观测样本推得的墙钟执行跨度；公开执行汇总没有相对起止时间或独立执行实例 ID。[一手说明](https://github.com/alibaba/clusterdata/issues/259#issuecomment-5494018728)

## 统一时间压缩

选定执行段中最长原始时长约 911.994444 小时。此前讨论的 360 倍是全局分布示例，套在这一批长任务上，最长服务时间仍约 9,119.94 秒，即 152 分钟，达不到首轮实验约 1～3 分钟的目的。

本次不截断长尾，采用统一倍率：

```text
time_scale_factor = ceil(max(source_duration_hours × 3600) / 90)
                  = 36480

service_sec = source_duration_hours × 3600 / 36480
```

所有任务共享同一倍率，执行时长的相对比例保留。最短约 0.000548 秒，中位数约 66.3114 秒，最长约 89.99945 秒。非常短的任务仍保留原值，没有人为拉长或设置最小时长；实际计时精度可能无法分辨亚毫秒任务。

## 到达时间与到达间隔

原始 trace 不提供这些 Pod 的精确提交时间。以下全部为新生成的仿真到达时间，**不属于恢复出的真实到达轨迹**：

- 随机种子 42，先固定随机排列这 50 个 Pod。
- 首个任务在 t=0 提交。
- 后续 49 个间隔独立服从均值为 0.3 秒的指数分布，参数到达率为 3.333333 个/秒。
- 提交时间为间隔的累计和，不根据任务自身执行时间排序或调整。
- 使用 NumPy `SeedSequence(42).spawn(2)` 分别生成任务顺序和间隔的独立随机流。

```text
interarrival_sec[0] = 0
interarrival_sec[i] ~ Exponential(mean = 0.3 seconds), i=1..49
submit_time_sec[i] = sum(interarrival_sec[0..i])
```

这次 49 个间隔的实际均值为 0.331315 秒，最后一个任务在 **16.234444 秒**提交。固定生成 50 个任务，所以结束到达的时刻是随机结果；不是声称固定 16.234444 秒的泊松窗口必然产生 50 个任务。

原到达时间未知，本次不能按原轨迹同比例缩放；这些到达是压缩后的仿真时间尺度上的新实验参数，不声称与原始流量等价。

## 实际达到的场景 A 压力

节点容量继续固定为 160 GPU 和 3,840 CPU 核，CPU 背景配额为 0。测量窗口为 **[0,120] 秒**。

按提交后立即开始的理想执行区间与窗口求交，重新积分 GPU 时间和 CPU 核时间，得到：

- GPU 提供负载约 **5.7416%**。
- CPU 提供负载约 **3.8292%**。
- 理想峰值 GPU 请求为 22.96，即容量的 **14.35%**。
- 理想峰值 CPU 请求为 489 核，即容量的 **12.7344%**。
- 无排队时最后一个任务约在 **105.740805 秒**结束。

这落在附件约定的“双低”区间，但**不是精确的 30%/30%**。前一步的 30%/30% 仅是 50 个请求总和除以容量；实际时长不相同、到达时间不相同，任务并不会始终全部重叠，因此该比例不能替代时间积分。此处没有通过添加背景、重复任务、拉长短任务或裁剪长任务凑百分比。

105.74 秒是理想立即运行的模拟完成时间，不是已测试的真实程序耗时。仿真器应使用：实际完成时间 = **实际开始时间 + service_sec**。CSV 中的 `ideal_finish_time_sec` 只用于输入压力审计，不能作为排队任务的实际释放时刻。

## 任务输入字段

- `arrival_order`：生成后的提交顺序。
- `pod_id`、`gpu_spec_public`、`gpu_request`、`cpu_request_cores`：保留的真实身份及资源配对。
- `submit_time_sec`：相对本次实验开始的合成到达时间。
- `interarrival_sec`：与上一任务的到达间隔，首行为 0。
- `service_sec`：统一压缩后的服务时间，从实际开始运行时计时。
- `source_duration_hours`：选定的原始执行段时长。
- `matching_execution_span_count`、`duration_match_status`：匹配数量及歧义标记。
- `execution_summary_row_index_0based`：选定执行段在原始 Parquet 中的零起始全局行号。
- `arrival_is_synthetic`：全部为 true，避免把生成时间当历史事实。
- 其余来源字段保留用于追溯。原请求清单行号改名为 `source_row_index`。

## 复算和核验

先运行上两级目录的 `match_scene_A_durations.py`，再运行 `build_scene_A_timing.py`。可在第二个脚本用参数调整最长服务时间目标、平均间隔、测量窗口和种子；改变测量窗口后必须重新计算提供负载。

已检查 50 个原始 Pod ID 及资源配对一致、最终 ID 唯一、执行时长与原始 Parquet 指定行一致、压缩倍率一致、间隔为正且累计和等于到达时间。详细指标与文件哈希保存在 `timing_audit.json`。没有运行或修改 VMAgent 调度器。
