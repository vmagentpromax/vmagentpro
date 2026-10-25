# 按小时请求指标重建自然场景

沿用原 20 台 H20（160 GPU / 3,840 CPU 核）。遵照用户明确选择：只用自然高低窗口，不存在就报告缺失，不构造替代。

## 结果

完整读取 day0～day184 的 **4,440 个小时分区**，穷举连续 6～24 小时共 **84,094 个窗口**。窗口分类完全不使用执行时长。

高阈值为平均 GPU 请求饱和度 ≥80%、CPU ≥60%；低阈值为两者均 ≤35%；GPU 正请求的唯一 Pod 数限制为 50～150。最低窗口平均 GPU 请求饱和度为 **85.4167%**。

![固定20台H20的6小时滚动平均GPU/CPU请求饱和度](D:/一般文件/VMAgent/docs/server_selection/request_load_scenes/request_pressure.png)

图：原始Pod小时表，day0～184；分母固定为160 GPU / 3,840 CPU核。实线为6小时滚动平均，横线为本轮阈值；完整6～24小时逐窗核验另存CSV。

**低负载：缺失。** 满足负载阈值的窗口 0 个，同时满足任务数限制的候选 0 个。没有生成替代任务文件。

**高负载：day144/00～day144/23，连续 24 小时。** 源平均 GPU/CPU 请求饱和度分别为 **94.9969% / 101.4800%**；源 GPU 资源时间比例为 94.7482%，不是 SM 利用率。

源窗口有 100 个 GPU 正请求唯一 Pod，排除请求变化、型号或单节点容量不合格的 0 个，保留 100 个稳定任务。匹配时长后交付 **100 个任务**，未匹配 0 个，84 个存在多个兼容执行段；全部标记 `duration_proxy=true`。所选窗口每一小时的 20 台节点配置均已独立核验。

分数 / 单卡 / 多卡任务占比为 26.00% / 53.00% / 21.00%。

主输入目标强度 ρ=0.85，统一 K=68,160。服务时间均值 / P95 / 最大值为 40.706106 / 89.998826 / 90.000000 秒。seed42 的期望 / 实际到达跨度为 32.328912 / 37.134082 秒，指数间隔均值为 0.326555 秒。

按保留任务的CPU工作量及期望到达跨度计算的CPU输入强度为 **62.5847%**，独立于源窗口的CPU请求饱和度。

共生成 360 个输入：30 个种子（42～71），4 个输入强度，3 个统一服务时间尺度（最长约60/90/120秒）。主入口是 [high/tasks.json](high/tasks.json)，CSV 含完整来源字段；全部序列见 [high/sequence_manifest.csv](high/sequence_manifest.csv)。

**该样本不构成高占用仿真基准。** 对这组固定任务及服务时间代理，任何非负到达序列和调度方法的makespan均不小于最长服务时间，因此全程AGU上界为 `W/(160×max(service))` = **30.5329%**。即使将任务同时提交，也不能达到80%的全程GPU占用；统一改变K不改变这个上界。ρ=0.85只能作为到达强度参数，不能把该输入命名为已经验证的85%占用场景。

## 解释边界

**本轮完成了自然高请求窗口抽取、可复算任务输入及压力验证；严格的自然高低占用对照并未成立。** 自然低负载类别缺失，高请求样本的有限运行占用又未达到高负载目标。

自然分类来自源小时请求量。任务和 GPU/CPU 配对是真实的，秒级到达是新生成的指数序列，服务时间是执行跨度代理的统一缩放，不能称为精确历史回放。请求强度可以超过100%，不证明小时内所有 Pod 同时运行。

`W/(ρ×160)` 设定的是输入强度，不保证有限任务、空集群起步的实测占用率等于ρ。没有预热或背景负载，不声称稳态。源 CPU 指标包括 CPU-only Pod，而输入只含 GPU 正请求 Pod；源 CPU 压力不等于仿真 CPU 压力。

缺少自然低负载窗口时，无法检查高低窗口资源形状是否可比，也不能完成或声称高低负载之间的算法比较。本结论仅适用于固定的这20台节点及当前阈值，不外推为整个阿里集群没有低负载。

## 窗口长度检查

- 6小时：高负载且任务数合格 1,591 个，低负载且任务数合格 0 个；最低平均GPU请求饱和度 85.4167%。
- 12小时：高负载且任务数合格 473 个，低负载且任务数合格 0 个；最低平均GPU请求饱和度 93.7917%。
- 24小时：高负载且任务数合格 44 个，低负载且任务数合格 0 个；最低平均GPU请求饱和度 94.9188%。

全部19种窗口长度的逐窗结果保留在 `all_window_checks.csv`；以上是来源窗口可行性检查，不冒充跨窗口长度的算法稳健性结论。

## 仿真实测

FirstFit / BestFit / WorstFit 共 **1,080 轮**；各条件共用30个种子。全部排空，无拒绝、无未完成，资源完全归还，GPU资源时间积分逐轮核验通过。

主条件（最长约90秒；高负载ρ=0.85）的30种子均值如下。AGU为整个有限运行期间的GPU预留率，不是SM利用率；AFR为等待期间总GPU和CPU足够但无法在单节点共同放置的时间占比。

- high/first_fit：平均JCT 40.706106秒，等待 0.000000秒，makespan 119.346345秒，AGU 23.0584%，AFR 0.0000%。
- high/best_fit：平均JCT 40.706106秒，等待 0.000000秒，makespan 119.346345秒，AGU 23.0584%，AFR 0.0000%。
- high/worst_fit：平均JCT 41.744725秒，等待 1.038619秒，makespan 122.721903秒，AGU 22.4547%，AFR 29.0742%。

每个指标的双侧Student-t 95%置信区间见 [validation/metrics_mean_ci95.csv](validation/metrics_mean_ci95.csv)，同种子算法配对差值及区间见 [validation/paired_differences_ci95.csv](validation/paired_differences_ci95.csv)。所有种子的原始统计与瞬态截点后的次级指标也保存在 `validation/`。seed42 的详细事件与资源日志见仓库 `outputs/gpu/request_scenes/`。

## 复现

在仓库根目录执行（Python需numpy/pandas/pyarrow/scipy）：

```powershell
python docs/server_selection/rebuild_request_scenes.py --workers 4
python docs/server_selection/construct_request_scenes.py
python docs/server_selection/run_request_scenes.py --sensitivity
python docs/server_selection/write_request_scene_report.py
```

扫描支持逐日缓存续跑。原始数据不修改。完整方法与指标定义见 [METHOD.md](METHOD.md)；`scan_audit.json`、`window_selection_audit.json`、`selected_windows.json`、`construction_audit.json` 串联保留来源证据。旧 `two_gpu_scenes/relative_low` 与 `relative_high` 仅作历史留存，不再作为本流程的合格高低场景。
