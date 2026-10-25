"""Render the completed construction audits as a concise Chinese README."""
import json
from pathlib import Path
import shutil

import pandas as pd

from rebuild_request_scenes import OUT

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def window_name(start, end):
    return f"day{start//24}/{start%24:02d}～day{(end-1)//24}/{(end-1)%24:02d}"


def plot_load():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    hours = pd.read_csv(OUT / "hourly_request_load.csv").set_index("absolute_hour")
    # Reindex to preserve missing-hour gaps in the rolling means.
    hours = hours.reindex(range(int(hours.index.min()),int(hours.index.max())+1))
    fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True,layout="constrained")
    for ax, key, name, high in zip(axes,("rho_gpu","rho_cpu"),("GPU","CPU"),(.8,.6)):
        ax.plot(hours.index/24,hours[key].rolling(6,min_periods=6).mean(),linewidth=.8,
                color="#28587b",label="6-hour rolling mean of hourly requests / fixed capacity")
        ax.axhline(.35,color="#a14135",linestyle="--",linewidth=1,label="Natural low threshold (35%)")
        ax.axhline(high,color="#666666",linestyle=":",linewidth=1,label=f"Natural high threshold ({high:.0%})")
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"{name} request saturation (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(loc="upper right",fontsize=7,frameon=False)
    axes[-1].set_xlabel("Relative trace day (0-184)")
    fig.suptitle("Original 20 H20 servers: natural-window request pressure",fontsize=13)
    fig.savefig(OUT / "request_pressure.png",dpi=160)
    plt.close(fig)


def main():
    scan = read(OUT / "scan_audit.json")
    windows = read(OUT / "window_selection_audit.json")
    selected = read(OUT / "selected_windows.json")
    built = read(OUT / "construction_audit.json")
    checks = pd.read_csv(OUT / "all_window_checks.csv")
    plot_load()
    lines = ["# 按小时请求指标重建自然场景", "",
        "沿用原 20 台 H20（160 GPU / 3,840 CPU 核）。遵照用户明确选择：只用自然高低窗口，不存在就报告缺失，不构造替代。", "",
        "## 结果", "",
        f"完整读取 day{scan['day_min']}～day{scan['day_max']} 的 **{scan['hours_scanned']:,} 个小时分区**，穷举连续 6～24 小时共 **{windows['window_count']:,} 个窗口**。窗口分类完全不使用执行时长。", "",
        f"高阈值为平均 GPU 请求饱和度 ≥80%、CPU ≥60%；低阈值为两者均 ≤35%；GPU 正请求的唯一 Pod 数限制为 50～150。最低窗口平均 GPU 请求饱和度为 **{windows['minimum_window_mean_rho_gpu']:.4%}**。", "",
        f"![固定20台H20的6小时滚动平均GPU/CPU请求饱和度]({(OUT / 'request_pressure.png').as_posix()})", "",
        "图：原始Pod小时表，day0～184；分母固定为160 GPU / 3,840 CPU核。实线为6小时滚动平均，横线为本轮阈值；完整6～24小时逐窗核验另存CSV。", ""]
    for label, chinese in (("low","低负载"),("high","高负载")):
        info = selected["natural_windows"][label]
        if info["status"] == "missing":
            lines += [f"**{chinese}：缺失。** 满足负载阈值的窗口 {info['threshold_candidates']:,} 个，同时满足任务数限制的候选 {info['count_qualified_candidates']:,} 个。没有生成替代任务文件。", ""]
        else:
            b = built["natural_scenes"][label]
            main_input = read(OUT / label / "tasks.json")
            match = read(OUT / label / "duration_match_audit.json")
            lines += [f"**{chinese}：{window_name(info['start_hour'],info['end_hour_exclusive'])}，连续 {info['hours']} 小时。** 源平均 GPU/CPU 请求饱和度分别为 **{info['mean_rho_gpu']:.4%} / {info['mean_rho_cpu']:.4%}**；源 GPU 资源时间比例为 {info['mean_used_gpu_ratio']:.4%}，不是 SM 利用率。", "",
                f"源窗口有 {info['unique_gpu_pods']} 个 GPU 正请求唯一 Pod，排除请求变化、型号或单节点容量不合格的 {info['excluded_pods']} 个，保留 {info['retained_stable_gpu_pods']} 个稳定任务。匹配时长后交付 **{b['task_count']} 个任务**，未匹配 {match['unmatched_templates']} 个，{match['multiple_compatible_span_pods']} 个存在多个兼容执行段；全部标记 `duration_proxy=true`。所选窗口每一小时的 20 台节点配置均已独立核验。", "",
                f"分数 / 单卡 / 多卡任务占比为 {b['shape_fractions']['fractional']:.2%} / {b['shape_fractions']['single']:.2%} / {b['shape_fractions']['multi']:.2%}。", "",
                f"主输入目标强度 ρ={b['main_rho']}，统一 K={b['time_scale_factor']:,}。服务时间均值 / P95 / 最大值为 {main_input['service_mean_sec']:.6f} / {main_input['service_p95_sec']:.6f} / {main_input['service_max_sec']:.6f} 秒。seed42 的期望 / 实际到达跨度为 {main_input['expected_arrival_span_sec']:.6f} / {main_input['actual_arrival_span_sec']:.6f} 秒，指数间隔均值为 {main_input['mean_interarrival_sec']:.6f} 秒。", "",
                f"按保留任务的CPU工作量及期望到达跨度计算的CPU输入强度为 **{main_input['cpu_work_over_expected_arrival_span_ratio']:.4%}**，独立于源窗口的CPU请求饱和度。", "",
                f"共生成 {b['sequence_count']} 个输入：30 个种子（42～71），4 个输入强度，3 个统一服务时间尺度（最长约60/90/120秒）。主入口是 [{label}/tasks.json]({label}/tasks.json)，CSV 含完整来源字段；全部序列见 [{label}/sequence_manifest.csv]({label}/sequence_manifest.csv)。", ""]
            upper_bound = main_input['gpu_work_gpu_sec']/(160*main_input['service_max_sec'])
            if label == "high" and upper_bound < .8:
                lines += [f"**该样本不构成高占用仿真基准。** 对这组固定任务及服务时间代理，任何非负到达序列和调度方法的makespan均不小于最长服务时间，因此全程AGU上界为 `W/(160×max(service))` = **{upper_bound:.4%}**。即使将任务同时提交，也不能达到80%的全程GPU占用；统一改变K不改变这个上界。ρ=0.85只能作为到达强度参数，不能把该输入命名为已经验证的85%占用场景。", ""]
    lines += ["## 解释边界", "",
        "**本轮完成了自然高请求窗口抽取、可复算任务输入及压力验证；严格的自然高低占用对照并未成立。** 自然低负载类别缺失，高请求样本的有限运行占用又未达到高负载目标。", "",
        "自然分类来自源小时请求量。任务和 GPU/CPU 配对是真实的，秒级到达是新生成的指数序列，服务时间是执行跨度代理的统一缩放，不能称为精确历史回放。请求强度可以超过100%，不证明小时内所有 Pod 同时运行。", "",
        "`W/(ρ×160)` 设定的是输入强度，不保证有限任务、空集群起步的实测占用率等于ρ。没有预热或背景负载，不声称稳态。源 CPU 指标包括 CPU-only Pod，而输入只含 GPU 正请求 Pod；源 CPU 压力不等于仿真 CPU 压力。", "",
        "缺少自然低负载窗口时，无法检查高低窗口资源形状是否可比，也不能完成或声称高低负载之间的算法比较。本结论仅适用于固定的这20台节点及当前阈值，不外推为整个阿里集群没有低负载。", "",
        "## 窗口长度检查", ""]
    for length in (6,12,24):
        subset = checks[checks.hours==length]
        lines.append(f"- {length}小时：高负载且任务数合格 {int(subset.high_candidate.sum()):,} 个，低负载且任务数合格 {int(subset.low_candidate.sum()):,} 个；最低平均GPU请求饱和度 {subset.mean_rho_gpu.min():.4%}。")
    lines += ["", "全部19种窗口长度的逐窗结果保留在 `all_window_checks.csv`；以上是来源窗口可行性检查，不冒充跨窗口长度的算法稳健性结论。", ""]
    result = ROOT / "outputs/gpu/request_scenes"
    if (result / "evaluation_audit.json").exists():
        audit = read(result / "evaluation_audit.json")
        validation = OUT / "validation"
        validation.mkdir(exist_ok=True)
        for name in ("per_seed_metrics.csv","metrics_mean_ci95.csv","paired_differences_ci95.csv","evaluation_audit.json"):
            shutil.copyfile(result / name, validation / name)
        means = pd.read_csv(result / "metrics_mean_ci95.csv")
        lines += ["## 仿真实测", "",
            f"FirstFit / BestFit / WorstFit 共 **{audit['runs']:,} 轮**；各条件共用30个种子。全部排空，无拒绝、无未完成，资源完全归还，GPU资源时间积分逐轮核验通过。", "",
            "主条件（最长约90秒；高负载ρ=0.85）的30种子均值如下。AGU为整个有限运行期间的GPU预留率，不是SM利用率；AFR为等待期间总GPU和CPU足够但无法在单节点共同放置的时间占比。", ""]
        for label in built.get("natural_scenes",{}):
            rho = built["natural_scenes"][label]["main_rho"]
            for policy in ("first_fit","best_fit","worst_fit"):
                sub = means[(means.scene==label)&(means.rho==rho)&(means.max_service_sec==90)&(means.policy==policy)].set_index("metric")
                lines.append(f"- {label}/{policy}：平均JCT {sub.loc['mean_jct_sec','mean']:.6f}秒，等待 {sub.loc['mean_wait_sec','mean']:.6f}秒，makespan {sub.loc['makespan_sec','mean']:.6f}秒，AGU {sub.loc['agu_reserved','mean']:.4%}，AFR {sub.loc['afr_joint','mean']:.4%}。")
        lines += ["", "每个指标的双侧Student-t 95%置信区间见 [validation/metrics_mean_ci95.csv](validation/metrics_mean_ci95.csv)，同种子算法配对差值及区间见 [validation/paired_differences_ci95.csv](validation/paired_differences_ci95.csv)。所有种子的原始统计与瞬态截点后的次级指标也保存在 `validation/`。seed42 的详细事件与资源日志见仓库 `outputs/gpu/request_scenes/`。", ""]
    lines += ["## 复现", "", "在仓库根目录执行（Python需numpy/pandas/pyarrow/scipy）：", "", "```powershell",
        "python docs/server_selection/rebuild_request_scenes.py --workers 4",
        "python docs/server_selection/construct_request_scenes.py",
        "python docs/server_selection/run_request_scenes.py --sensitivity",
        "python docs/server_selection/write_request_scene_report.py", "```", "",
        "扫描支持逐日缓存续跑。原始数据不修改。完整方法与指标定义见 [METHOD.md](METHOD.md)；`scan_audit.json`、`window_selection_audit.json`、`selected_windows.json`、`construction_audit.json` 串联保留来源证据。旧 `two_gpu_scenes/relative_low` 与 `relative_high` 仅作历史留存，不再作为本流程的合格高低场景。", ""]
    (OUT / "README.md").write_text("\n".join(lines),encoding="utf-8")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
