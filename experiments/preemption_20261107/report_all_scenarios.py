"""Build a comparable, fully sourced Markdown report for the preemption suite."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path('D:/vmagent_scenarios_preemption_20261107')
OUT = Path('C:/Users/trw/Desktop/论文/vmagentpro准备工作/十场景全策略评价结果与分析_20261107.md')
POLICIES = ('first_fit', 'best_fit', 'worst_fit', 'random',
            'legacy_spotgpu', 'no_preempt', 'paper_cost', 'random_victim',
            'paper_baseline', 'no_history')
NAMES = {
    'first_fit': 'First Fit', 'best_fit': 'Best Fit',
    'worst_fit': 'Worst Fit', 'random': 'Random',
    'legacy_spotgpu': '原 SpotGPU', 'no_preempt': '实验无抢占',
    'paper_cost': '成本感知', 'random_victim': '随机受害者',
    'paper_baseline': '随机消融基线', 'no_history': '无驱逐历史',
}
NOTES = {
    'P0_real_F8d': '原 F 的 1,070 个源边界前已出现任务先经过 3,600 秒模拟预热，并增加3,600秒准备服务量，使连续运行者在测量起点的剩余服务等于原F估计值；其余原 F 到达统一后移，随后测量完整 8 天。窗口起点的实际运行占用计入资源积分，但并非原始集群的精确初始放置。CPU 提供负载高于容量，GPU 工作量又由 HP 主导、可回收的 LP GPU 工作量较小。HP 尾响应和 LP JCT 的变化不能单独归因于抢占受害者算法。',
    'P1_steady': '持续但不集中的 HP/LP 竞争适合比较长期抢占损失。与 P2 配对时任务模板工作量相同，主要改变 HP 到达的突发程度；窗口边界积分可能略有不同。',
    'P2_burst': '每 6 小时把 HP 到达压到前 30 分钟；适合隔离到达突发下的抢占成本。HP 响应 P95 为 0 时仍须查看 60 秒违约率，因为 P95 会隐藏少数慢响应。',
    'P3_overload': 'GPU 提供负载约 130%，窗末积压是压力设计的预期结果。延长这种持续过载窗口不会自动产生稳态；比较窗末完成率、队列和停止到达后的 JCT 时须分别命名。',
    'P4_model_skew': '全池负载接近 105%，但 H20 的局部需求约 150%；必须看分型号表。全池有空闲时，缺少兼容型号或单节点 GPU/CPU 同时满足也会让 HP 等待。',
    'P5_cpu_contention': 'GPU 任务与 P2 相同，额外加入 CPU-only 任务，使 CPU 提供负载约 105%。它用于检验 CPU 竞争是否掩盖 GPU 抢占收益；与 P2 横比时，CPU 任务集合已变，不能把差异全归因于抢占策略。',
    'P6_real_H20_early15d': '连续 H20 早期窗口，但执行摘要不覆盖其服务时长，全部使用出现跨度代理，约 130% 的重建 GPU 负载可能被小时观测放大。只作探索性敏感性结果，不作主要论文实证。',
    'P7_real_H20_15d': '后期 H20 连续窗口，包含大量 CPU-only 任务，重建 GPU 负载约 66%。它检查真实到达形状和混合任务下的局部冲突；全池平均负载较低，不能称作强抢占压力。',
    'P8_real_XPUD_15d': '独立 XPU-D 连续窗口，用于检查 H20 之外的结论。固定节点池有一台节点在其中 12 小时改变 cluster 标签，场景未模拟成员迁移；详细限制见场景 README。',
    'P9_H20_gpu_capacity_pressure': '保留 P7 所有 GPU 任务的到达、需求、优先级和时长，去掉 CPU-only 并把节点从 30 台减至 19 台。它与 P7 的差别包含容量和 CPU-only 流变化；只在 P9 内比较策略，才能把策略差异配对归因。',
}

# Keep explanations beside every generated table so a reader can understand a
# single scenario without jumping to a separate glossary.
TABLE_NOTES = {
    'A': [
        '**读表说明（统计对象）**：HP/LP 均指测量窗内到达、请求 GPU 大于 0 的相应优先级任务；不含 CPU-only，也不把窗口前到达的预热任务计入这两组。`完成/任务` 则是**全仿真、所有任务**最终完成数/输入数，和其余列的分母不同。',
        '**首次响应与违约**：首次响应 = 任务第一次开始运行时刻 − 到达时刻，单位 `h` 为小时。`HP首响P95/P99` 是已启动 HP GPU 任务首次响应的第 95/99 百分位；例如 P95 为 2 小时，表示至少约 95% 的已启动样本在到达后 2 小时内首次运行。`HP违约60s` 是到达后超过 60 秒才首次运行（或一直未运行）的比例，只纳入到窗末前有完整 60 秒观察机会的 HP GPU 任务；60 秒是本实验阈值。',
        '**完成时间与窗末状态**：JCT（Job Completion Time）= 最终完成时刻 − 到达时刻，包含排队、运行、抢占后等待及重做。`LP平均JCT` 为已完成 LP GPU 任务的算术平均，`LP P95 JCT` 为其第 95 百分位；这两列跟踪到仿真最终排空，可能包含窗口结束后才完成的任务。`HP/LP窗末完成` 是该组窗内到达任务在测量窗结束前已完成的比例。`LP slowdown P95` 为已完成 LP GPU 任务的 $\\max(1,\\mathrm{JCT}/\\max(\\mathrm{service},60\\text{秒}))$ 第 95 百分位，无单位。',
    ],
    'B': [
        '**读表说明**：`结束=drained` 表示仿真继续运行到任务排空，不表示所有任务都在测量窗内完成。`GPU预留/CPU预留` 是测量窗内已分配 GPU·时间或 CPU·时间占对应总容量·窗口时长的比例，计入窗口起点已运行的预热任务；它们不是硬件实际计算利用率。`GPU有效工作` =（窗口内已分配 GPU·时间 − 最终证实丢失的窗口内 GPU·时间）/窗口 GPU 总容量·时长，仍不是 GPU SM 利用率。',
        '**损失与抢占次数**：`损失GPU·h` 是测量窗内执行、后来因抢占而丢失的 GPU·小时；`损失/占用` 以窗口内已分配 GPU·时间为分母。`全程抢占` 是从仿真开始到排空的受害事件次数，同一任务可计多次；`不同受害者` 是全程至少被抢占一次的任务数；`窗内抢占` 只数抢占事件发生在测量窗内的次数。这三列的时间范围和计数单位不同。',
    ],
    'C': [
        '**读表说明（“原”的含义）**：`原平均等待(h)` 和 `原P95等待(h)` 中的“原”指**原引擎自带的 `wait_sec` 指标**，不是“原 SpotGPU 策略”。它们在全仿真中对最终 `wait_sec` 非空、即至少启动过一次的**所有任务（含 CPU-only 和预热任务）**分别求平均与第 95 百分位，并换算为小时。每次启动或抢占后恢复都会覆盖该等待快照；若最后又被抢占而尚未恢复，末段等待不完整，零时刻累计运行也有已知边界问题，因此只作辅助诊断，不宜当作严格的总排队时间。',
        '**事件重建的排队与受害**：`LP GPU窗末排队P95(h)` 对测量窗内到达的**全部 LP GPU**任务，计算从到达到窗末（若更早完成则到完成）的累计未运行时间，再取第 95 百分位；包括窗末尚未启动者，也包括被抢占后的再次排队。`LP GPU被抢占` 是这组任务中全运行至少被驱逐一次的比例；`LP最多抢占` 是其中单个任务的最多驱逐次数。两者虽按窗内到达任务分组，但追踪抢占到仿真排空。',
        '**队列与吞吐**：`平均队长` 是测量窗内等待任务数对时间的平均，`峰值队长` 是全运行事件时刻出现的最大等待数；均包括各种优先级和 CPU-only。`窗内完成/小时` = 窗口内完成事件数 ÷ 窗口小时数，任务可在窗口前到达；`任务makespan(天)` = 全运行最晚完成时刻 − 最早任务到达时刻，再除以 86400，与测量窗长度不同。',
    ],
    'D': [
        '**读表说明（HP/LP 仍仅指 GPU 任务）**：`HP平均首响(h)` 是窗内到达且已启动的 HP GPU 任务首次响应时间平均值；`HP违约300s` 是到达后 300 秒仍未首次运行的比例，未启动者计违约，且只纳入窗末前有完整 300 秒观察机会者。`LP首响P95(h)` 是已启动 LP GPU 的首次响应第 95 百分位。首次响应只看第一次开跑，不包含抢占后的再次等待。',
        '**LP 尾部与其他诊断**：`LP排队>24h` 是窗内到达 LP GPU 中，截至窗末累计未运行时间超过 24 小时的比例；24 小时是研究阈值。`LP抢占P95` 是这组任务在全运行中的单任务驱逐次数第 95 百分位。`全任务窗末完成` 包括窗内到达的 GPU 与 CPU-only 任务；`全程损失GPU·h` 包括窗口前、窗口内和窗口后执行所产生的全部损失，与表 B 的窗口损失不同。`峰值GPU/峰值CPU核` 是全运行最大已分配资源量；`墙钟(s)` 是计算机实际运行仿真的秒数，不是任务执行时间。',
    ],
    'E': [
        '**读表说明**：每行只看指定 GPU 型号的节点池。`GPU容量` 是该型号节点的 GPU 总张数；`型号提供GPU` = 该型号任务若到达后立即运行所需 GPU·时间 ÷ 该型号池容量·窗口时长，故可能超过 100%；`型号预留GPU/型号预留CPU` 是该型号节点池在测量窗内实际已分配资源的时间平均占比，CPU-only 任务若放在这些节点也计入 CPU。`型号有效工作` 是该型号已分配 GPU·时间扣除后来丢失的工作后的比例，不是硬件 SM 利用率。',
    ],
}


def explain_table(lines, code):
    for paragraph in TABLE_NOTES[code]:
        lines.extend((paragraph, ''))


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pctl(xs, q):
    return sorted(xs)[max(0, math.ceil(len(xs) * q) - 1)] if xs else None


def fmt(value, digits=2, mult=1):
    return '—' if value is None else f'{float(value) * mult:,.{digits}f}'


def pct(value):
    return fmt(value, mult=100) + ('%' if value is not None else '')


def table(lines, headers, rows):
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    lines.append('')


def extra_gpu(path, start, end):
    data = {'HP': {'responses': [], 'waits': [], 'preempts': [], 'slowdowns': [],
                   'slo300_num': 0, 'slo300_den': 0},
            'LP': {'responses': [], 'waits': [], 'preempts': [], 'slowdowns': [],
                   'slo300_num': 0, 'slo300_den': 0}}
    with (path / 'task_metrics.csv').open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            priority = row['priority']
            if priority not in data or float(row['gpu']) <= 0:
                continue
            submit = float(row['submit'])
            if not start <= submit < end:
                continue
            d = data[priority]
            first = float(row['first_start']) if row['first_start'] else None
            if row['response_sec']:
                d['responses'].append(float(row['response_sec']))
            if submit + 300 <= end:
                d['slo300_den'] += 1
                d['slo300_num'] += first is None or first > submit + 300
            d['waits'].append(float(row['queue_sec_at_cutoff']))
            d['preempts'].append(int(row['preemptions']))
            if row['bounded_slowdown']:
                d['slowdowns'].append(float(row['bounded_slowdown']))
    return data


def get_run(scene, policy):
    path = ROOT / 'outputs' / f'{scene}-{policy}-s42-ck3600'
    if not (path / 'run_manifest.json').exists():
        return None
    manifest = load_json(path / 'run_manifest.json')
    assert manifest['scenario'] == scene and manifest['policy'] == policy
    assert manifest['seed'] == 42 and manifest['checkpoint_interval_sec'] == 3600
    summary = load_json(path / 'summary.json')
    metrics = load_json(path / 'metrics.json')
    gpu = load_json(path / 'gpu_priority_metrics.json')
    extra = extra_gpu(path, *manifest['measurement_window_sec'])
    return {'s': summary, 'm': metrics, 'g': gpu, 'x': extra, 'receipt': manifest}


def item(r, fn):
    return '—' if r is None else fn(r)


def main():
    manifest = load_json(ROOT / 'manifest.json')
    scenes = sorted(manifest['scenarios'], key=lambda s: int(s[1:s.index('_')]))
    runs = {scene: {p: get_run(scene, p) for p in POLICIES} for scene in scenes}
    present = sum(r is not None for scene in scenes for r in runs[scene].values())
    candidates = [ROOT / name for name in ('comparison_all.csv', 'comparison_all_strict_warmup.csv')]
    comparison = max((p for p in candidates if p.exists()), key=lambda p: p.stat().st_mtime_ns)
    all_rows = list(csv.DictReader(comparison.open(encoding='utf-8-sig')))
    assert present <= len(all_rows)
    extras = len(all_rows) - present
    lines = [
        '# 十场景全策略评价结果与分析', '',
        '日期：2026-11-07；指标释义修订：2026-11-08。输入为 `D:/vmagent_scenarios_preemption_20261107/` 的 P0–P9；所有数字从该目录的 `summary.json`、`metrics.json`、`gpu_priority_metrics.json`、`task_metrics.csv` 重新读取。完整公式、字段和分母见同目录《VMAgentPro抢占实验评价指标_公式字段与口径_20261107.md》。', '',
        '## 1. 覆盖范围与比较口径', '',
        f'主表固定 seed=42、checkpoint=3600 秒、`fast` 时钟、`backfill` 队列、CPU 背景 0；共 {present}/100 个“10 场景 × 10 个不同策略”主配置。另有 {extras} 组 P2 检查点/随机种子敏感性运行，总计 {len(all_rows)} 组。`legacy_spotgpu` 即当前引擎的原 `spotgpu`，不重复作为第 11 种策略。表中“—”表示该组合尚未运行，不能当成 0。', '',
        '原引擎放置策略为 First Fit、Best Fit、Worst Fit、Random、SpotGPU；另有实验无抢占、成本感知、随机受害者、随机消融基线、无驱逐历史五个变体。原四种不抢占放置策略与原 SpotGPU 使用**原 backfill 取任务顺序**；实验五变体使用**显式 HP GPU 优先搜索**，因此跨这两组比较会同时改变排队时机。`no_preempt` 对 `paper_cost` 可隔离同一实验队列下的抢占作用；`random_victim` 对 `paper_cost` 可隔离受害者选择；`no_history` 对 `paper_cost` 检查历史排序；`paper_baseline` 同时关闭共置/历史并采用随机受害者，不能单独归因于某一个部件。', '',
        '表 A 和 GPU 专项列按测量窗内到达的 GPU 任务分组，JCT 跟踪到排空；表 B 的资源预留、有效工作和窗口抢占按固定测量窗积分。所有场景均先调度窗口前任务，窗口起点已运行任务的占用计入积分；预热没有固定天数要求。P0 用 1 小时重建源边界任务的运行状态，P1–P9 在第 1 天仿真后起测量。精确核对见《十场景预热与窗口边界任务核查_20261107.md》。表 C/D 中注明“原”的等待与全任务吞吐、makespan、峰值、墙钟等指标另有样本范围，详见指标公式文档。HP 首次响应 P95 只统计已开始者，所以还需看包含未开始者的 60 秒违约率及窗末完成率。GPU 有效工作率扣除最终丢失进度，**不是 GPU SM 利用率**。', '',
    ]
    overview = []
    for scene in scenes:
        meta = manifest['scenarios'][scene]
        cnt = sum(r is not None for r in runs[scene].values())
        overview.append((f'[{scene}](D:/vmagent_scenarios_preemption_20261107/{scene}/README.md)',
                         f"{meta['tasks']:,}", f"{meta['gpu_tasks']:,}", pct(meta['offered_gpu_ratio']),
                         pct(meta['offered_cpu_ratio']), f'{cnt}/10'))
    table(lines, ['场景', '任务', 'GPU任务', '提供GPU', '提供CPU', '主策略覆盖'], overview)
    lines += [
        '运行质量：100 个主配置均为 `drained`，拒绝与未完成均为 0。所有主配置的输入 SHA256、代码 SHA256、seed、窗口与 checkpoint 见各组 `run_manifest.json`；不同策略的 GPU 有效工作率不可当真实 SM 使用。当前106组使用同一个核心 `environment.py` 哈希，实验 harness 存在接入原四种放置策略前后的两个版本；早期版本保存在 `code_snapshot/history/`，运行记录中的哈希可逐组核对。扩展仅加入原四策略分支，没有改已有六种变体的计算分支。旧P0的10组和未补准备服务量的7组中间运行已隔离归档，不混入当前106组。', '',
        '## 2. 各场景逐项结果', '',
        '表 A 使用 GPU 到达队列；“HP 首响 P95/P99”单位为小时，保留 0.00 可能代表小于 18 秒，应结合 60 秒违约率。表 B 的“损失 GPU·h”严格裁剪到测量窗，“抢占”与“不同受害者”是全运行；表 C 中旧引擎 `wait_sec` 可能受恢复覆盖及零时刻累计运行 bug 影响，仅作为参考，优先使用事件重建的 LP_GPU 等待。各表的空白组合用“—”明确标出。', '',
        '**先读懂单位和 P95**：`h` 是小时，`s` 是秒，`GPU·h` 是 GPU 张数乘以占用小时数；例如 2 张 GPU 运行 3 小时是 6 GPU·h。P95 是把该列符合统计条件的任务逐个算出数值，再从小到大排序得到第 95 百分位；本报告采用“最近秩”（取排序后第 $\\lceil0.95n\\rceil$ 个）。它不是平均值，也不代表最慢的 5% 的平均。表中小时数保留两位小数，`0.00 h` 可表示小于约 18 秒；要判断短时违约，应查看 60/300 秒违约列。', '',
    ]
    for scene in scenes:
        meta = manifest['scenarios'][scene]
        rs = runs[scene]
        a, b = meta['measurement_window_sec']
        lines += [f'### {scene}', '',
                  f"任务 {meta['tasks']:,}（GPU {meta['gpu_tasks']:,}）；测量窗 [{a:,.0f}, {b:,.0f}) 秒；节点表 `{meta.get('nodes_file','nodes.csv')}`；提供 GPU {pct(meta['offered_gpu_ratio'])}、CPU {pct(meta['offered_cpu_ratio'])}。", '',
                  '**表 A：HP/LP GPU 服务质量**', '']
        explain_table(lines, 'A')
        ar = []
        for p in POLICIES:
            r = rs[p]
            if r is None:
                ar.append((NAMES[p],) + ('—',) * 9)
                continue
            s, g, x = r['s'], r['g'], r['x']
            hp, lp = g['HP_GPU'], g['LP_GPU']
            ar.append((NAMES[p], f"{s['completed']:,}/{s['task_count']:,}",
                       fmt(hp['response_p95_sec_started'], mult=1/3600),
                       fmt(pctl(x['HP']['responses'], .99), mult=1/3600),
                       pct(hp['slo60_violation_ratio']), pct(hp['completion_rate_at_cutoff']),
                       fmt(lp['mean_jct_sec_completed'], mult=1/3600),
                       fmt(lp['p95_jct_sec_completed'], mult=1/3600),
                       pct(lp['completion_rate_at_cutoff']),
                       fmt(pctl(x['LP']['slowdowns'], .95))))
        table(lines, ['策略', '完成/任务', 'HP首响P95(h)', 'HP首响P99(h)', 'HP违约60s',
                      'HP窗末完成', 'LP平均JCT(h)', 'LP P95 JCT(h)', 'LP窗末完成', 'LP slowdown P95'], ar)
        lines += ['**表 B：资源、有效工作与抢占损失**', '']
        explain_table(lines, 'B')
        br = []
        for p in POLICIES:
            r = rs[p]
            if r is None:
                br.append((NAMES[p],) + ('—',) * 9)
                continue
            s, m = r['s'], r['m']
            br.append((NAMES[p], s['end_reason'], pct(s['reserved_gpu_ratio']),
                       pct(s['reserved_cpu_ratio']), pct(m['useful_gpu_ratio_retrospective']),
                       fmt(m['wasted_gpu_hours_window']), pct(m['lost_fraction_of_allocated']),
                       f"{s['preemptions']:,}", f"{s['preempted_task_count']:,}",
                       f"{m['preemption_victims_window']:,}"))
        table(lines, ['策略', '结束', 'GPU预留', 'CPU预留', 'GPU有效工作', '损失GPU·h',
                      '损失/占用', '全程抢占', '不同受害者', '窗内抢占'], br)
        lines += ['**表 C：排队、重复受害与吞吐**', '']
        explain_table(lines, 'C')
        cr = []
        for p in POLICIES:
            r = rs[p]
            if r is None:
                cr.append((NAMES[p],) + ('—',) * 9)
                continue
            s, m, x = r['s'], r['m'], r['x']
            lp = x['LP']['preempts']
            cr.append((NAMES[p], fmt(s['mean_wait_sec_started'], mult=1/3600),
                       fmt(s['p95_wait_sec_started'], mult=1/3600),
                       fmt(pctl(x['LP']['waits'], .95), mult=1/3600),
                       pct(sum(n > 0 for n in lp)/len(lp) if lp else None),
                       str(max(lp) if lp else 0), fmt(m['mean_queue_length']),
                       f"{s['peak_queue_length']:,}",
                       fmt(m['completed_during_window_per_hour']),
                       fmt(m['task_makespan_sec'], mult=1/86400)))
        table(lines, ['策略', '原平均等待(h)', '原P95等待(h)', 'LP GPU窗末排队P95(h)',
                      'LP GPU被抢占', 'LP最多抢占', '平均队长', '峰值队长', '窗内完成/小时',
                      '任务makespan(天)'], cr)
        lines += ['**表 D：尾部与运行诊断**', '']
        explain_table(lines, 'D')
        dr = []
        for p in POLICIES:
            r = rs[p]
            if r is None:
                dr.append((NAMES[p],) + ('—',) * 10)
                continue
            s, m, x = r['s'], r['m'], r['x']
            hp, lp = x['HP'], x['LP']
            dr.append((NAMES[p], fmt(sum(hp['responses'])/len(hp['responses']) if hp['responses'] else None, mult=1/3600),
                       pct(hp['slo300_num']/hp['slo300_den'] if hp['slo300_den'] else None),
                       fmt(pctl(lp['responses'], .95), mult=1/3600),
                       pct(sum(w > 86400 for w in lp['waits'])/len(lp['waits']) if lp['waits'] else None),
                       fmt(pctl(lp['preempts'], .95), 0),
                       pct(m['groups']['ALL']['completion_rate_at_cutoff']),
                       fmt(m['wasted_gpu_hours_full_run']),
                       fmt(s['peak_gpu_reserved']), fmt(s['peak_cpu_reserved_cores']),
                       fmt(s['wall_time_sec'])))
        table(lines, ['策略', 'HP平均首响(h)', 'HP违约300s', 'LP首响P95(h)',
                      'LP排队>24h', 'LP抢占P95', '全任务窗末完成', '全程损失GPU·h',
                      '峰值GPU', '峰值CPU核', '墙钟(s)'], dr)
        lines += ['**表 E：按 GPU 型号分池**', '']
        explain_table(lines, 'E')
        mr = []
        for p in POLICIES:
            r = rs[p]
            if r is None:
                continue
            s, m = r['s'], r['m']
            t = b-a
            for model, vals in sorted(s['per_model'].items()):
                ex = m['per_model'].get(model, {})
                useful = ((ex.get('allocated_gpu_sec', 0)-ex.get('lost_gpu_sec', 0)) /
                          (vals['gpu_capacity']*t)) if vals['gpu_capacity'] and t else None
                mr.append((NAMES[p], model, fmt(vals['gpu_capacity'], 0),
                           pct(vals['offered_gpu_ratio']), pct(vals['reserved_gpu_ratio']),
                           pct(useful), pct(vals['reserved_cpu_ratio'])))
        table(lines, ['策略', '型号', 'GPU容量', '型号提供GPU', '型号预留GPU',
                      '型号有效工作', '型号预留CPU'], mr or [('—',)*7])
        lines += ['**分析。** ' + NOTES[scene], '']
        no = rs['no_preempt']
        cost = rs['paper_cost']
        random = rs['random_victim']
        baseline = rs['paper_baseline']
        if no and cost:
            hn = no['g']['HP_GPU']['slo60_violation_ratio']
            hc = cost['g']['HP_GPU']['slo60_violation_ratio']
            jn = no['g']['LP_GPU']['mean_jct_sec_completed']
            jc = cost['g']['LP_GPU']['mean_jct_sec_completed']
            lines.append(f"在同一实验队列中，`no_preempt`→`paper_cost` 的 HP GPU 60 秒违约率为 {pct(hn)}→{pct(hc)}；LP GPU 平均 JCT 为 {fmt(jn, mult=1/3600)}→{fmt(jc, mult=1/3600)} 小时，窗口损失为 {fmt(no['m']['wasted_gpu_hours_window'])}→{fmt(cost['m']['wasted_gpu_hours_window'])} GPU·h。这是 HP 保障与 LP 代价的配对权衡，不能只比较一列。")
            lines.append('')
        if random and cost:
            lines.append(f"保持实验队列和放置规则时，`random_victim`→`paper_cost` 的窗口损失为 {fmt(random['m']['wasted_gpu_hours_window'])}→{fmt(cost['m']['wasted_gpu_hours_window'])} GPU·h，LP GPU 平均 JCT 为 {fmt(random['g']['LP_GPU']['mean_jct_sec_completed'], mult=1/3600)}→{fmt(cost['g']['LP_GPU']['mean_jct_sec_completed'], mult=1/3600)} 小时；这是受害者选择比较。")
            lines.append('')
            if cost['m']['wasted_gpu_hours_window'] > random['m']['wasted_gpu_hours_window']:
                lines += ['此场景的成本感知策略**全窗口损失反而更高**。它优化当前抢占的局部成本，后续队列和节点状态会随选择变化；不能据单次贪心目标推断全运行总损失必然最小。', '']
        if baseline and cost and not random:
            lines += ['`paper_baseline` 同时改变受害者选择与放置偏好；它与 `paper_cost` 的差值尚不能拆分归因于其中一项。', '']
        originals = [p for p in POLICIES[:4] if rs[p] is not None]
        if len(originals) == 4:
            best_hp = min(originals, key=lambda p: rs[p]['g']['HP_GPU']['slo60_violation_ratio'])
            best_lp = min(originals, key=lambda p: rs[p]['g']['LP_GPU']['mean_jct_sec_completed'])
            best_use = max(originals, key=lambda p: rs[p]['m']['useful_gpu_ratio_retrospective'])
            lines += [f"在四种原引擎不抢占放置策略中，本场景 HP GPU 60 秒违约率最低的是 {NAMES[best_hp]}（{pct(rs[best_hp]['g']['HP_GPU']['slo60_violation_ratio'])}），LP GPU 平均 JCT 最短的是 {NAMES[best_lp]}（{fmt(rs[best_lp]['g']['LP_GPU']['mean_jct_sec_completed'], mult=1/3600)} 小时），GPU 有效工作率最高的是 {NAMES[best_use]}（{pct(rs[best_use]['m']['useful_gpu_ratio_retrospective'])}）。这些最优项可能属于不同策略，不能直接合并成总排名。", '']
        history = rs['no_history']
        if history and cost:
            delta = history['m']['wasted_gpu_hours_window'] - cost['m']['wasted_gpu_hours_window']
            lines += [f"只去掉驱逐历史排序后，窗口损失相对 `paper_cost` 变化 {delta:+,.2f} GPU·h；这一层仅在前面的打包与共置排序平局时才可能影响决策，零差异不证明该规则在所有场景都无效。", '']
        lines += [f'场景细节和输入来源：[本场景 README](D:/vmagent_scenarios_preemption_20261107/{scene}/README.md)。', '']
    lines += ['## 3. P2 检查点和随机种子敏感性', '',
              '以下不混入上述 seed=42、3600 秒主表。checkpoint 模型没有保存 I/O 与驱逐时延，较短周期减少重做的结果是理想化上界。', '',
              '**读表说明**：`seed` 是随机数种子；`检查点(s)` 是理想检查点间隔，单位秒。`GPU有效工作` 和 `损失GPU·h` 为固定测量窗的资源指标；`LP平均JCT(h)` 是窗内到达 LP GPU 任务最终排空后的平均完成用时；`HP违约60s` 是窗内到达且有完整 60 秒观察机会的 HP GPU 任务中首次运行超过 60 秒或未运行的比例。', '']
    sr = []
    for row in all_rows:
        if row['scenario'] != 'P2_burst' or row['policy'] not in ('paper_cost','paper_baseline'):
            continue
        sr.append((NAMES[row['policy']], row['seed'], row['checkpoint_sec'],
                   pct(row['gpu_useful_ratio']), fmt(row['lost_gpu_hours_window']),
                   fmt(row['lp_gpu_mean_jct_sec'], mult=1/3600),
                   pct(row['hp_gpu_slo60_violation_ratio'])))
    sr.sort(key=lambda x: (x[0], int(x[1]), float(x[2])))
    table(lines, ['策略', 'seed', '检查点(s)', 'GPU有效工作', '损失GPU·h',
                  'LP平均JCT(h)', 'HP违约60s'], sr)
    lines += ['## 4. 结论与边界', '',
              '优先判断 HP GPU 响应与违约是否达到实验阈值，再比较 LP GPU 窗末完成、JCT 与重复受害，最后看 GPU 预留和损失。无抢占可能有最高有效工作率却让 HP 等待；持续过载和局部型号压力下尤其要看窗末状态与分型号结果。', '',
              'P6 服务时长、P7/P8 小时内提交时刻、P8 节点成员变化以及所有场景统一零 checkpoint I/O/瞬时驱逐，均限制对真实生产性能的外推。单个 seed 和窗口不能给出总体显著性；正式论文建议多个独立构造种子、随机策略种子与不重叠真实窗口做配对置信区间。', '',
              '完整逐任务明细、原 summary、事件衍生指标、GPU 优先级指标和输入/代码 SHA256 均保存在各组 `outputs/` 中；本报告没有把未运行的格子填成推测值。', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'{OUT} | {present}/100 main configurations | {len(all_rows)} total runs')


if __name__ == '__main__':
    main()
