"""Verify pre-window arrivals and boundary occupancy of the no-preempt runs."""
import csv
import json
from pathlib import Path

ROOT = Path('D:/vmagent_scenarios_preemption_20261107')
OUT = Path('C:/Users/trw/Desktop/论文/vmagentpro准备工作/十场景预热与窗口边界任务核查_20261107.md')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def audit(name, meta):
    start = float(meta['measurement_window_sec'][0])
    task_path = ROOT / name / 'tasks.csv'
    tasks = {r['pod_id']: r for r in rows(task_path)}
    run = ROOT / 'outputs' / f'{name}-no_preempt-s42-ck3600'
    receipt = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    assert receipt['task_sha256'] == meta['sha256']
    assert float(receipt['measurement_window_sec'][0]) == start
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    assert summary['preemptions'] == 0 and summary['end_reason'] == 'drained'
    pre = ideal = running = waiting = finished = gpu_running = 0
    gpu_share = cpu_cores = 0.0
    for r in rows(run / 'task_metrics.csv'):
        assert r['pod_id'] in tasks
        submit = float(r['submit'])
        if submit >= start:
            continue
        pre += 1
        if submit + float(r['service']) > start:
            ideal += 1
        finish = float(r['finish']) if r['finish'] else float('inf')
        first = float(r['first_start']) if r['first_start'] else float('inf')
        if finish <= start:
            finished += 1
        elif first < start:
            running += 1
            gpu = float(r['gpu'])
            gpu_running += gpu > 0
            gpu_share += gpu
            cpu_cores += float(tasks[r['pod_id']]['cpu_request_cores'])
        else:
            waiting += 1
    assert pre == running + waiting + finished
    assert running > 0
    return {'name': name, 'start': start, 'pre': pre, 'ideal': ideal,
            'running': running, 'gpu_running': gpu_running,
            'gpu_share': gpu_share, 'cpu_cores': cpu_cores,
            'waiting': waiting, 'finished': finished}


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    names = sorted(manifest['scenarios'], key=lambda s: int(s.split('_', 1)[0][1:]))
    facts = [audit(name, manifest['scenarios'][name]) for name in names]
    p0 = manifest['scenarios']['P0_real_F8d']
    assert p0['warmup_duration_sec'] == 3600
    assert facts[0]['pre'] == p0['source_boundary_presence_tasks'] == 1070
    assert facts[0]['ideal'] == 1070
    lines = [
        '# 十场景预热与窗口边界任务核查', '',
        '核查日期：2026-11-07。输入和输出：`D:/vmagent_scenarios_preemption_20261107/`；复现脚本：`C:/Users/trw/Desktop/论文/vmagentpro/experiments/preemption_20261107/audit_warmup.py`。', '',
        '## 1. 定义和判定', '',
        '预热没有固定天数。对测量窗起点 $t_0$，先在 $[0,t_0)$ 处理已到达任务，随后从 $t_0$ 开始计资源占用和服务质量。严格的仿真预热要求存在 $a_i<t_0$ 且在 $t_0$ 仍运行的任务，其运行段与测量窗的交集进入资源积分。窗口前到达但已完成或仍排队的任务不虚构为运行占用；仍排队者会影响后续排队。', '',
        '本表用 `no_preempt`、seed=42、checkpoint=3600 秒的逐任务结果核对起点状态。由于该策略不抢占，`submit<t0`、`first_start<t0<finish` 可准确判定当时正在运行。GPU 预留是这些任务的 `gpu_request` 之和，CPU 预留是相应输入行的 `cpu_request_cores` 之和。其他策略的起点运行集合可因抢占与放置而不同；所有策略均从仿真0秒开始执行，`harness.py` 把每段实际运行时间与测量窗求交后计入 GPU 资源积分。', '',
        '| 场景 | 测量起点(s) | 起点前到达 | 理想服务跨界 | 实际运行 | 其中GPU任务 | GPU预留 | CPU预留核 | 等待 | 已完成 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for f in facts:
        lines.append(f"| {f['name']} | {f['start']:,.0f} | {f['pre']:,} | {f['ideal']:,} | {f['running']:,} | {f['gpu_running']:,} | {f['gpu_share']:,.2f} | {f['cpu_cores']:,.2f} | {f['waiting']:,} | {f['finished']:,} |")
    lines += [
        '',
        '“理想服务跨界”只按输入的 `submit_time_sec + service_sec > t0` 判断，未考虑排队，**不是**实际运行数。实际运行数按仿真结果判断；因此两列可能不同。表中 `等待+实际运行+已完成=起点前到达`。', '',
        '## 2. P0 的修正与限制', '',
        '旧 P0 的测量窗从 0 秒开始，尽管有 1,070 个 `presence_remaining` 边界任务（其中 217 个 GPU 任务），它们在 0 秒才进入空节点池，无法构成窗口起点已运行占用。旧输入、旧 `manifest.json` 和旧 10 组结果已归档到 `D:/vmagent_scenarios_preemption_20261107/archive_pre_warmup_P0_20261107/`。首次预热但尚未补上准备服务量的7组中间运行隔离于 `archive_p0_warmup_unadjusted_service_20261107/`，均不计入最终比较。', '',
        '新 P0 将这 1,070 个边界任务放在模拟0秒到达，并为它们增加3,600秒准备服务量；原 F 中所有正时间到达任务统一后移3,600秒，保持彼此间到达差与完整8天顺序。若边界任务连续运行，则新测量起点的剩余服务恰好等于旧F的 `service_sec`；排队任务仍可能剩余更多，这是重建误差。先调度1小时，测量窗改为 `[3600,694800)` 秒；这样窗口起点实际运行的边界任务占用进入 GPU/CPU 负载。1小时来自原数据按小时观测的粒度，是本实验的重建选择，不是预热的定义要求。原 `jobs_F.csv` 未改，P0 的任务表已是可追溯的变换版本。', '',
        '源数据没有这些边界任务在精确时刻的真实提交时间、剩余服务或在原机群上的完整初始放置，因此新 P0 是**严格的仿真预热**，不能声称精确复原阿里源集群的初始节点状态。与旧 P0 的指标不可当作同一输入版本直接配对。', '',
        '## 3. 其他场景的口径', '',
        'P1–P5 是从真实任务模板重采样的受控场景，第1天已有合成到达，在第2天起点检查实际运行；它们不复制原F的边界 Pod。P6–P8 从原始连续小时记录重建，并读取测量来源窗口前一小时；首见于前一小时的 Pod 作为模拟0秒到达代理。P9继承P7的GPU任务。P6–P9 也先运行模拟第1天再起测量，原始小时记录和部分执行时长仍是代理。所有 P1–P9 在本表所列起点都已有实际运行任务，且占用按运行段积分。', '',
        '| 场景 | 源窗口前一小时首见Pod | 其中GPU任务 |',
        '|---|---:|---:|',
    ]
    for name in names[6:]:
        prior = [r for r in rows(ROOT / name / 'tasks.csv')
                 if r.get('first_relative_hour') == '-1']
        lines.append(f"| {name} | {len(prior):,} | {sum(float(r['gpu_request']) > 0 for r in prior):,} |")
    lines += [
        '', '这些是源窗口边界的小时级存在代理，不等于精确源提交时刻或精确初始放置；测量起点是随后模拟第2天，实际仍运行数见第1节。', '',
        '## 4. 复核字段和计算', '',
        '- 输入 `tasks.csv`：`pod_id, submit_time_sec, service_sec, gpu_request, cpu_request_cores`；P0 另看 `duration_match_status`，P6–P9 看 `first_relative_hour`。',
        '- 结果 `task_metrics.csv`：`pod_id, submit, service, first_start, finish, gpu`；`no_preempt` 的 `summary.json.preemptions=0`、`end_reason=drained`；`run_manifest.json.task_sha256` 与场景 `manifest.json.sha256` 一致。',
        r'- 边界 GPU/CPU 预留：对 `submit<t0` 且 `first_start<t0<finish` 的任务分别求 $\sum_i g_i$、$\sum_i c_i$。GPU/CPU 资源时间积分分别使用 $\sum_i g_i\,|[s_i,e_i)\cap[t_0,t_1)|$ 和 $\sum_i c_i\,|[s_i,e_i)\cap[t_0,t_1)|$；窗口前已开始但仍在运行的段自然贡献面积。引擎 `environment.py` 的 `gpu_area`、`cpu_area` 也用测量窗交集更新。',
        '- 端点采用半开区间；若任务恰在 $t_0$ 开始或结束，该时刻本身面积为0。上表“实际运行”按严格跨界计数，不把恰在 $t_0$ 启动的任务列为窗口前已运行。',
        '',
    ]
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'warmup_audit.json').write_text(json.dumps({'policy': 'no_preempt',
        'seed': 42, 'checkpoint_interval_sec': 3600, 'scenes': facts},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('wrote', OUT, 'verified scenes:', len(facts), flush=True)


if __name__ == '__main__':
    main()
