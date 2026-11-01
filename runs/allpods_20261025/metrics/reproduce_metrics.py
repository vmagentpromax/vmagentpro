"""Reproduce the five newly fetched all-pod workloads and record finite-run metrics."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from vmagent.gpu.run import make_environment

HERE = Path(__file__).resolve().parent
INPUTS = ROOT / 'runs/allpods_20261025/inputs'
OUT = HERE / 'reproduced_runs'


def compute(env):
    rows = list(env.records.values())
    t0 = min(r['submit_time_sec'] for r in rows)
    t1 = max(r['finish_time_sec'] for r in rows)
    duration = t1 - t0
    running, waiting = set(), set()
    gpu_area = fragment_time = pending_time = 0.0
    for event, next_event in zip(env.events, env.events[1:]):
        pid, kind = event['pod_id'], event['event']
        if kind == 'arrival':
            waiting.add(pid)
        elif kind == 'start':
            waiting.discard(pid)
            running.add(pid)
        elif kind == 'complete':
            running.discard(pid)
        elif kind == 'reject':
            waiting.discard(pid)
        dt = max(0.0, min(t1, next_event['time_sec']) - max(t0, event['time_sec']))
        if not dt:
            continue
        free_g = {n.server_id: n.gpu_capacity for n in env.nodes}
        free_c = {n.server_id: n.cpu_capacity for n in env.nodes}
        for rid in running:
            row = env.records[rid]
            free_g[row['server_id']] -= row['gpu_request']
            free_c[row['server_id']] -= row['cpu_request_cores']
        fragmentation = False
        for wid in waiting:
            task = env.by_id[wid]
            compatible = [n for n in env.nodes if task.gpu == 0 or n.gpu_type in task.allowed_gpu_types]
            enough_total = (sum(free_g[n.server_id] for n in compatible) + 1e-8 >= task.gpu
                            and sum(free_c[n.server_id] for n in compatible) + 1e-8 >= task.cpu)
            fits_node = any(free_g[n.server_id] + 1e-8 >= task.gpu
                            and free_c[n.server_id] + 1e-8 >= task.cpu for n in compatible)
            if enough_total and not fits_node:
                fragmentation = True
                break
        gpu_area += event['gpu_reserved'] * dt
        fragment_time += fragmentation * dt
        pending_time += bool(waiting) * dt
    work = sum(t.gpu * t.service_time for t in env.tasks)
    assert math.isclose(gpu_area, work, rel_tol=1e-10, abs_tol=1e-5)
    assert math.isclose(duration, env.current_time - t0, abs_tol=1e-7)
    assert 0 <= fragment_time <= pending_time + 1e-8
    jct = sum(r['finish_time_sec'] - r['submit_time_sec'] for r in rows) / len(rows)
    wait = sum(r['start_time_sec'] - r['submit_time_sec'] for r in rows) / len(rows)
    assert math.isclose(jct, wait + sum(t.service_time for t in env.tasks) / len(rows), abs_tol=1e-7)
    return dict(task_count=len(rows), mean_jct_sec=jct, makespan_sec=duration,
                mean_wait_sec=wait, agu_reserved=gpu_area / (env.total_gpu * duration),
                afr_joint=fragment_time / duration, throughput_tasks_per_sec=len(rows) / duration,
                throughput_tasks_per_hour=len(rows) * 3600 / duration,
                gpu_work_gpu_sec=work, fragment_time_sec=fragment_time,
                first_submit_sec=t0, final_finish_sec=t1,
                agu_first_6h=env.summary()['reserved_gpu_ratio'])


def main():
    results = []
    for scene in 'ABCDE':
        task_path = INPUTS / f'jobs_{scene}.csv'
        env, _ = make_environment(INPUTS / 'nodes.csv', task_path, clock='fast',
                                  queue_policy='backfill', measurement_window=[0, 21600],
                                  max_sim_time=150000, seed=42)
        summary = env.run('first_fit')
        assert summary['completed'] == len(env.tasks)
        assert summary['unfinished'] == summary['rejected'] == 0
        assert summary['end_reason'] == 'drained'
        assert all(abs(g - n.gpu_capacity) < 1e-7 and abs(c - n.cpu_capacity) < 1e-7
                   for n, g, c in zip(env.nodes, env.free_gpu, env.free_cpu))
        metric = dict(scene=scene, policy='first_fit', **compute(env))
        dest = OUT / scene
        env.export(dest)
        (dest / 'metrics.json').write_text(json.dumps(metric, indent=2), encoding='utf-8')
        results.append(metric)
        print(json.dumps(metric), flush=True)
    (HERE / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (HERE / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    manifest = dict(policy='first_fit', queue_policy='backfill', seed=42,
                    clock='fast', measurement_seconds=21600, max_sim_time=150000,
                    metric_window='first submit to last completion',
                    afr_definition='positive-duration intervals with a pending task whose aggregate compatible GPU and CPU resources suffice but no single node jointly fits, divided by makespan',
                    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS.glob('*.csv')})
    (HERE / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
