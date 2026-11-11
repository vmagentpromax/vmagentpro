"""One-time migration of the existing P0 suite to simulated pre-window occupancy."""
import json
import shutil
from pathlib import Path

from build_scenarios import (DEFAULT_OUT, SOURCE, P0_WARMUP_SEC, DAY,
                             audit, make_p0_strict, read, sha, write)


def main():
    root = DEFAULT_OUT.resolve()
    archive = (root / 'archive_pre_warmup_P0_20261107').resolve()
    assert archive.is_relative_to(root) and not archive.exists()
    outputs = sorted((root / 'outputs').glob('P0_real_F8d-*-s42-ck3600'))
    assert len(outputs) == 10 and all((p / 'run_manifest.json').exists() for p in outputs)
    archive.mkdir()
    (archive / 'outputs').mkdir()
    for src in (root / 'P0_real_F8d/tasks.csv', root / 'manifest.json',
                root / 'comparison_all.csv', root / '首轮验证结果.md'):
        if src.exists():
            shutil.copy2(src, archive / src.name)
    report = Path('C:/Users/trw/Desktop/论文/vmagentpro准备工作/十场景全策略评价结果与分析_20261107.md')
    if report.exists():
        shutil.copy2(report, archive / '原十场景全策略评价结果与分析.md')
    for src in outputs:
        dest = archive / 'outputs' / src.name
        assert src.resolve().is_relative_to(root) and dest.resolve().is_relative_to(root)
        src.rename(dest)

    original = read(SOURCE / 'jobs_F.csv')
    rows = make_p0_strict(original)
    p0 = root / 'P0_real_F8d'
    window = [P0_WARMUP_SEC, P0_WARMUP_SEC + 8 * DAY]
    write(p0 / 'tasks.csv', rows)
    (p0 / 'tasks.json').write_text(json.dumps({'measurement_window_sec': window,
                                               'tasks': rows}, ensure_ascii=False), encoding='utf-8')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    nodes = read(root / 'nodes.csv')
    meta = {'kind': 'F trace-derived reconstruction with simulated boundary warmup',
            'measurement_window_sec': window, 'arrival_horizon_sec': window[1],
            'max_sim_time_sec': 90 * DAY + P0_WARMUP_SEC,
            'source_boundary_presence_tasks': sum(float(r['submit_time_sec']) == 0 for r in original),
            'warmup_duration_sec': P0_WARMUP_SEC,
            'warmup_construction': 'F t=0 presence_remaining pods arrive at simulation t=0 and gain 3600 seconds of preparatory service; all other F arrivals shift by 3600 seconds',
            'sha256': sha(p0 / 'tasks.csv'), **audit(rows, nodes, window)}
    manifest['scenarios']['P0_real_F8d'] = meta
    manifest['assumptions'][0] = 'P0 has a 1-hour simulated warmup for F boundary-presence pods; 3600 seconds preparatory service preserves original boundary remaining service when uninterrupted'
    manifest['assumptions'].insert(1, 'P1-P9 use 1-day simulated warmup + 14-day measurement; empty initial pool')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('P0 migrated; old inputs and 10 old runs archived in', archive, flush=True)
    print('new P0 input SHA256:', meta['sha256'], 'window:', window, flush=True)


if __name__ == '__main__':
    main()
