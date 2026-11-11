"""Collect all completed runs and audit provenance. Re-run after additional experiments."""
import csv,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_scenarios import DEFAULT_OUT,read,write
from harness import percentile,average

root=DEFAULT_OUT;manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
rows=[]
for path in sorted((root/'outputs').glob('*/run_manifest.json')):
    run=json.loads(path.read_text(encoding='utf-8'));s=json.loads(path.with_name('summary.json').read_text())
    m=json.loads(path.with_name('metrics.json').read_text());detail=read(path.with_name('task_metrics.csv'))
    start,end=run['measurement_window_sec']
    gpugroups={}
    for pri in ('HP','LP'):
        cohort=[r for r in detail if r['priority']==pri and float(r['gpu'])>0 and start<=float(r['submit'])<end]
        finished=[r for r in cohort if r['finish']!=''];started=[r for r in cohort if r['response_sec']!='']
        mature=[r for r in cohort if float(r['submit'])+60<=end]
        gpugroups[pri+'_GPU']={'n':len(cohort),'completed_eventually':len(finished),
          'mean_jct_sec_completed':average([float(r['jct_sec']) for r in finished]),
          'p95_jct_sec_completed':percentile([float(r['jct_sec']) for r in finished],.95),
          'response_p95_sec_started':percentile([float(r['response_sec']) for r in started],.95),
          'slo60_violation_ratio':sum(r['first_start']=='' or float(r['first_start'])>float(r['submit'])+60 for r in mature)/len(mature) if mature else None,
          'completion_rate_at_cutoff':sum(float(r['finish'])<=end for r in finished)/len(cohort) if cohort else None}
    path.with_name('gpu_priority_metrics.json').write_text(json.dumps(gpugroups,indent=2),encoding='utf-8')
    rows.append({'scenario':run['scenario'],'policy':run['policy'],'seed':run['seed'],
       'checkpoint_sec':run['checkpoint_interval_sec'],'completed':s['completed'],'rejected':s['rejected'],
       'unfinished':s['unfinished'],'end_reason':s['end_reason'],'gpu_allocated_ratio':m['allocated_gpu_ratio'],
       'gpu_useful_ratio':m['useful_gpu_ratio_retrospective'],'lost_gpu_hours_window':m['wasted_gpu_hours_window'],
       'preemptions_full_run':s['preemptions'],'hp_gpu_p95_response_sec':gpugroups['HP_GPU']['response_p95_sec_started'],
       'hp_gpu_slo60_violation_ratio':gpugroups['HP_GPU']['slo60_violation_ratio'],
       'lp_gpu_mean_jct_sec':gpugroups['LP_GPU']['mean_jct_sec_completed'],
       'lp_gpu_completion_at_cutoff':gpugroups['LP_GPU']['completion_rate_at_cutoff'],
       'conservation_max_error_sec':m['completed_work_conservation_max_error_sec']})
comparison_name='comparison_all.csv'
try:
    write(root/comparison_name,rows)
except PermissionError:
    # A spreadsheet preview may hold the previous report open on Windows.
    comparison_name='comparison_all_strict_warmup.csv'
    write(root/comparison_name,rows)
    print('old comparison_all.csv is open; wrote',comparison_name,flush=True)
lines=['# 抢占场景首轮验证结果','',
 '本表使用 fast 时钟、固定场景输入；除列出的敏感性实验外 checkpoint=3600 秒、seed=42。',
 'HP/LP 的结果只统计 GPU 任务，避免 CPU-only 数量稀释抢占效果。JCT 为测量窗内到达队列追踪至排空的完成时间。',
 'GPU有效工作率是扣除最终丢失进度的回溯量，不是硬件SM利用率；到达窗口后的排空阶段不混入占用率分母。','',
 '| 场景 | 策略 | seed | ck(s) | 占用% | 有效工作% | 损失GPU·h | HP GPU P95响应(s) | LP GPU平均JCT(h) | LP窗末完成% |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    def fmt(x,scale=1):return '-' if x is None else f'{x*scale:.2f}'
    lines.append('| '+ ' | '.join([r['scenario'],r['policy'],str(r['seed']),str(r['checkpoint_sec']),
       fmt(r['gpu_allocated_ratio'],100),fmt(r['gpu_useful_ratio'],100),fmt(r['lost_gpu_hours_window']),
       fmt(r['hp_gpu_p95_response_sec']),fmt(r['lp_gpu_mean_jct_sec'],1/3600),fmt(r['lp_gpu_completion_at_cutoff'],100)])+' |')
lines+=['',f'已完成 {len(rows)} 组；完整数据见 {comparison_name}，各组目录内保留输入和代码 SHA256、逐任务明细、原 summary 与新增 metrics。',
       '正式论文建议增加独立场景构造种子及更多非重叠窗口；当前重复随机策略种子只衡量给定工作负载下的算法随机性，不代表数据总体置信区间。']
(root/'首轮验证结果.md').write_text('\n'.join(lines),encoding='utf-8')
scene=['# 场景清单','', '| 场景 | 任务数 | GPU任务 | GPU提供负载 | CPU提供负载 | 说明 |','|---|---:|---:|---:|---:|---|']
scene_descriptions={
    'P0_real_F8d':'原F边界任务先仿真预热1小时；随后测量8天，CPU压力较高',
    'P1_steady':'真实任务模板重采样；15天持续HP/LP竞争',
    'P2_burst':'与P1使用相同工作模板；HP每6小时集中突发',
    'P3_overload':'提高LP任务量；GPU持续过载，观察积压',
    'P4_model_skew':'H20局部高压、XPU-A相对宽松；观察型号瓶颈',
    'P5_cpu_contention':'与P2相同GPU任务；叠加CPU-only任务制造CPU竞争',
    'P6_real_H20_early15d':'H20第48–62天连续窗口；时长为出现跨度代理，仅作探索',
    'P7_real_H20_15d':'H20第158–172天连续窗口；保留CPU-only任务',
    'P8_real_XPUD_15d':'XPU-D第132–146天连续窗口；固定节点池重建',
    'P9_H20_gpu_capacity_pressure':'保留P7全部GPU任务；节点池从30台缩至19台',
}
for name,m in sorted(manifest['scenarios'].items(),key=lambda item:int(item[0].split('_',1)[0][1:])):
    scene.append(f"| {name} | {m['tasks']:,} | {m['gpu_tasks']:,} | {m['offered_gpu_ratio']:.2%} | {m['offered_cpu_ratio']:.2%} | {scene_descriptions[name]} |")
(root/'场景清单.md').write_text('\n'.join(scene),encoding='utf-8')
print('collected',len(rows),'runs',len(manifest['scenarios']),'scenarios')
