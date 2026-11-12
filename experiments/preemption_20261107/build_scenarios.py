"""Trace-derived controlled scenarios; Python standard library only."""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

DAY = 86400
P0_WARMUP_SEC = 3600
REPO_ROOT = Path(__file__).resolve().parents[2]
# Scene F inputs are versioned in the repository; override VMAGENTPRO_SCENE_F to read elsewhere.
SOURCE = Path(os.environ.get('VMAGENTPRO_SCENE_F', REPO_ROOT / 'runs/hetero8d_20261101/inputs'))
# Derived scenarios are regenerable outputs, so they stay out of version control (/outputs/gpu/ is gitignored).
DEFAULT_OUT = Path(os.environ.get('VMAGENTPRO_PREEMPTION_OUT', REPO_ROOT / 'outputs/gpu/preemption_20261107'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def overlap(a,b,c,d):
    return max(0., min(b,d)-max(a,c))

def make_p0_strict(raw):
    """Replay F's boundary-presence pods before its original eight-day window."""
    assert raw and all(float(r['submit_time_sec']) >= 0 for r in raw)
    assert any(float(r['submit_time_sec']) == 0 for r in raw)
    assert all(r['duration_match_status'] == 'presence_remaining'
               for r in raw if float(r['submit_time_sec']) == 0)
    rows = []
    for original in raw:
        r = dict(original)
        old = float(r['submit_time_sec'])
        if old > 0:
            r['submit_time_sec'] = str(old + P0_WARMUP_SEC)
        else:
            # F's boundary duration estimates *remaining* time from its old t=0.
            # Prepend warmup work so uninterrupted residual service at the new
            # measurement start still equals that estimate.
            r['service_sec'] = str(float(r['service_sec']) + P0_WARMUP_SEC)
        rows.append(r)
    return rows

def audit(rows, nodes, window, check_ids=True):
    a,b=window
    caps={m:sum(float(n['gpu_count']) for n in nodes if n['gpu_spec_public']==m)
          for m in sorted({n['gpu_spec_public'] for n in nodes})}
    gc=sum(caps.values()); cc=sum(float(n['cpu_capacity_cores']) for n in nodes)
    gg=cp=0.; model=Counter(); priorities=Counter(); full=Counter()
    assert not check_ids or len({r['pod_id'] for r in rows})==len(rows)
    for r in rows:
        g,c,s,p=map(float,[r['gpu_request'],r['cpu_request_cores'],r['submit_time_sec'],r['service_sec']])
        assert s>=0 and p>0 and any((g==0 or n['gpu_spec_public']==r['gpu_spec_public']) and
            g<=float(n['gpu_count']) and c<=float(n['cpu_capacity_cores']) for n in nodes)
        dt=overlap(s,s+p,a,b); gg+=g*dt; cp+=c*dt
        model[r['gpu_spec_public']]+=g*dt
        priorities[r['priority_class']]+=g*dt
        full[r['priority_class']]+=g*p
    return {'tasks':len(rows),'gpu_tasks':sum(float(r['gpu_request'])>0 for r in rows),
        'priority_counts':dict(Counter(r['priority_class'] for r in rows)),
        'offered_gpu_ratio':gg/gc/(b-a),'offered_cpu_ratio':cp/cc/(b-a),
        'per_model_offered_gpu_ratio':{m:model[m]/caps[m]/(b-a) for m in caps},
        'priority_offered_gpu_ratio':{p:v/gc/(b-a) for p,v in priorities.items()},
        'total_service_gpu_hours_by_priority':{p:v/3600 for p,v in full.items()},
        'unique_source_pods':len({r.get('source_pod_id',r['pod_id']) for r in rows}),
        'empty_node_infeasible_tasks':0}

def sample_group(pool, target, seed, end, burst=False):
    rng=random.Random(seed); result=[]; work=0.
    assert pool, 'Empty template stratum'
    while work<target:
        r=dict(rng.choice(pool)); result.append(r)
        work+=float(r['gpu_request'] or 0)*float(r['service_sec']) if float(r['gpu_request'])>0 else float(r['cpu_request_cores'])*float(r['service_sec'])
    # Stratification limits accidental all-day gaps without concealing bursts.
    for i,r in enumerate(result):
        t=(i+rng.random())/len(result)*end
        if burst:
            t=int(t//(6*3600))*6*3600 + (t%(6*3600))/12
        r['source_pod_id']=r['pod_id']
        r['source_submit_time_sec']=r['submit_time_sec']
        r['submit_time_sec']=round(t,6)
        r['construction']='joint_template_resampling; original gpu/cpu/model/service/priority retained'
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--seed',type=int,default=20261107); args=ap.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Refusing to overwrite nonempty {out}')
    out.mkdir(parents=True,exist_ok=True)
    raw=read(SOURCE/'jobs_F.csv'); nodes=read(SOURCE/'nodes.csv')
    shutil.copyfile(SOURCE/'nodes.csv',out/'nodes.csv')
    # Pure GPU competition strata: no relabelling, no independent resource shuffling.
    pool=[r for r in raw if float(r['submit_time_sec'])>0 and
          300<=float(r['service_sec'])<=DAY and float(r['gpu_request'])>0 and
          r['priority_class'] in ('HP','LP') and
          float(r['cpu_request_cores'])/float(r['gpu_request'])<=12 and
          (r['priority_class']=='HP' or float(r['service_sec'])>=1800)]
    horizon=15*DAY; window=[DAY,horizon]
    configs=[('P1_steady',.35,.70,False,False),('P2_burst',.35,.70,True,False),
             ('P3_overload',.35,.95,True,False),('P4_model_skew',.35,.70,True,True)]
    specs={}; saved_burst=None
    for name,hp,lp,burst,skew in configs:
        rows=[]
        for mi,(m,cap) in enumerate([('H20',80),('XPU-A',160)]):
            for pi,(priority,ratio) in enumerate([('HP',hp),('LP',lp)]):
                # Same cluster-average intensity, concentrated demand on H20.
                factor=(1.5/1.05 if m=='H20' else .825/1.05) if skew else 1.
                sub=[r for r in pool if r['gpu_spec_public']==m and r['priority_class']==priority]
                rows+=sample_group(sub,ratio*factor*cap*horizon,args.seed+mi*100+pi,horizon,
                                   burst and priority=='HP')
        specs[name]=(rows,window,{'hp_workload_target':hp,'lp_workload_target':lp,
             'hp_bursts':'each 6h compressed into first 30min' if burst else None,
             'model_skew':skew})
        if name=='P2_burst': saved_burst=rows
    cpu_pool=[r for r in raw if float(r['gpu_request'])==0 and
              16<=float(r['cpu_request_cores'])<=96 and 300<=float(r['service_sec'])<=DAY and
              float(r['submit_time_sec'])>0]
    foreground=audit(saved_burst,nodes,window,check_ids=False)['offered_cpu_ratio']
    cpu=sample_group(cpu_pool,max(0.,1.05-foreground)*5760*horizon,args.seed+999,horizon)
    specs['P5_cpu_contention']=([dict(r) for r in saved_burst]+cpu,window,
        {'gpu_sequence_identical_to':'P2_burst (except scenario-local IDs)',
         'cpu_target_approx':1.05,'cpu_only_service':'inherited observed-presence proxy'})
    manifest={'schema_version':1,'seed':args.seed,'source':str(SOURCE),
        'source_sha256':{f:sha(SOURCE/f) for f in ('nodes.csv','jobs_F.csv')},
        'template_filter':'arrival>0; HP/LP GPU; 300<=service<=86400; cpu/gpu<=12; LP service>=1800',
        'template_strata':dict(Counter(r['priority_class']+'|'+r['gpu_spec_public'] for r in pool)),
        'assumptions':['P0 has a 1-hour simulated warmup for F boundary-presence pods; 3600 seconds preparatory service preserves original boundary remaining service when uninterrupted',
            'P1-P9 use 1-day simulated warmup + 14-day measurement; empty initial pool',
            'duration is inherited execution-span proxy, not measured uninterrupted GPU work',
            'LP eligibility assumed from priority label; no Standby event reconstruction',
            'templates sampled with replacement; submission times synthetic; demand tuples retained',
            'one task per pod, one homogeneous node; no gang, network, checkpoint I/O or eviction delay'],
        'scenarios':{}}
    real=out/'P0_real_F8d'; real.mkdir()
    p0_rows=make_p0_strict(raw)
    p0_window=[P0_WARMUP_SEC,P0_WARMUP_SEC+8*DAY]
    write(real/'tasks.csv',p0_rows)
    (real/'tasks.json').write_text(json.dumps({'measurement_window_sec':p0_window,'tasks':p0_rows},ensure_ascii=False),encoding='utf-8')
    manifest['scenarios']['P0_real_F8d']={'kind':'F trace-derived reconstruction with simulated boundary warmup',
        'measurement_window_sec':p0_window,'arrival_horizon_sec':p0_window[1],
        'max_sim_time_sec':90*DAY+P0_WARMUP_SEC,
        'source_boundary_presence_tasks':sum(float(r['submit_time_sec'])==0 for r in raw),
        'warmup_duration_sec':P0_WARMUP_SEC,
            'warmup_construction':'F t=0 presence_remaining pods arrive at simulation t=0 and gain 3600 seconds of preparatory service; all other F arrivals shift by 3600 seconds',
        'sha256':sha(real/'tasks.csv'),**audit(p0_rows,nodes,p0_window)}
    for name,(rows,win,details) in specs.items():
        rows.sort(key=lambda r:(float(r['submit_time_sec']),r['source_pod_id']))
        for i,r in enumerate(rows): r['pod_id']=f'{name}-{i:07d}'
        folder=out/name; folder.mkdir(); write(folder/'tasks.csv',rows)
        meta={'kind':'controlled trace-template resampling','measurement_window_sec':win,
              'arrival_horizon_sec':horizon,'max_sim_time_sec':90*DAY,
              'sha256':sha(folder/'tasks.csv'),**details,**audit(rows,nodes,win)}
        manifest['scenarios'][name]=meta
        # JSON can be used directly by the existing CLI, including a nonzero warmup boundary.
        (folder/'tasks.json').write_text(json.dumps({'measurement_window_sec':win,'tasks':rows},ensure_ascii=False),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    for name,m in manifest['scenarios'].items():
        print(name,m['tasks'],round(m['offered_gpu_ratio'],4),round(m['offered_cpu_ratio'],4),flush=True)

if __name__=='__main__': main()
