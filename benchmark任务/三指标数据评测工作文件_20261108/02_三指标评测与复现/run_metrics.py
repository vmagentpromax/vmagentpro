import concurrent.futures,copy,csv,hashlib,json,math,platform,time
from pathlib import Path
from engine import simulate,POLICIES
from source_verifier import replay

ROOT=Path(__file__).resolve().parent

def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def near(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-5),(a,b)

def core_metrics(initial,arrivals,events,cfg):
    req={p['request_id']:p for p in arrivals}
    pods={p['pod_id']:p for p in initial+arrivals}
    new_ids={p['pod_id'] for p in arrivals}
    active={p['pod_id']:p for p in initial}
    starts={};complete={};pauses={};current=None;previous=0.;work=0.;total_work=0.;cost=0.
    window=cfg['allocation_window_sec']
    def advance(t):
        nonlocal previous,work,total_work
        dt=max(0.,min(t,window)-min(previous,window))
        work+=dt*sum(p['gpu_request'] for pid,p in active.items() if pid in new_ids and pid!=current)
        total_work+=dt*sum(p['gpu_request'] for pid,p in active.items() if pid!=current)
        previous=t
    for e in events:
        t=e['time_sec'];advance(t)
        if e['event']=='start':
            p=req[e['request_id']];starts[e['request_id']]=t;active[p['pod_id']]=p
        elif e['event']=='completion':
            complete[e['pod_id']]=t;del active[e['pod_id']]
        elif e['event']=='migration_start':
            assert current is None
            current=e['pod_id'];cost+=pods[current]['gpu_request']*e['pause_sec']
            pauses.setdefault(current,[]).append((t,t+e['pause_sec']))
        elif e['event']=='migration_complete':current=None
    advance(cfg['horizon_sec'])
    # Independent per-task interval calculation checks the integral, excluding
    # initial Pods, queueing, reserved GPUs, and all overlapping migration pauses.
    work_by_intervals=0.
    for rid,s in starts.items():
        p=req[rid];end=min(window,complete.get(p['pod_id'],cfg['horizon_sec']))
        dt=max(0.,end-max(0.,s))
        for a,b in pauses.get(p['pod_id'],[]):dt-=max(0.,min(end,b)-max(s,a,0.))
        assert dt>=-1e-7
        work_by_intervals+=p['gpu_request']*dt
    near(work,work_by_intervals)
    out={'A_GPU':None,'W_new_GPU_sec':work,'C_pause_GPU_sec':cost,'W_new_GPU_h':work/3600.,
         'A_GPU_large':None,'W_new_large_GPU_sec':0.,'total_effective_work_3600':total_work}
    for suffix,subset in [('',arrivals),('_large',[p for p in arrivals if p['gpu_request']>=4])]+[(f'_{g}',[p for p in arrivals if p['gpu_request']==g]) for g in [1,2,4,8]]:
        den=sum(p['gpu_request'] for p in subset)
        num=sum(p['gpu_request'] for p in subset if p['request_id'] in starts and starts[p['request_id']]-p['arrival_time_sec']<=cfg['start_deadline_sec']+1e-7)
        out['A_GPU'+suffix]=num/den if den else None
    for p in arrivals:
        s=starts.get(p['request_id'])
        if s is None or p['gpu_request']<4:continue
        end=min(window,complete.get(p['pod_id'],cfg['horizon_sec']))
        dt=max(0.,end-s)
        for a,b in pauses.get(p['pod_id'],[]):dt-=max(0.,min(end,b)-max(s,a))
        out['W_new_large_GPU_sec']+=p['gpu_request']*dt
    assert 0<=work<=sum(p['gpu_request']*p['duration_sec'] for p in arrivals)+1e-5
    return out

def test_semantics():
    hosts=[dict(server_id=s,gpu_count=8,cpu_capacity_cores=32,gpu_spec_public='H20') for s in ['a','b']]
    initial=[dict(pod_id='p'+s,server_id=s,gpu_request=4,cpu_request_cores=4,gpu_spec_public='H20',migratable=True,remaining_duration_sec=1000.) for s in ['a','b']]
    arrivals=[dict(pod_id='r',request_id='r',gpu_request=8,cpu_request_cores=8,gpu_spec_public='H20',migratable=True,arrival_time_sec=0.,duration_sec=100.)]
    cfg=load(ROOT/'S1.json')['config'];cfg.update(horizon_sec=200.,allocation_window_sec=200.,start_deadline_sec=60.)
    for budget,expected_a,expected_w,expected_c in [(0,0,0,0),(209.5,0,0,0),(209.6,1,800,209.6)]:
        cfg['migration']['max_gpu_pause_sec']=budget
        m,ev=simulate(hosts,initial,arrivals,cfg,'first_fit','one_move');r=core_metrics(initial,arrivals,ev,cfg)
        near(r['A_GPU'],expected_a);near(r['W_new_GPU_sec'],expected_w);near(r['C_pause_GPU_sec'],expected_c)
    cfg.update(horizon_sec=20.,allocation_window_sec=20.,start_deadline_sec=10.)
    m,ev=simulate(hosts,initial,arrivals,cfg,'first_fit','one_move');r=core_metrics(initial,arrivals,ev,cfg)
    near(r['C_pause_GPU_sec'],209.6);near(m['migration_gpu_pause_accrued_sec'],80.)
    req=[dict(pod_id='x',request_id='x',gpu_request=1,arrival_time_sec=0.,duration_sec=10.),dict(pod_id='y',request_id='y',gpu_request=8,arrival_time_sec=0.,duration_sec=10.)]
    r=core_metrics([],req,[dict(time_sec=0.,event='start',request_id='x',pod_id='x'),dict(time_sec=10.,event='completion',pod_id='x')],cfg)
    near(r['A_GPU'],1/9);near(r['W_new_GPU_sec'],10.)
    print('PASS semantic tests: GPU weighting, new-only work, pauses, exact budget boundary and full in-flight cost.',flush=True)

def run_one(job):
    scenario,placement,migration,budget=job
    data=load(ROOT/(scenario+'.json'));cfg=copy.deepcopy(data['config'])
    if budget is not None:cfg['migration']['max_gpu_pause_sec']=budget
    started=time.perf_counter()
    metrics,events=simulate(data['servers'],data['initial_pods'],data['arrivals'],cfg,placement,migration)
    label='original' if budget is None else str(budget)
    folder=ROOT/'logs_py311'/scenario/label;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(placement+'__'+migration+'.json')
    path.write_text(json.dumps({'metrics':metrics,'events':events},ensure_ascii=False,allow_nan=False),encoding='utf8')
    audit=replay(path,data['initial_pods'],data['servers'],data['arrivals'],cfg)
    core=core_metrics(data['initial_pods'],data['arrivals'],events,cfg)
    near(core['C_pause_GPU_sec'],metrics['migration_gpu_pause_sec'])
    near(core['total_effective_work_3600'],metrics['gpu_effective_allocation_rate']*800*3600)
    if budget is not None:assert core['C_pause_GPU_sec']<=budget+1e-7
    else:
        old_key=placement+'__'+migration
        old=data['old_metrics'].get(old_key)
        if old is None:
            matches=[m for m in data['old_metrics'].values() if m['placement']==placement and m['migration']==migration]
            assert len(matches)==1;old=matches[0]
        for k in ['started','migration_count','migration_gpu_pause_sec','gpu_effective_allocation_rate','start_within_deadline_rate']:
            near(metrics[k],old[k])
    core.pop('total_effective_work_3600')
    row=dict(scenario=scenario,placement=placement,migration=migration,budget_GPU_sec=budget,**core,
             diagnostics_migration_count=metrics['migration_count'],diagnostics_unstarted=metrics['unstarted'],
             events=len(events),resource_replay_passed=True,metric_checks_passed=True,
             original_reproduction_passed=budget is None,wall_seconds=time.perf_counter()-started,
             log_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return row

def main():
    test_semantics()
    jobs=[(s,p,m,b) for b in [None,0,1440,2880,8640] for s in ['S1','S2','S3','S4'] for p,m in POLICIES]
    rows=[];begin=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run_one,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:row=future.result()
            except Exception as exc:
                print('FAILED JOB '+str(futures[future])+' '+repr(exc),flush=True)
                for pending in futures:pending.cancel()
                raise
            rows.append(row)
            print(f'{len(rows)}/{len(jobs)} {row["scenario"]} {row["placement"]}/{row["migration"]} B={row["budget_GPU_sec"]} A={row["A_GPU"]:.6f} W={row["W_new_GPU_sec"]:.1f} C={row["C_pause_GPU_sec"]:.1f}',flush=True)
    rows.sort(key=lambda r:(-1 if r['budget_GPU_sec'] is None else r['budget_GPU_sec'],r['scenario'],r['placement'],r['migration']))
    (ROOT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    with (ROOT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for file,digest in load(ROOT/'source_manifest.json').items():
        source=Path(file)
        if not source.is_absolute():source=ROOT/source
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    report=dict(run_count=len(rows),original_reproductions=32,budget_runs=128,semantic_tests_passed=True,
                all_event_replays_passed=True,all_metric_cross_checks_passed=True,source_inputs_unchanged=True,
                elapsed_wall_seconds=time.perf_counter()-begin,python=platform.python_version(),
                engine_sha256=hashlib.sha256((ROOT/'engine.py').read_bytes()).hexdigest())
    (ROOT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
