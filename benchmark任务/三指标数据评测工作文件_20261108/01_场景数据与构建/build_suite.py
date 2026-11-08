"""Build trace-derived controlled layouts, evaluate a fixed candidate grid.
No policy winner is used by selection; all generated candidates are retained.
"""
from __future__ import annotations
import argparse,collections,copy,hashlib,json,platform,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import source_adapter as src
from engine import State,simulate,plans,POLICIES,move_seconds

SEEDS=[20261107,20261108,20261109]
GRID=[(s,i) for i in (10.,16.) for s in SEEDS]+[(SEEDS[0],40.)]
HIST={'S1':[8]*25+[2]*20+[0]*55,'S2':[6]*20+[4]*20+[2]*20+[0]*40,'S3':[1]*10+[2]*40+[3]*50}

def dump(p,x):src.dump(Path(p),x)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def read_inputs(p):return [pd.read_parquet(p/f).to_dict('records') for f in ('servers.parquet','initial_pods.parquet','arrivals.parquet')]

def layout(hosts,initial,scenario,seed):
    """Exact GPU free histogram; CPU checked without altering Pod requests."""
    if scenario=='S3':return initial.copy()
    for attempt in range(5):
        rng=np.random.default_rng(seed+attempt*313)
        free=np.array(HIST[scenario]);rng.shuffle(free)
        want=hosts.gpu_count.to_numpy()-free;cpu=hosts.cpu_capacity_cores.to_numpy().astype(float)
        assigned={};remaining=want.copy()
        order=initial.assign(tie=rng.random(len(initial))).sort_values(['gpu_request','cpu_request_cores','tie'],ascending=[False,False,True])
        for idx,p in order.iterrows():
            candidates=np.where((remaining>=p.gpu_request)&(cpu+1e-8>=p.cpu_request_cores))[0]
            if not len(candidates):break
            # Tight GPU packing; CPU headroom breaks ties, followed by seeded jitter.
            j=min(candidates,key=lambda k:(remaining[k]-p.gpu_request,-cpu[k],rng.random()))
            assigned[idx]=hosts.iloc[j].server_id;remaining[j]-=int(p.gpu_request);cpu[j]-=p.cpu_request_cores
        else:
            assert not remaining.any()
            out=initial.copy();out['server_id']=[assigned[i] for i in initial.index]
            State(hosts.to_dict('records'),out.to_dict('records')).check();return out
    # A failure of the greedy constructor is not an infeasibility claim.
    # Aggregate identical demands, solve integral per-host assignment, then map IDs.
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import lil_matrix
    groups=list(initial.groupby(['gpu_request','cpu_request_cores'],sort=True))
    n=len(groups);nh=len(hosts);mat=lil_matrix((n+2*nh,n*nh));ub=np.zeros(n+2*nh);lb=ub.copy()
    for k,((g,c),frame) in enumerate(groups):
        mat[k,k*nh:(k+1)*nh]=1;lb[k]=ub[k]=len(frame)
        for j in range(nh):mat[n+j,k*nh+j]=g;mat[n+nh+j,k*nh+j]=c
    for j in range(nh):lb[n+j]=ub[n+j]=want[j];lb[n+nh+j]=0;ub[n+nh+j]=hosts.iloc[j].cpu_capacity_cores
    result=milp(np.zeros(n*nh),integrality=np.ones(n*nh),bounds=Bounds(0,np.inf),constraints=LinearConstraint(mat.tocsr(),lb,ub),options={'time_limit':60.})
    if result.x is None:raise RuntimeError(f'Layout search unproven {scenario}, seed {seed}: {result.message}')
    counts=np.rint(result.x).astype(int).reshape(n,nh);assigned={}
    for k,(_,frame) in enumerate(groups):
        ids=list(frame.index);offset=0
        for j in range(nh):
            for idx in ids[offset:offset+counts[k,j]]:assigned[idx]=hosts.iloc[j].server_id
            offset+=counts[k,j]
        assert offset==len(ids)
    out=initial.copy();out['server_id']=[assigned[i] for i in initial.index]
    st=State(hosts.to_dict('records'),out.to_dict('records'));st.check()
    assert sorted(int(st.free(s)[0]) for s in st.ids)==sorted(HIST[scenario])
    return out

def workload(hosts,pool,summary,seed,interval):
    src.SEED=seed
    hosts,initial,arrivals=src.construct(hosts,pool,summary)
    arrivals['arrival_time_sec']*=interval/35.
    arrivals['duration_base_sec']=arrivals.duration_sec
    initial['duration_base_sec']=initial.duration_sec
    rng=np.random.default_rng(seed+10000)
    rest=pool[~pool.pod_id.isin(initial.pod_id)]
    raw=pd.concat([src.stratified_sample(rest[rest.gpu_request.eq(g)],n,rng) for g,n in [(1,12),(2,8),(4,40),(8,60)]])
    raw=raw.iloc[rng.permutation(len(raw))].reset_index(drop=True)
    s4=src.decorate(raw,summary,rng,'arrival');s4['duration_base_sec']=s4.duration_sec
    factor=float((arrivals.gpu_request*arrivals.duration_sec).sum()/(s4.gpu_request*s4.duration_sec).sum())
    s4['duration_sec']*=factor
    # Four fixed-width bursts spread across the same arrival window.
    span=float(arrivals.arrival_time_sec.max());times=[]
    for b in range(4):times.extend(b*span*.30+rng.uniform(0,span*.10,30))
    s4['arrival_time_sec']=sorted(times)
    s4['request_id']=[f'req_{i:04d}' for i in range(len(s4))]
    return hosts,initial,arrivals,s4,factor

def config(scenario,seed,interval,factor):
    return {'schema_version':2,'scenario':scenario,'seed':seed,'rng':'numpy.PCG64',
      'horizon_sec':10800.,'allocation_window_sec':3600.,'start_deadline_sec':300.,'decision_interval_sec':30.,
      'arrival_mean_interval_sec_S1_S3':interval,'S4_duration_global_scale':factor if scenario=='S4' else 1.,
      'migration':{'base_sec':20.,'per_gpu_sec':8.,'per_cpu_core_sec':.1,'max_plan_moves':6,'max_total_moves':80,
        'synthetic':True,'destination_reserved_before_source_release':True,'pause_service':True,'serial_engine':True,
        'admission_paused_during_plan':True},
      'duration':{'trace_summary_scale':.01,'base_floor_sec':1800.,'base_ceiling_sec':7200.,'initial_remaining_uniform':[.4,.9],'synthetic':True},
      'semantics':{'single_host_independent_pod':True,'gpu_model':'exact','gpu_cards':'whole, exclusive',
        'queue':'FIFO scan with backfill','completion_before_arrival':True,'preemption':False,'gang':False,
        'initial_layout_is_synthetic':True,'all_pods_assumed_migratable':True,
        'p95':'all requests; unstarted wait lower-bound H-arrival; right-censored',
        'effective_gpu':'integral of computing allocated GPU (excluding paused migration) over [0,3600] / (800*3600); full-horizon metric also saved',
        'deadline_denominator':'all arrivals; all arrive before H-300'},
      'policy_observation':'engine.public_observation allowlist; currently running Pods and arrived queue only; no future or evaluator files',
      'limitations':['No memory/NUMA/network/topology constraints','No gang or application dependencies','No true checkpoint measurements','Single H20 cluster, engineered development suite; not held-out generalization evidence']}

def frozen_audit(servers,initial,arrivals,cfg):
    st=State(servers,initial);st.check();hist=collections.Counter(int(st.free(s)[0]) for s in st.ids)
    probes={};witness=[]
    for g in [1,2,4,8]:
        req=next(r for r in arrivals if r['gpu_request']==g)
        typ=st.classify(req);opts=plans(st,req,cfg,6) if typ=='fragmentation' else []
        probes[str(g)]={'representative_pod_id':req['pod_id'],'class':typ,
          'legal_direct_hosts':sum(st.fits(req,s) for s in st.ids),
          'found_plan_count':len(opts),'min_found_moves':min((len(p['moves']) for p in opts),default=None),
          'min_found_gpu_pause_sec':min((p['gpu_pause_sec'] for p in opts),default=None),
          'max_found_gpu_pause_sec':max((p['gpu_pause_sec'] for p in opts),default=None)}
        if typ=='fragmentation':
            # Independent replay of every distinct min/max-cost witness.
            selected=[opts[0],opts[-1]] if opts else []
            for plan in selected:
                test=State(servers,initial)
                for m in plan['moves']:
                    p=test.pods[m['pod_id']];assert p['server_id']==m['source'] and test.fits(p,m['destination'])
                    p=test.remove(m['pod_id']);test.add(p,m['destination'],p['finish']);test.check()
                assert test.fits(req,plan['target']);test.add(req,plan['target'],req['duration_sec']);test.check()
            witness.append({'gpu':g,'request':req,'plans':selected})
    classes=collections.Counter(st.classify(r) for r in arrivals)
    free_gpu=sum(st.free(s)[0] for s in st.ids)
    max_pod=max(p['gpu_request'] for p in initial)
    lower_bound_8=int(np.ceil((8-max(st.free(s)[0] for s in st.ids))/max_pod))
    return {'servers':len(servers),'initial_pods':len(initial),'arrival_requests':len(arrivals),
      'gpu_capacity':sum(h['gpu_count'] for h in servers),'initial_gpu':sum(p['gpu_request'] for p in initial),
      'initial_cpu':sum(p['cpu_request_cores'] for p in initial),'free_gpu_histogram':dict(sorted(hist.items())),
      'gpu8_unusable_free_fraction':1-sum(int(st.free(s)[0]//8)*8 for s in st.ids)/free_gpu,
      'gpu4_unusable_free_fraction':1-sum(int(st.free(s)[0]//4)*4 for s in st.ids)/free_gpu,
      'frozen_request_classes':dict(classes),'probe_by_size':probes,
      'eight_gpu_min_moves_resource_lower_bound':max(0,lower_bound_8),
      'initial_remaining_sec_quantiles':np.quantile([p['remaining_duration_sec'] for p in initial],[0,.25,.5,.75,1]).tolist(),
      'arrival_gpu_counts':src.counts(r['gpu_request'] for r in arrivals),
      'arrival_gpu_service_work_sec':sum(r['gpu_request']*r['duration_sec'] for r in arrivals),
      'arrival_end_sec':max(r['arrival_time_sec'] for r in arrivals),
      'duration_quantiles_sec':np.quantile([r['duration_sec'] for r in arrivals],[0,.25,.5,.75,1]).tolist(),
      'source_and_initial_legality':True,'witness_replay_passed':True},witness

def evaluate(path):
    servers,initial,arrivals=read_inputs(path);cfg=load(path/'config.json')
    audit,witness=frozen_audit(servers,initial,arrivals,cfg)
    dump(path/'evaluator_only'/'migration_witnesses.json',witness)
    metrics=[]
    for p,m in POLICIES:
        row,events=simulate(servers,initial,arrivals,cfg,p,m);metrics.append(row)
        dump(path/'baselines'/f'{p}__{m}.json',{'metrics':row,'events':events})
    pd.DataFrame(metrics).drop(columns=['first_attempt_classes','blocked_request_seconds']).to_csv(path/'baseline_results.csv',index=False)
    # Noncausal packed reference changes ONLY the initial placement, never feeds policies.
    # S1 is the shared low-fragment reference; used for S2/S3/S4 too.
    reference=path/'evaluator_only'/'reference_initial.parquet'
    if not reference.exists():reference=path.parent/'reference_initial.parquet'
    ref=pd.read_parquet(reference).to_dict('records')
    controls=[]
    for placement in ('first_fit','best_fit'):
        row,_=simulate(servers,ref,arrivals,cfg,placement,'none');controls.append(row)
    dump(path/'evaluator_only'/'low_fragment_controls.json',controls)
    audit['low_fragment_control']=controls
    virtual=dict(servers[0],server_id='RELAXED_POOL',gpu_count=sum(h['gpu_count'] for h in servers),cpu_capacity_cores=sum(h['cpu_capacity_cores'] for h in servers))
    relaxed_initial=[dict(p,server_id='RELAXED_POOL') for p in initial]
    relaxed,_=simulate([virtual],relaxed_initial,arrivals,cfg,'first_fit','none')
    audit['aggregate_capacity_relaxation']=relaxed
    audit['aggregate_capacity_relaxation_note']='Evaluator-only optimistic resource pool; permits splitting across hosts; NOT a legal placement policy or migration witness.'
    audit['all_dynamic_resource_checks']=all(r['all_resource_checks_passed'] for r in metrics)
    dump(path/'validation.json',audit)
    return audit,metrics

def decide(scenario,v,rows):
    # Fixed, direction-neutral differences: no prescribed winning baseline.
    rates=[r['start_within_deadline_rate'] for r in rows];p95=[r['p95_wait_censored_sec'] for r in rows]
    migrations=[r['migration_gpu_pause_sec'] for r in rows if r['migration_count']]
    differences={'deadline_rate_range':max(rates)-min(rates),'p95_wait_range_sec':max(p95)-min(p95),
      'nonzero_migration_cost_range_sec':max(migrations)-min(migrations) if migrations else 0.}
    general=(differences['deadline_rate_range']>=.05 or differences['p95_wait_range_sec']>=120)
    reasons=[]
    if not general:reasons.append('no >=5 percentage point deadline or >=120s P95 difference')
    nomig=[r for r in rows if r['migration']=='none']
    controls=v['low_fragment_control']
    control_gain=max(r['start_within_deadline_rate'] for r in controls)-max(r['start_within_deadline_rate'] for r in nomig)
    differences['low_fragment_control_deadline_gain']=control_gain
    if scenario=='S1':
        if v['gpu8_unusable_free_fraction']>.2:reasons.append('S1 initial unusable-free fraction above 20%')
        ff,bf=nomig
        placement_gap=abs(ff['start_within_deadline_rate']-bf['start_within_deadline_rate'])
        placement_p95=abs(ff['p95_wait_censored_sec']-bf['p95_wait_censored_sec'])
        if placement_gap<.05 and placement_p95<120:reasons.append('no measurable prevention choice between placements')
    elif scenario in ('S2','S4'):
        p=v['probe_by_size']['8']
        if p['class']!='fragmentation' or p['found_plan_count']<2:reasons.append('no multiple legal 8-GPU migration plans')
        if not p['max_found_gpu_pause_sec'] or p['max_found_gpu_pause_sec']-p['min_found_gpu_pause_sec']<20:reasons.append('migration cost choices too similar')
    else:
        p=v['probe_by_size']['8']
        if v['gpu4_unusable_free_fraction']<.9:reasons.append('not high 4-GPU fragmentation')
        if v['eight_gpu_min_moves_resource_lower_bound']<2 or not p['found_plan_count']:reasons.append('multi-step necessity/feasibility not proven')
    if scenario!='S1' and control_gain<.03:
        # A within-placement migration gain is also causal evidence of recoverability.
        gains=[max(r['start_within_deadline_rate'] for r in rows if r['placement']==p)-next(r['start_within_deadline_rate'] for r in nomig if r['placement']==p) for p in ('first_fit','best_fit')]
        if max(gains)<.03:reasons.append('no >=3pp deadline recovery from migration or changed layout')
    return {'passed':not reasons,'reasons':reasons,'differences':differences}

def source_checks(root,hosts,initial,arrivals):
    pool=pd.read_parquet(root/'evaluator_only'/'source_eligible_pods.parquet').set_index('pod_id')
    ss=pd.read_parquet(root/'evaluator_only'/'source_server_snapshot.parquet').set_index('server_id')
    for r in hosts.to_dict('records'):
        for key in ('gpu_count','cpu_capacity_cores','gpu_spec_public','cluster_id'):assert r[key]==ss.loc[r['server_id'],key]
    for frame in (initial,arrivals):
        assert frame.pod_id.is_unique
        for r in frame.to_dict('records'):
            for key in ('gpu_request','cpu_request_cores','gpu_spec_public'):assert r[key]==pool.loc[r['pod_id'],key]
    assert set(initial.pod_id).isdisjoint(arrivals.pod_id)

def build(args):
    out=args.output.resolve()
    if out.exists():raise RuntimeError('Output must be new; existing artifacts are never overwritten')
    out.mkdir(parents=True)
    source=args.source_cache
    shutil.copytree(source,out/'source_audit')
    hosts=pd.read_parquet(source/'evaluator_only'/'source_server_snapshot.parquet')
    pool=pd.read_parquet(source/'evaluator_only'/'source_eligible_pods.parquet')
    summary=pd.read_parquet(source/'evaluator_only'/'source_summary_matches.parquet')
    manifest=[]
    for seed,interval in GRID:
        h,initial,arrivals,s4,factor=workload(hosts,pool,summary,seed,interval)
        group=out/'candidates'/f'seed{seed}_interval{int(interval)}';group.mkdir(parents=True)
        layouts={s:layout(h,initial,s,seed) for s in HIST}
        layouts['S4']=layouts['S2'].copy()
        layouts['S1'].to_parquet(group/'reference_initial.parquet',index=False)
        for scenario in ('S1','S2','S3','S4'):
            path=group/scenario;path.mkdir();a=s4 if scenario=='S4' else arrivals
            source_checks(source,h,layouts[scenario],a)
            h.to_parquet(path/'servers.parquet',index=False);layouts[scenario].to_parquet(path/'initial_pods.parquet',index=False);a.to_parquet(path/'arrivals.parquet',index=False)
            dump(path/'config.json',config(scenario,seed,interval,factor))
            print(f'EVALUATE {scenario} seed={seed} interval={interval}',flush=True)
            v,rows=evaluate(path);decision=decide(scenario,v,rows)
            item={'scenario':scenario,'seed':seed,'interval_sec':interval,'path':str(path.relative_to(out)),**decision}
            manifest.append(item);dump(path/'screening.json',item)
            print(json.dumps(item,ensure_ascii=False),flush=True)
        # Assert all controls rather than rely on build intent.
        assert all(layouts[s].drop(columns='server_id').equals(layouts['S1'].drop(columns='server_id')) for s in ('S2','S3'))
        assert layouts['S4'].equals(layouts['S2'])
        assert abs((arrivals.gpu_request*arrivals.duration_sec).sum()-(s4.gpu_request*s4.duration_sec).sum())<1e-6
    selected={}
    # Prefer earliest fixed seed/pressure that meets every class, never best baseline score.
    groups=[(s,i) for s,i in GRID if all(any(r['scenario']==c and r['seed']==s and r['interval_sec']==i and r['passed'] for r in manifest) for c in ('S1','S2','S3','S4'))]
    choice=groups[0] if groups else None
    if choice:
        for scenario in ('S1','S2','S3','S4'):
            r=next(r for r in manifest if r['scenario']==scenario and (r['seed'],r['interval_sec'])==choice)
            dest=out/f'{scenario}_selected';shutil.copytree(out/r['path'],dest);selected[scenario]=dest.name
            shutil.copyfile((out/r['path']).parent/'reference_initial.parquet',dest/'evaluator_only'/'reference_initial.parquet')
    dump(out/'candidate_registry.json',{'fixed_grid':GRID,'candidates':manifest,'selected':selected,'common_group_required':True,'selection':'earliest predeclared grid group passing all classes; no favored baseline'})
    for name in ('engine.py','build_suite.py','source_adapter.py','verify_suite.py','reproduce.py','finalize.py','test_engine.py'):
        shutil.copyfile(Path(__file__).parent/name,out/name)
    import scipy
    dump(out/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__,'scipy':scipy.__version__})
    from finalize import finalize
    finalize(out)
    print('SELECTED '+json.dumps(selected),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-cache',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);build(ap.parse_args())
