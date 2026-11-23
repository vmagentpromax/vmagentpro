"""Deterministic, trace-derived S2 instance builder. Python >=3.10.
Run: python build_s2.py --source E:\\cluster-trace-gpu-v2026\\data --output D:\\一般文件\\benchmark任务\\S2_001
No source files are modified. Outputs must not already exist unless --force.
"""
from __future__ import annotations
import argparse, collections, hashlib, itertools, json, math, platform, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED=20261107
CLUSTER='1f28e063c4e5e1bc48ff55269617dd2f'
MODEL='H20'
TOL=1e-8

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def counts(values):return {str(k):int(v) for k,v in sorted(collections.Counter(values).items(),key=lambda x:str(x[0]))}

def source_path(root,name):
    d=root/('asi_opensource_'+name)/('asi_opensource_'+name)
    return d/'part-000.parquet' if name=='job_execution_summary' else d/'day=0'/'hour=00'/'part-000.parquet'

def stratified_sample(frame,n,rng):
    """Largest remainder allocation across observed job type x priority strata."""
    groups=list(frame.groupby(['job_type_public','priority_class'],dropna=False,sort=True))
    weights=np.array([len(g) for _,g in groups],float)*n/len(frame)
    alloc=np.floor(weights).astype(int)
    for i in sorted(range(len(groups)),key=lambda i:(-(weights[i]-alloc[i]),str(groups[i][0])))[:n-int(alloc.sum())]:alloc[i]+=1
    chosen=[]
    for (_,g),k in zip(groups,alloc):
        if k:chosen.extend(rng.choice(g.index.to_numpy(),size=int(k),replace=False).tolist())
    return frame.loc[chosen].copy()

def load_source(root,out):
    files={n:source_path(root,n) for n in ['server_hourly','pod_hourly','job_execution_summary']}
    servers=pq.read_table(files['server_hourly']).to_pandas()
    pods=pq.read_table(files['pod_hourly']).to_pandas()
    positive=pods[pods.gpu_request.gt(0)].copy()
    integer=lambda x:np.isclose(x,x.round(),rtol=0,atol=TOL)
    scope=positive[positive.cluster_id.eq(CLUSTER)&positive.gpu_spec_public.eq(MODEL)].copy()
    stages={'snapshot_rows':len(pods),'snapshot_positive_gpu_rows':len(positive),
            'snapshot_fractional_positive_rows':int((~integer(positive.gpu_request)).sum()),
            'scope_positive_gpu_rows':len(scope),'scope_fractional_positive_rows':int((~integer(scope.gpu_request)).sum())}
    pool=scope[integer(scope.gpu_request)].copy();stages['after_integer_gpu']=len(pool)
    pool=pool[pool.ready_status.eq(True)&pool.used_gpu_hours.gt(0)].copy();stages['after_ready_and_positive_gpu_activity']=len(pool)
    pool=pool[np.isfinite(pool.cpu_request_cores)&pool.cpu_request_cores.ge(0)&pool.cpu_request_cores.le(192)&pool.gpu_request.le(8)].copy()
    stages['after_valid_cpu_and_single_host_fit']=len(pool)
    joined=pool.merge(servers,on='server_id',suffixes=('','_server'),how='inner',validate='many_to_one')
    ok=(joined.cluster_id.eq(joined.cluster_id_server)&joined.gpu_spec_public.eq(joined.gpu_spec_public_server)&joined.server_gpu_count.eq(joined.gpu_count)&joined.server_cpu_capacity_cores.eq(joined.cpu_capacity_cores))
    valid=set(joined.loc[ok,'pod_id']);pool=pool[pool.pod_id.isin(valid)].copy()
    stages['after_source_server_consistency']=len(pool)
    assert pool.pod_id.is_unique
    ids=pa.array(sorted(pool.pod_id));matches=[];offset=0
    pf=pq.ParquetFile(files['job_execution_summary'])
    for batch in pf.iter_batches(batch_size=262144):
        positions=pc.indices_nonzero(pc.is_in(batch.column('pod_id'),value_set=ids)).to_numpy()
        if len(positions):
            tab=pa.Table.from_batches([batch]).take(pa.array(positions))
            tab=tab.append_column('summary_row_index',pa.array(positions+offset,type=pa.int64()))
            matches.append(tab)
        offset+=batch.num_rows
    summary=pa.concat_tables(matches).to_pandas()
    summary=summary[np.isfinite(summary.duration_hours)&summary.duration_hours.gt(0)].copy()
    expected=pool.set_index('pod_id')
    summary=summary[summary.apply(lambda r:r.gpu_spec_public==expected.loc[r.pod_id,'gpu_spec_public'] and abs(r.gpu_request-expected.loc[r.pod_id,'gpu_request'])<TOL,axis=1)].sort_values('summary_row_index').reset_index(drop=True)
    stages['summary_matching_rows']=len(summary);stages['summary_matching_unique_pods']=summary.pod_id.nunique()
    stages['summary_multiple_records_pods']=int((summary.groupby('pod_id').size()>1).sum())
    pool=pool[pool.gpu_request.isin([1.,2.,4.,8.])].copy().sort_values('pod_id').reset_index(drop=True)
    stages['after_s2_gpu_size_strata']=len(pool)
    stages['snapshot_fractional_exclusion_ratio']=stages['snapshot_fractional_positive_rows']/len(positive)
    stages['scope_fractional_exclusion_ratio']=stages['scope_fractional_positive_rows']/len(scope)
    stages['eligible_gpu_distribution']=counts(pool.gpu_request.astype(int))
    stages['eligible_job_type_distribution']=counts(pool.job_type_public)
    stages['eligible_priority_distribution']=counts(pool.priority_class)
    audit=out/'evaluator_only';audit.mkdir(parents=True,exist_ok=True)
    pool.to_parquet(audit/'source_eligible_pods.parquet',index=False)
    summary.to_parquet(audit/'source_summary_matches.parquet',index=False)
    servers.to_parquet(audit/'source_server_snapshot.parquet',index=False)
    schemas={n:str(pq.ParquetFile(f).schema_arrow) for n,f in files.items()}
    provenance={'source_root':str(root),'snapshot':{'day':0,'hour':'00','time_semantics':'trace-relative hour; not a wall-clock timestamp'},
       'files':{n:{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'rows':pq.ParquetFile(f).metadata.num_rows,'sha256':digest(f)} for n,f in files.items()},
       'schemas':schemas,'screening':stages,
       'libraries':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'pyarrow':pa.__version__}}
    return servers,pool,summary,provenance

def decorate(frame,summary,rng,kind):
    rows=[]
    for _,p in frame.iterrows():
        own=summary[summary.pod_id.eq(p.pod_id)]
        donors=own if len(own) else summary[summary.gpu_request.eq(p.gpu_request)]
        assert len(donors),f'No duration donor for {p.gpu_request}'
        d=donors.iloc[int(rng.integers(len(donors)))]
        dur=float(np.clip(d.duration_hours*3600*.01,1800.,7200.))
        row={'pod_id':p.pod_id,'workload_id':None if pd.isna(p.workload_id) else p.workload_id,
          'source_server_id':p.server_id,'source_cluster_id':p.cluster_id,'source_day':0,'source_hour':0,
          'gpu_spec_public':p.gpu_spec_public,'gpu_request':int(p.gpu_request),'cpu_request_cores':float(p.cpu_request_cores),
          'source_gpu_request':float(p.gpu_request),'gpu_mem_request_trace':int(p.gpu_mem_request),
          'priority_class':p.priority_class,'job_type_public':p.job_type_public,'model_type_public':p.model_type_public,
          'is_genai_request':bool(p.is_genai_request),'source_state_public':p.state_public,'source_ready_status':bool(p.ready_status),
          'source_used_gpu_hours':float(p.used_gpu_hours),'duration_source_pod_id':d.pod_id,
          'duration_summary_row_index':int(d.summary_row_index),'duration_hours_raw':float(d.duration_hours),
          'duration_source_method':'same_pod_record_sample' if len(own) else 'same_model_gpu_size_record_resample',
          'duration_sec':dur,'migratable':True}
        if kind=='initial':row['remaining_duration_sec']=float(rng.uniform(.4,.9)*dur);row['simulation_start_sec']=0.
        rows.append(row)
    return pd.DataFrame(rows)

def construct(servers,pool,summary):
    rng=np.random.Generator(np.random.PCG64(SEED))
    hosts=servers[servers.cluster_id.eq(CLUSTER)&servers.gpu_spec_public.eq(MODEL)&servers.gpu_count.eq(8)&servers.cpu_capacity_cores.eq(192)].sort_values('server_id')
    hosts=hosts.iloc[np.sort(rng.choice(len(hosts),100,replace=False))].copy().reset_index(drop=True)
    hosts['source_day']=0;hosts['source_hour']=0
    arrivals_raw=pd.concat([stratified_sample(pool[pool.gpu_request.eq(g)],n,rng) for g,n in [(1,80),(2,32),(4,24),(8,24)]])
    arrivals_raw=arrivals_raw.iloc[rng.permutation(len(arrivals_raw))].reset_index(drop=True)
    remaining=pool[~pool.pod_id.isin(arrivals_raw.pod_id)].copy()
    bins={g:list(rng.permutation(remaining[remaining.gpu_request.eq(g)].index.to_numpy())) for g in [1,2,4]}
    free=np.array([1]*10+[2]*40+[3]*50);rng.shuffle(free)
    four_hosts=set(rng.choice(100,60,replace=False).tolist())
    selected=[];assigned=[]
    for i,s in hosts.iterrows():
        need=8-int(free[i]);cpu=192.
        pattern=[]
        if i in four_hosts:pattern.append(4);need-=4
        while need:
            g=2 if need>=2 and rng.random()<.60 else 1
            pattern.append(g);need-=g
        for g in pattern:
            available=[k for k in bins[g] if remaining.loc[k,'cpu_request_cores']<=cpu+TOL]
            if not available:raise RuntimeError('CPU-safe initial pool exhausted')
            k=available[0];bins[g].remove(k);cpu-=float(remaining.loc[k,'cpu_request_cores'])
            selected.append(k);assigned.append(s.server_id)
    initial=decorate(remaining.loc[selected].reset_index(drop=True),summary,rng,'initial')
    initial['server_id']=assigned
    initial['initial_state']='running_synthetic'
    arrivals=decorate(arrivals_raw,summary,rng,'arrival')
    arrivals['request_id']=[f'req_{i:04d}' for i in range(len(arrivals))]
    arrivals['arrival_time_sec']=np.cumsum(rng.exponential(35.,len(arrivals)))
    return hosts,initial,arrivals

class State:
    def __init__(self,servers,initial):
        self.servers={r['server_id']:r for r in servers.to_dict('records')}
        self.used={s:[0.,0.] for s in self.servers}
        self.pods={}
        for p in initial.to_dict('records'):self.add(p,p['server_id'],float(p['remaining_duration_sec']))
    def fits(self,p,s):
        h=self.servers[s];u=self.used[s]
        return p['gpu_spec_public']==h['gpu_spec_public'] and u[0]+p['gpu_request']<=h['gpu_count']+TOL and u[1]+p['cpu_request_cores']<=h['cpu_capacity_cores']+TOL
    def add(self,p,s,finish):
        assert p['pod_id'] not in self.pods and self.fits(p,s)
        p=dict(p);p['server_id']=s;p['finish']=finish;self.pods[p['pod_id']]=p
        self.used[s][0]+=p['gpu_request'];self.used[s][1]+=p['cpu_request_cores']
    def remove(self,pid):
        p=self.pods.pop(pid);u=self.used[p['server_id']];u[0]-=p['gpu_request'];u[1]-=p['cpu_request_cores'];return p
    def move(self,pid,dst):
        p=self.pods[pid];assert dst!=p['server_id'] and self.fits(p,dst),'destination must fit before releasing source'
        p=self.remove(pid);self.add(p,dst,p['finish'])
    def direct(self,p):return next((s for s in sorted(self.servers) if self.fits(p,s)),None)
    def classify(self,p):
        if self.direct(p):return 'direct'
        compatible=[s for s,h in self.servers.items() if h['gpu_spec_public']==p['gpu_spec_public']]
        fg=[self.servers[s]['gpu_count']-self.used[s][0] for s in compatible]
        fc=[self.servers[s]['cpu_capacity_cores']-self.used[s][1] for s in compatible]
        if not compatible:return 'model_mismatch'
        if sum(fg)+TOL<p['gpu_request']:return 'gpu_capacity_shortage'
        if sum(fc)+TOL<p['cpu_request_cores']:return 'cpu_capacity_shortage'
        if max(fg)+TOL<p['gpu_request']:return 'gpu_fragment_blocked'
        return 'cpu_or_joint_fragment_blocked'
    def check(self):
        for s,h in self.servers.items():
            ps=[p for p in self.pods.values() if p['server_id']==s]
            assert abs(sum(p['gpu_request'] for p in ps)-self.used[s][0])<TOL
            assert abs(sum(p['cpu_request_cores'] for p in ps)-self.used[s][1])<TOL
            assert -TOL<=self.used[s][0]<=h['gpu_count']+TOL and -TOL<=self.used[s][1]<=h['cpu_capacity_cores']+TOL
            assert all(p['gpu_spec_public']==h['gpu_spec_public'] for p in ps)

def find_plan(state,p,max_moves=6):
    """Bounded constructive search. Failure means unknown, never impossibility."""
    if state.direct(p):return {'target':state.direct(p),'moves':[]}
    byhost={s:[q for q in state.pods.values() if q['server_id']==s and q['migratable']] for s in state.servers}
    hosts=sorted(state.servers,key=lambda s:(-state.servers[s]['gpu_count']+state.used[s][0],len(byhost[s]),s))
    for target in hosts:
        h=state.servers[target]
        if h['gpu_spec_public']!=p['gpu_spec_public']:continue
        fg=h['gpu_count']-state.used[target][0];fc=h['cpu_capacity_cores']-state.used[target][1]
        ps=sorted(byhost[target],key=lambda q:(-q['gpu_request'],q['pod_id']))
        if len(ps)>12:continue
        for k in range(1,min(max_moves,len(ps))+1):
            for subset in itertools.combinations(ps,k):
                if fg+sum(q['gpu_request'] for q in subset)+TOL<p['gpu_request'] or fc+sum(q['cpu_request_cores'] for q in subset)+TOL<p['cpu_request_cores']:continue
                used={s:list(u) for s,u in state.used.items()};moves=[]
                # Greedy destinations, tried in best-fit order. Constructive witnesses only.
                success=True
                for q in subset:
                    dest=[s for s,hs in state.servers.items() if s!=target and hs['gpu_spec_public']==q['gpu_spec_public'] and used[s][0]+q['gpu_request']<=hs['gpu_count']+TOL and used[s][1]+q['cpu_request_cores']<=hs['cpu_capacity_cores']+TOL]
                    if not dest:success=False;break
                    s=min(dest,key=lambda s:(state.servers[s]['gpu_count']-used[s][0]-q['gpu_request'],state.servers[s]['cpu_capacity_cores']-used[s][1]-q['cpu_request_cores'],s))
                    used[s][0]+=q['gpu_request'];used[s][1]+=q['cpu_request_cores']
                    moves.append({'pod_id':q['pod_id'],'source':target,'destination':s})
                if success:return {'target':target,'moves':moves}
    return None

def replay_witness(servers,initial,request,plan):
    st=State(servers,initial)
    assert st.direct(request) is None
    for m in plan['moves']:
        assert st.pods[m['pod_id']]['server_id']==m['source']
        st.move(m['pod_id'],m['destination']);st.check()
    assert st.fits(request,plan['target'])
    st.add(request,plan['target'],request['duration_sec']);st.check()
    return True

def simulate(servers,initial,arrivals,migrate):
    st=State(servers,initial);requests=arrivals.to_dict('records');i=0;t=0.;pending=[];results={};events=[];moves_total=0;gpu_moved=0;first_attempt={};iterations=0
    while i<len(requests) or pending or st.pods:
        iterations+=1;assert iterations<10000
        nxt_a=requests[i]['arrival_time_sec'] if i<len(requests) else math.inf
        nxt_f=min((p['finish'] for p in st.pods.values()),default=math.inf)
        t=min(nxt_a,nxt_f);assert np.isfinite(t)
        for pid in sorted([pid for pid,p in st.pods.items() if p['finish']<=t+TOL]):
            p=st.remove(pid);events.append({'time_sec':t,'event':'complete','pod_id':pid,'server_id':p['server_id']})
        while i<len(requests) and requests[i]['arrival_time_sec']<=t+TOL:pending.append(requests[i]);i+=1
        waiting=[]
        for r in pending:
            rid=r['request_id'];classification=st.classify(r)
            if rid not in first_attempt:first_attempt[rid]=classification
            target=st.direct(r);plan=None
            if target is None and migrate and classification.endswith('fragment_blocked'):
                plan=find_plan(st,r)
                if plan:
                    for m in plan['moves']:
                        gpu_moved+=st.pods[m['pod_id']]['gpu_request'];st.move(m['pod_id'],m['destination']);moves_total+=1
                        events.append({'time_sec':t,'event':'migrate',**m})
                    target=plan['target']
            if target is None:waiting.append(r);continue
            st.add(r,target,t+r['duration_sec'])
            results[rid]={'arrival_time_sec':r['arrival_time_sec'],'start_time_sec':t,'finish_time_sec':t+r['duration_sec'],'wait_sec':t-r['arrival_time_sec'],'server_id':target}
            events.append({'time_sec':t,'event':'start','pod_id':r['pod_id'],'request_id':rid,'server_id':target})
        pending=waiting;st.check()
    waits=[r['wait_sec'] for r in results.values()]
    assert len(results)==len(requests)
    return {'completed_requests':len(results),'makespan_sec':t,'mean_wait_sec':float(np.mean(waits)),
      'p95_wait_sec':float(np.percentile(waits,95)),'max_wait_sec':max(waits),'immediate_requests':sum(w<=TOL for w in waits),
      'first_attempt_classes':counts(first_attempt.values()),'migration_count':moves_total,'gpu_units_moved':gpu_moved,
      'all_event_resource_checks_passed':True,'event_count':len(events)}, {'events':events,'requests':results,'first_attempt_classes':first_attempt}

def validate(servers,initial,arrivals,out,provenance):
    st=State(servers,initial);st.check()
    assert len(servers)==100 and servers.server_id.is_unique and initial.pod_id.is_unique and arrivals.pod_id.is_unique and arrivals.request_id.is_unique
    assert not set(initial.pod_id)&set(arrivals.pod_id)
    assert servers.cluster_id.nunique()==1 and servers.cluster_id.iloc[0]==CLUSTER
    assert servers.gpu_spec_public.eq(MODEL).all()
    assert servers.source_day.eq(0).all() and servers.source_hour.eq(0).all()
    original_servers=pd.read_parquet(out/'evaluator_only'/'source_server_snapshot.parquet').set_index('server_id')
    for h in servers.to_dict('records'):
        src=original_servers.loc[h['server_id']]
        assert h['cluster_id']==src.cluster_id and h['gpu_spec_public']==src.gpu_spec_public
        assert h['gpu_count']==src.gpu_count and h['cpu_capacity_cores']==src.cpu_capacity_cores
    assert (initial.remaining_duration_sec.gt(0)&initial.remaining_duration_sec.le(initial.duration_sec)).all()
    assert arrivals.arrival_time_sec.is_monotonic_increasing and arrivals.arrival_time_sec.gt(0).all() and arrivals.duration_sec.gt(0).all()
    for frame in [initial,arrivals]:
        assert frame.gpu_request.gt(0).all() and np.equal(frame.gpu_request,frame.source_gpu_request).all()
        assert np.isfinite(frame.cpu_request_cores).all() and frame.cpu_request_cores.ge(0).all()
        assert frame.gpu_spec_public.eq(MODEL).all()
        assert frame.source_ready_status.all() and frame.source_used_gpu_hours.gt(0).all()
        assert frame.source_day.eq(0).all() and frame.source_hour.eq(0).all() and frame.source_cluster_id.eq(CLUSTER).all()
        assert np.isfinite(frame.duration_sec).all()
    types=collections.Counter();witnesses=[];unproven=0
    for r in arrivals.to_dict('records'):
        typ=st.classify(r);types[typ]+=1
        if typ.endswith('fragment_blocked'):
            plan=find_plan(st,r)
            if plan:
                replay_witness(servers,initial,r,plan)
                witnesses.append({'request_id':r['request_id'],'snapshot':'independent initial-state counterfactual, not sequential arrivals','plan':plan,'verified':True})
            else:unproven+=1
    free=[int(h['gpu_count']-st.used[s][0]) for s,h in st.servers.items()]
    used_gpu=sum(u[0] for u in st.used.values());cap=servers.gpu_count.sum();rate=sum(v for k,v in types.items() if k.endswith('fragment_blocked'))/len(arrivals)
    s2_checks={'gpu_allocation_between_60_and_80_percent':.6<=used_gpu/cap<=.8,
      'frozen_initial_fragment_block_rate_between_10_and_40_percent':.1<=rate<=.4,
      'direct_requests_present':types['direct']>0,'proven_reorganizable_requests_present':len(witnesses)>0,
      'all_frozen_fragment_blocks_proven':unproven==0,'initial_free_gpu_has_multiple_positive_sizes':len(set(free))>=2}
    dump(out/'evaluator_only'/'initial_state_witnesses.json',witnesses)
    baseline,trace0=simulate(servers,initial,arrivals,False)
    defrag,trace1=simulate(servers,initial,arrivals,True)
    dump(out/'evaluator_only'/'baseline_no_migration.json',trace0)
    dump(out/'evaluator_only'/'baseline_bounded_defrag.json',trace1)
    # Verify requests and durations against frozen source extracts independently of selection code.
    pool=pd.read_parquet(out/'evaluator_only'/'source_eligible_pods.parquet').set_index('pod_id')
    summary=pd.read_parquet(out/'evaluator_only'/'source_summary_matches.parquet').set_index('summary_row_index')
    for f in [initial,arrivals]:
        for r in f.to_dict('records'):
            src=pool.loc[r['pod_id']];d=summary.loc[r['duration_summary_row_index']]
            assert r['gpu_request']==src.gpu_request and abs(r['cpu_request_cores']-src.cpu_request_cores)<TOL
            assert r['gpu_spec_public']==src.gpu_spec_public and r['source_server_id']==src.server_id
            assert r['duration_source_pod_id']==d.pod_id and abs(r['duration_hours_raw']-d.duration_hours)<TOL
            assert d.gpu_spec_public==r['gpu_spec_public'] and d.gpu_request==r['gpu_request']
            assert abs(r['duration_sec']-float(np.clip(d.duration_hours*36,1800,7200)))<TOL
    s2_checks['dynamic_fragment_blocks_present']=sum(v for k,v in baseline['first_attempt_classes'].items() if k.endswith('fragment_blocked'))>0
    s2_checks['dynamic_capacity_shortage_fraction_at_most_half']=sum(v for k,v in baseline['first_attempt_classes'].items() if k.endswith('capacity_shortage'))/len(arrivals)<=.5
    result={'status':'PASS' if all(s2_checks.values()) else 'FAIL','seed':SEED,
      'sample':{'servers':len(servers),'initial_pods':len(initial),'arrival_requests':len(arrivals),'total_gpu_capacity':int(cap),'cluster_id':CLUSTER,'gpu_model':MODEL,'source_day':0,'source_hour':0},
      'hard_checks':{'initial_resource_constraints':True,'gpu_model_compatibility':True,'cpu_capacity_at_each_checked_action':True,'single_cluster_same_snapshot':True,
        'server_capacity_matches_frozen_source_snapshot':True,
        'unique_and_disjoint_source_pod_ids':True,'positive_and_ordered_simulation_times':True,'unchanged_real_gpu_and_cpu_requests':True,
        'all_arrivals_fit_empty_compatible_host':all(any(r['gpu_request']<=h['gpu_count'] and r['cpu_request_cores']<=h['cpu_capacity_cores'] and r['gpu_spec_public']==h['gpu_spec_public'] for h in servers.to_dict('records')) for r in arrivals.to_dict('records')),
        'duration_source_record_and_transform_verified':True,'stepwise_migration_witnesses_verified':True},
      'initial_state':{'allocated_gpu':int(used_gpu),'allocation_ratio':float(used_gpu/cap),'free_gpu_total':sum(free),'free_gpu_histogram':counts(free),
        'cpu_allocation_ratio':float(sum(u[1] for u in st.used.values())/servers.cpu_capacity_cores.sum()),
        'max_server_cpu_fraction':float(max(st.used[s][1]/h['cpu_capacity_cores'] for s,h in st.servers.items())),
        'source_server_changed_pods':int(initial.server_id.ne(initial.source_server_id).sum())},
      'frozen_initial_request_probes':{'semantics':'Each request independently probes t=0; counts are not sequential throughput.',
        'classification_counts':dict(types),'fragment_blocked_fraction':rate,'reorganizable_witness_count':len(witnesses),'unproven_blocked_requests':unproven,
        'migration_steps_histogram':counts(len(w['plan']['moves']) for w in witnesses),'migration_step_limit':6},
      's2_operational_definition_checks':s2_checks,'screening':provenance['screening'],
      'distribution':{'initial_gpu_request':counts(initial.gpu_request),'arrival_gpu_request':counts(arrivals.gpu_request),
        'initial_job_type':counts(initial.job_type_public),'arrival_job_type':counts(arrivals.job_type_public),
        'initial_priority':counts(initial.priority_class),'arrival_priority':counts(arrivals.priority_class),
        'initial_duration_methods':counts(initial.duration_source_method),'arrival_duration_methods':counts(arrivals.duration_source_method),
        'arrival_cpu_cores_quantiles':{str(q):float(arrivals.cpu_request_cores.quantile(q)) for q in [0,.25,.5,.75,1]},
        'arrival_time_end_sec':float(arrivals.arrival_time_sec.max()),
        'initial_remaining_duration_range_sec':[float(initial.remaining_duration_sec.min()),float(initial.remaining_duration_sec.max())],
        'arrival_duration_range_sec':[float(arrivals.duration_sec.min()),float(arrivals.duration_sec.max())]},
      'diagnostic_simulation':{'semantics':'FIFO with backfill; first-fit hosts sorted by server_id; complete before arrivals at ties; ideal instantaneous migration; no measured real-system performance.',
         'no_migration':baseline,'bounded_defrag':defrag},
      'not_verified':['Physical GPU memory capacity (no server field)','GPU interconnect, NUMA, host RAM, network bandwidth','Actual checkpointability and migration latency','Exact real submission/start/end timestamps','Actual snapshot running lifecycle because state_public=Unknown','Joint multi-pod workload or gang semantics'],
      'notes':['S2 thresholds are an explicit operational definition for this instance, not an Alibaba trace label.','Bounded constructive search does not prove optimality or impossibility.','Scenario is a development fixture, not an independent held-out test set.']}
    dump(out/'validation.json',result)
    assert result['status']=='PASS',json.dumps(s2_checks)
    assert all(result['hard_checks'].values())
    return result

def write_readme(out,v):
    screen=v['screening'];probe=v['frozen_initial_request_probes'];dist=v['distribution']
    text=f'''# GPU 碎片整理 Benchmark S2_001

这是基于本地真实资源需求构造的、可复现的受控重放置实例。原始需求未扩卡、取整或改写。它不是原始集群的完整运行快照，也不是生产迁移性能测量。

## 实际规模与验证

- 来源：E:\\cluster-trace-gpu-v2026\\data；day=0/hour=00，同一 cluster_id={CLUSTER}。
- {v['sample']['servers']} 台真实 H20 服务器，每台 8 GPU / 192 CPU 核，总计 800 GPU。
- 初始 {v['sample']['initial_pods']} 个 Pod，分配 {v['initial_state']['allocated_gpu']} GPU（{v['initial_state']['allocation_ratio']:.1%}）。空闲 GPU 分布：{json.dumps(v['initial_state']['free_gpu_histogram'],ensure_ascii=False)}（键为每台空闲卡数，值为服务器数）。
- {v['sample']['arrival_requests']} 个互不重复且与初始集合互斥的真实 Pod 请求。GPU 请求分布：{json.dumps(dist['arrival_gpu_request'])}。
- 在同一个初始状态上独立探测每条请求：{json.dumps(probe['classification_counts'])}；碎片阻塞 {probe['fragment_blocked_fraction']:.1%}，其中 {probe['reorganizable_witness_count']} 条有逐步检查 CPU/GPU 和目标机容量的合法迁移见证。
- 验证状态：{v['status']}。详细统计、检查布尔值和两套动态诊断结果见 validation.json。所有统计均为脚本实际计算。

## 文件与接口

评测策略输入仅包括 servers.parquet、initial_pods.parquet、arrivals.parquet、config.json。不要将 evaluator_only 目录、validation.json 中的诊断结果或复现代码中的搜索逻辑作为策略提示、训练答案或策略输入。

- servers.parquet：server_id、cluster_id、gpu_spec_public、gpu_count、cpu_capacity_cores、source_day/hour；asw_id 原值全空，不用于拓扑约束。
- initial_pods.parquet：pod_id、server_id 为新布局，source_server_id 保留真实来源；gpu_request 为整数 GPU 卡数，cpu_request_cores 为真实核数；remaining_duration_sec 是仿真 t=0 的剩余运行时间，duration_sec 为合成总服务时间；初始状态固定为 running_synthetic。
- arrivals.parquet：request_id、pod_id、arrival_time_sec、duration_sec、真实 GPU/CPU/型号；请求按时间排序。启动后 finish=start+duration_sec，等待不消耗服务时间。
- 两类 Pod 都携带来源时间、来源状态、任务类型、优先级、真实 GPU 使用量、时长 donor 的 pod_id 和 summary 行号等审计字段。workload_id 可空；pod_id 是唯一请求实体。
- gpu_mem_request_trace 原样保存，仅审计；服务器没有可用的显存容量字段，因此此版不实施显存约束，不能声称已验证显存。
- provenance.json：真实源文件 schema、大小、SHA-256、筛选漏斗、依赖版本。evaluator_only/source_* 是读取源文件后冻结的必要证据摘录。
- evaluator_only/initial_state_witnesses.json：仅供验证者使用的逐请求迁移见证。两份 baseline_*.json 为完整动态事件日志；它们与独立 t=0 探测分开统计。

## 数据筛选和代表性

1. 从服务器真实快照选择同一集群、同一时刻的 100 台 H20 八卡机，按 server_id 排序后用固定种子无放回抽样。Pod 来源池覆盖该集群该型号，来源服务器也必须在同一快照出现且容量字段一致；Pod 不限于最终抽取的 100 台原始宿主机。
2. pod_hourly 的 state_public 本小时全为 Unknown，因此不能恢复精确的运行列表。采用 ready_status=true 且 used_gpu_hours>0 的活动证据选取需求，受控创建仿真初始运行状态；这不是断言它们在某个真实瞬间都在运行。
3. 首版独占整数 GPU：不对小数请求向上取整，也不把分数混入卡数。全快照正 GPU 需求 {screen['snapshot_positive_gpu_rows']} 条，分数 {screen['snapshot_fractional_positive_rows']} 条，筛除比例 {screen['snapshot_fractional_exclusion_ratio']:.6%}；选中集群 H20 正需求 {screen['scope_positive_gpu_rows']} 条，分数 {screen['scope_fractional_positive_rows']} 条，筛除比例 {screen['scope_fractional_exclusion_ratio']:.6%}。分母均为正 GPU 需求，不含 CPU-only Pod。
4. 整数、活动、合法 CPU、单机可容纳、源机信息一致的漏斗见 provenance.json。为建立 1/2/4/8 卡场景，另外剔除 5/6 卡罕见需求；CPU-only 工作负载不参与本实例，也不作为隐含 CPU 背景。
5. 到达按 GPU 大小固定分层为 1卡80、2卡32、4卡24、8卡24；层内按 pod_hourly 的 job_type_public × priority_class 的观测频率做最大余数分配并无放回抽样，最后随机排序。它覆盖多种实际需求，但有意增加 2/4 卡占比，不能当作原始流量的无偏重放。来源池 GPU 分布为 {json.dumps(screen['eligible_gpu_distribution'])}；输出任务类型与优先级分布见 validation.json。
6. 初始 Pod 从扣除到达集合后的同一池选取，先按真实 CPU/GPU 需求合法放置，生成 10 台余1卡、40台余2卡、50台余3卡的布局。60台含一个4卡Pod，其余按真实1/2卡需求填充。仅保留成功合法放置的样本；这种选择会改变初始任务分布，完整分布已报告。

## 时长、时间与迁移假设

job_execution_summary 没有 cpu_request_cores、cluster_id、精确提交/起止时间；CPU 取自 pod_hourly，执行时长仅从 summary.duration_hours 取。summary 按 pod_id 连接并核对 GPU 型号和需求。本次候选活动 Pod 中仅 {screen['summary_matching_unique_pods']} 个有匹配摘要；匹配 {screen['summary_matching_rows']} 行，{screen['summary_multiple_records_pods']} 个 Pod 有多行，无法确定哪一行对应本次小时活动。严禁把这些摘要当作精确生命周期。

对每个选中 Pod：若有自己的匹配摘要，固定种子在其记录中采样一条；否则在相同 H20 型号、相同 GPU 卡数的匹配摘要中按记录重采样（多记录 Pod 权重较大）。保存 donor 的 pod_id、summary_row_index 和 duration_hours_raw。两种情况都只代表 trace-derived 合成服务时长；duration_sec=clip(duration_hours_raw×3600×0.01,1800,7200)。这个缩放与截断会改变时长分布，不用来推断真实作业性能。

初始剩余时间=duration_sec×U(0.4,0.9)，完全合成，不从 used_gpu_hours 推算已运行时间。到达间隔为均值35秒的指数分布累加，固定 PCG64 种子 {SEED}；arrival_time_sec 从仿真 t=0 起算，不是实际提交时间。源数据 day/hour 是 trace 相对时间，未映射到日历时区。到达结束约 {dist['arrival_time_end_sec']:.2f} 秒，初始残余和服务时长的实际范围见 validation.json。

每个 Pod 是可独立调度的单机单实体，不恢复 gang、多 Pod 依赖或工作负载原子性；优先级与任务类型只保留元数据。GPU 型号严格相等，禁止跨型号迁移，不假设 GPU 卡间互联、NUMA、主机内存或网络带宽。所有 Pod 的 migratable=true 是人为假设，不代表真实应用支持 checkpoint。

迁移采用理想化的零耗时原子重放置，目标机必须在源机释放前具有足够的空闲 CPU/GPU，禁止抢占、丢弃 Pod 或降低需求。保持剩余服务时间不变，资源逐步合法；代价只记迁移 Pod 次数和 GPU 单位，不将它解读为真实秒数。初始独立探测最多6次搬迁。真实 checkpoint、传输双占用时长、恢复开销和迁移失效风险不在本版模型内。搜索成功给出构造性证明，搜索失败只记未证明，不声称不可整理或最优。

## S2 判定与动态诊断

本实例显式把“中度碎片”定义为：初始 GPU 分配率60%–80%，独立 t=0 需求探测中10%–40%因碎片受阻，同时存在直接可放置请求，受阻请求有至多6步合法整理见证，空闲容量有多个尺寸。阈值是本实例设计，不是数据集官方标签。

独立 t=0 探测会为每条请求重置状态，不是按到达顺序重复消耗资源的仿真。仅当同型号总空闲 GPU 和 CPU 均足够且单机无法满足请求时才记碎片阻塞；资源总量不足独立统计。动态诊断真正处理到达、等待、完成、资源释放与（可选）迁移，不使用冻结状态结果代替动态指标。

两个诊断器都是 FIFO 遍历等待队列并允许后续任务回填，服务器按 ID first-fit；同刻先完成再到达。整理版仅在碎片阻塞时调用同一个有界构造搜索；它是合法性与生命周期闭环检查，不是算法排行榜或正式性能结论。当前只是一个开发实例，不是训练/验证/测试划分或统计充分的 Benchmark 套件。

## 复现与独立复核

建议使用 requirements.txt 中本次运行的版本，Python 版本见 provenance.json。

```powershell
python -m pip install -r requirements.txt
python build_s2.py --source 'E:\\cluster-trace-gpu-v2026\\data' --output 'D:\\一般文件\\benchmark任务\\S2_001_reproduced'
python verify_s2.py --instance 'D:\\一般文件\\benchmark任务\\S2_001_reproduced'
```

build_s2.py 默认拒绝覆盖已有输出。--force 只允许重写本脚本列出的同名文件，不递归清空目录。复现需要三个源 Parquet 的内容不变（SHA-256 可核对）和相同依赖版本。Parquet 字节级比较仅对同版本编码器有保证；逻辑内容、随机种子与 schema 也可单独核对。

不访问源数据也可运行 verify_s2.py，它会从已落盘的三个输入重新执行全部检查，并通过冻结源摘录交叉核对需求和 donor。它还独立重放已保存的迁移见证、对比重算结果与 validation.json，再核对 manifest.json 中的文件哈希；不把旧 PASS 当成新检查结果。
'''
    (out/'README.md').write_text(text,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--force',action='store_true');args=ap.parse_args()
    out=args.output
    if out.exists() and any(out.iterdir()) and not args.force:raise SystemExit('Output exists; select a new directory or explicitly use --force')
    out.mkdir(parents=True,exist_ok=True)
    print('Reading source schema, snapshot and summary...',flush=True)
    servers,pool,summary,provenance=load_source(args.source,out)
    print('Building controlled placement and arrival strata...',flush=True)
    servers,initial,arrivals=construct(servers,pool,summary)
    for name,frame in [('servers',servers),('initial_pods',initial),('arrivals',arrivals)]:frame.to_parquet(out/(name+'.parquet'),index=False)
    dump(out/'provenance.json',provenance)
    config={'instance_id':'S2_001','schema_version':1,'scenario':'S2_moderate_fragmentation','seed':SEED,'rng':'numpy.PCG64',
       'source_snapshot':{'day':0,'hour':0,'cluster_id':CLUSTER},'simulation_time_unit':'seconds','cpu_unit':'cores','gpu_unit':'exclusive whole devices',
       'source_gpu_fraction_handling':'exclude fractional positive requests; no rounding','gpu_model_matching':'exact',
       'arrival_generation':{'kind':'exponential_interarrival','mean_interval_sec':35,'gpu_strata':{'1':80,'2':32,'4':24,'8':24},'within_gpu_stratum':'job_type_public x priority_class proportional largest remainder'},
       'duration_generation':{'summary_scale':.01,'floor_sec':1800,'ceiling_sec':7200,'initial_remaining_fraction_uniform':[.4,.9],'exact_real_lifetimes_reconstructed':False},
       'migration':{'allowed':True,'assumed_all_pods_migratable':True,'duration_sec':0,'destination_must_fit_before_source_release':True,'preserve_remaining_service':True,'cost_metrics':['pod_moves','gpu_units_moved'],'offline_witness_max_moves':6},
       'scheduler_semantics':{'unit':'one independent single-host Pod','gang_scheduling':False,'cpu_only_background':False,'preemption':False,'completion_before_arrival_on_ties':True},
       's2_thresholds':{'initial_gpu_allocation_ratio':[.6,.8],'frozen_initial_fragment_block_fraction':[.1,.4],'witness_required':True},
       'policy_input_files':['servers.parquet','initial_pods.parquet','arrivals.parquet','config.json'],
       'evaluator_only_files':['validation.json','provenance.json','evaluator_only/'],'limitations':['no GPU memory capacity constraint','no topology or bandwidth constraints','synthetic arrivals and service times','idealized zero-latency migration','not a held-out test suite']}
    dump(out/'config.json',config)
    # Read the written Parquet files back before validating.
    print('Validating written inputs, constructive witnesses and dynamic baselines...',flush=True)
    v=validate(pd.read_parquet(out/'servers.parquet'),pd.read_parquet(out/'initial_pods.parquet'),pd.read_parquet(out/'arrivals.parquet'),out,provenance)
    write_readme(out,v)
    script=Path(__file__).resolve()
    if script!= (out/'build_s2.py').resolve():shutil.copyfile(script,out/'build_s2.py')
    verifier=script.with_name('verify_s2.py')
    if verifier.exists() and verifier.resolve()!=(out/'verify_s2.py').resolve():shutil.copyfile(verifier,out/'verify_s2.py')
    (out/'requirements.txt').write_text(f'numpy=={np.__version__}\npandas=={pd.__version__}\npyarrow=={pa.__version__}\n',encoding='utf-8')
    manifest={str(p.relative_to(out)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['manifest.json','verification.json'] and '__pycache__' not in p.parts}
    dump(out/'manifest.json',manifest)
    print(json.dumps({'output':str(out),'status':v['status'],'sample':v['sample'],'initial_state':v['initial_state'],'probes':v['frozen_initial_request_probes'],'diagnostics':v['diagnostic_simulation']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
