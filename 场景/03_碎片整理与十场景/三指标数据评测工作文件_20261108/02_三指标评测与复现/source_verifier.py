"""Independent source, event-replay, control-variable and checksum verifier.
Does not use engine.State, planner or simulate to validate saved trajectories.
"""
from __future__ import annotations
import argparse,collections,json,hashlib,math,sys
from pathlib import Path
import numpy as np
import pandas as pd

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def near(a,b):assert abs(a-b)<=1e-6*max(1,abs(b)),(a,b)

def replay(path,initial,servers,arrivals,cfg):
    result=load(path);metrics=result['metrics'];events=result['events']
    hs={r['server_id']:r for r in servers};active={p['pod_id']:dict(p,finish=p['remaining_duration_sec']) for p in initial}
    req={r['request_id']:r for r in arrivals};seen=set();starts={};reserved=None;current=None;previous=0.
    moves=0;completed_moves=0;gpu_moves=0;cost=0.;accrued=0.;work=0.;occupied_area=0.;fullwork=0.;count_actions=0
    def check():
        used={s:[0.,0.] for s in hs}
        for p in active.values():
            s=p['server_id'];assert s in hs and p['gpu_spec_public']==hs[s]['gpu_spec_public']
            used[s][0]+=p['gpu_request'];used[s][1]+=p['cpu_request_cores']
        if reserved:
            p=active[reserved[0]];s=reserved[1];used[s][0]+=p['gpu_request'];used[s][1]+=p['cpu_request_cores']
        for s,(g,c) in used.items():assert -1e-6<=g<=hs[s]['gpu_count']+1e-6 and -1e-6<=c<=hs[s]['cpu_capacity_cores']+1e-6,(s,g,c)
    def advance(t):
        nonlocal previous,work,occupied_area,fullwork,accrued
        assert t>=previous-1e-7 and t<=cfg['horizon_sec']+1e-7
        running=sum(p['gpu_request'] for pid,p in active.items() if current is None or pid!=current['pod_id'])
        occupied=sum(p['gpu_request'] for p in active.values())+(active[reserved[0]]['gpu_request'] if reserved else 0)
        dt=max(0.,min(t,cfg['allocation_window_sec'])-min(previous,cfg['allocation_window_sec']))
        work+=dt*running;occupied_area+=dt*occupied;fullwork+=(t-previous)*running
        if current:accrued+=(t-previous)*active[current['pod_id']]['gpu_request']
        previous=t
    check()
    for e in events:
        t=e['time_sec'];advance(t);kind=e['event'];count_actions+=1
        if kind=='arrival':
            rid=e['request_id'];assert rid not in seen;near(t,req[rid]['arrival_time_sec']);seen.add(rid)
        elif kind=='completion':
            p=active[e['pod_id']];assert current is None or current['pod_id']!=e['pod_id'];near(t,p['finish']);del active[e['pod_id']]
        elif kind=='start':
            rid=e['request_id'];p=req[rid];assert rid in seen and rid not in starts and p['pod_id'] not in active and current is None
            near(t-p['arrival_time_sec'],e['wait_sec']);starts[rid]=t;active[p['pod_id']]=dict(p,server_id=e['server_id'],finish=t+p['duration_sec'])
        elif kind=='migration_start':
            assert current is None and reserved is None
            p=active[e['pod_id']];assert p['migratable'] and p['server_id']==e['source'] and e['source']!=e['destination']
            assert hs[e['destination']]['gpu_spec_public']==p['gpu_spec_public']
            m=cfg['migration'];pause=m['base_sec']+m['per_gpu_sec']*p['gpu_request']+m['per_cpu_core_sec']*p['cpu_request_cores'];near(pause,e['pause_sec'])
            p['finish']+=pause;reserved=(p['pod_id'],e['destination']);current=dict(e,start=t)
            moves+=1;gpu_moves+=p['gpu_request'];cost+=pause*p['gpu_request']
        elif kind=='migration_complete':
            assert current and current['pod_id']==e['pod_id'];near(t,current['start']+e['pause_sec'])
            p=active[e['pod_id']];p['server_id']=e['destination'];reserved=None;current=None
            completed_moves+=1
        elif kind=='plan_start':
            assert e['request_id'] in seen and e['request_id'] not in starts
            assert len(e['planned_moves'])<=cfg['migration']['max_plan_moves']
        elif kind in ('plan_complete','planned_move_skipped_completed'):pass
        else:raise AssertionError(kind)
        check()
    advance(cfg['horizon_sec']);check()
    waits=[starts.get(p['request_id'],cfg['horizon_sec'])-p['arrival_time_sec'] for p in arrivals]
    near(metrics['p95_wait_censored_sec'],float(np.quantile(waits,.95)));near(metrics['mean_wait_censored_sec'],float(np.mean(waits)))
    near(metrics['start_within_deadline_rate'],sum(p['request_id'] in starts and starts[p['request_id']]-p['arrival_time_sec']<=cfg['start_deadline_sec']+1e-7 for p in arrivals)/len(arrivals))
    cap=sum(h['gpu_count'] for h in servers)
    near(metrics['gpu_effective_allocation_rate'],work/(cfg['allocation_window_sec']*cap));near(metrics['gpu_reserved_allocation_rate'],occupied_area/(cfg['allocation_window_sec']*cap))
    near(metrics['gpu_effective_allocation_full_horizon_rate'],fullwork/(cfg['horizon_sec']*cap))
    assert metrics['migration_count']==moves and moves<=cfg['migration']['max_total_moves']
    assert metrics['gpu_units_moved']==gpu_moves;near(metrics['migration_gpu_pause_sec'],cost)
    assert metrics['migration_completed_count']==completed_moves;near(metrics['migration_gpu_pause_accrued_sec'],accrued)
    assert metrics['started']==len(starts) and metrics['unstarted']==len(arrivals)-len(starts)
    assert metrics['completed']==sum(p['request_id'] in starts and p['pod_id'] not in active for p in arrivals)
    return {'events_replayed':count_actions,'moves':moves,'all_constraints_and_metrics_match':True}

def validate_instance(root,path):
    servers,initial,arrivals=[pd.read_parquet(path/f) for f in ('servers.parquet','initial_pods.parquet','arrivals.parquet')]
    cfg=load(path/'config.json')
    pool=pd.read_parquet(root/'source_audit'/'evaluator_only'/'source_eligible_pods.parquet').set_index('pod_id')
    source_servers=pd.read_parquet(root/'source_audit'/'evaluator_only'/'source_server_snapshot.parquet').set_index('server_id')
    summary=pd.read_parquet(root/'source_audit'/'evaluator_only'/'source_summary_matches.parquet').set_index('summary_row_index')
    assert len(servers)==100 and servers.server_id.is_unique
    for h in servers.to_dict('records'):
        for key in ('gpu_count','cpu_capacity_cores','gpu_spec_public','cluster_id'):assert h[key]==source_servers.loc[h['server_id'],key]
    for frame in (initial,arrivals):
        assert frame.pod_id.is_unique
        for p in frame.to_dict('records'):
            q=pool.loc[p['pod_id']]
            for key in ('gpu_request','cpu_request_cores','gpu_spec_public'):assert p[key]==q[key]
            assert p['source_server_id']==q.server_id and float(p['gpu_request']).is_integer()
            d=summary.loc[p['duration_summary_row_index']]
            assert d.pod_id==p['duration_source_pod_id'] and d.gpu_spec_public==p['gpu_spec_public'] and d.gpu_request==p['gpu_request']
            near(d.duration_hours,p['duration_hours_raw']);base=float(np.clip(d.duration_hours*36,1800,7200));near(base,p['duration_base_sec'])
            near(p['duration_sec'],base*(cfg['S4_duration_global_scale'] if frame is arrivals else 1.))
    assert set(initial.pod_id).isdisjoint(arrivals.pod_id)
    assert arrivals.request_id.is_unique and arrivals.arrival_time_sec.is_monotonic_increasing
    assert arrivals.arrival_time_sec.ge(0).all() and arrivals.arrival_time_sec.le(cfg['horizon_sec']-cfg['start_deadline_sec']).all()
    assert initial.remaining_duration_sec.gt(0).all() and initial.remaining_duration_sec.le(initial.duration_sec).all()
    reports={}
    for file in sorted((path/'baselines').glob('*.json')):
        reports[file.name]=replay(file,initial.to_dict('records'),servers.to_dict('records'),arrivals.to_dict('records'),cfg)
    # Check saved evaluator witnesses independently, including pre-release destination fit.
    for item in load(path/'evaluator_only'/'migration_witnesses.json'):
        for plan in item['plans']:
            state=initial.set_index('pod_id').to_dict('index');hs=servers.set_index('server_id').to_dict('index')
            def fits(p,s):
                others=[x for x in state.values() if x['server_id']==s]
                return p['gpu_spec_public']==hs[s]['gpu_spec_public'] and sum(x['gpu_request'] for x in others)+p['gpu_request']<=hs[s]['gpu_count']+1e-7 and sum(x['cpu_request_cores'] for x in others)+p['cpu_request_cores']<=hs[s]['cpu_capacity_cores']+1e-7
            for m in plan['moves']:
                p=state[m['pod_id']];assert p['server_id']==m['source'] and fits(p,m['destination']);p['server_id']=m['destination']
            assert fits(item['request'],plan['target'])
    return {'source_requests_and_capacities_unchanged':True,'lifecycle_transform_verified':True,'witnesses_replayed':True,'baselines':reports}

def verify(root,instance=None,check_hashes=True,raw_source=None):
    registry=load(root/'candidate_registry.json');reports={}
    if check_hashes:
        manifest=load(root/'manifest.json')
        for rel,h in manifest['sha256'].items():assert digest(root/rel)==h,rel
    if raw_source:
        prov=load(root/'source_audit'/'provenance.json')
        for data in prov['files'].values():assert digest(raw_source/data['path'])==data['sha256']
    paths=[instance] if instance else [root/r['path'] for r in registry['candidates']]+[root/x for x in registry['selected'].values()]
    for path in paths:
        reports[str(path.relative_to(root))]=validate_instance(root,path)
        print('VERIFIED '+str(path.relative_to(root)),flush=True)
    # Cross-class controlled variables, independently from construction.
    for parent in sorted((root/'candidates').iterdir()):
        if not parent.is_dir():continue
        sets={s:pd.read_parquet(parent/s/'initial_pods.parquet') for s in ('S1','S2','S3','S4')}
        for s in ('S2','S3'):pd.testing.assert_frame_equal(sets['S1'].drop(columns='server_id'),sets[s].drop(columns='server_id'))
        pd.testing.assert_frame_equal(sets['S2'],sets['S4'])
        base=pd.read_parquet(parent/'S1'/'arrivals.parquet')
        for s in ('S2','S3'):pd.testing.assert_frame_equal(base,pd.read_parquet(parent/s/'arrivals.parquet'))
        for s in ('S2','S3','S4'):pd.testing.assert_frame_equal(pd.read_parquet(parent/'S1'/'servers.parquet'),pd.read_parquet(parent/s/'servers.parquet'))
        s4=pd.read_parquet(parent/'S4'/'arrivals.parquet');near((base.gpu_request*base.duration_sec).sum(),(s4.gpu_request*s4.duration_sec).sum())
    return {'status':'PASS','instance_count':len(paths),'source_hashes_checked':raw_source is not None,'file_hashes_checked':check_hashes,'controlled_variables_passed':True,'instances':reports}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).parent);ap.add_argument('--instance',type=Path);ap.add_argument('--raw-source',type=Path);ap.add_argument('--skip-hashes',action='store_true');ap.add_argument('--report',type=Path)
    a=ap.parse_args();r=verify(a.root.resolve(),a.instance.resolve() if a.instance else None,not a.skip_hashes,a.raw_source)
    if a.report:a.report.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k!='instances'}),flush=True)
