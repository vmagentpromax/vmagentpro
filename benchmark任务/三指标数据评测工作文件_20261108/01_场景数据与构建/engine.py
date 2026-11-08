"""Online baselines. No filesystem access, no future-arrival or evaluator input."""
from __future__ import annotations
import itertools, math, copy, collections
import numpy as np

EPS=1e-7
POLICIES=[(p,m) for p in ('first_fit','best_fit') for m in ('none','one_move','cheap_multi','first_multi')]

def move_seconds(p,cfg):
    m=cfg['migration']; return m['base_sec']+m['per_gpu_sec']*p['gpu_request']+m['per_cpu_core_sec']*p['cpu_request_cores']

class State:
    def __init__(self,servers,initial):
        self.hosts={h['server_id']:dict(h) for h in servers}; self.ids=sorted(self.hosts)
        self.used={s:[0.,0.] for s in self.ids}; self.pods={}; self.reserved=None
        for p in initial: self.add(p,p['server_id'],p['remaining_duration_sec'])
    def fits(self,p,s):
        h=self.hosts[s]; u=self.used[s]
        return p['gpu_spec_public']==h['gpu_spec_public'] and u[0]+p['gpu_request']<=h['gpu_count']+EPS and u[1]+p['cpu_request_cores']<=h['cpu_capacity_cores']+EPS
    def add(self,p,s,finish):
        assert p['pod_id'] not in self.pods and self.fits(p,s)
        q=dict(p,server_id=s,finish=finish); self.pods[q['pod_id']]=q
        self.used[s][0]+=q['gpu_request']; self.used[s][1]+=q['cpu_request_cores']
    def remove(self,pid):
        p=self.pods.pop(pid); s=p['server_id']; self.used[s][0]-=p['gpu_request'];self.used[s][1]-=p['cpu_request_cores'];return p
    def free(self,s):
        h=self.hosts[s];u=self.used[s];return h['gpu_count']-u[0],h['cpu_capacity_cores']-u[1]
    def classify(self,p):
        compatible=[s for s in self.ids if self.hosts[s]['gpu_spec_public']==p['gpu_spec_public']]
        if not compatible:return 'model_unavailable'
        if any(self.fits(p,s) for s in compatible):return 'direct'
        if sum(self.free(s)[0] for s in compatible)+EPS<p['gpu_request']:return 'gpu_capacity_shortage'
        if sum(self.free(s)[1] for s in compatible)+EPS<p['cpu_request_cores']:return 'cpu_capacity_shortage'
        return 'fragmentation'
    def check(self):
        actual={s:[0.,0.] for s in self.ids}
        for p in self.pods.values():
            s=p['server_id'];assert self.hosts[s]['gpu_spec_public']==p['gpu_spec_public']
            actual[s][0]+=p['gpu_request'];actual[s][1]+=p['cpu_request_cores']
        if self.reserved:
            p,s=self.reserved;actual[s][0]+=p['gpu_request'];actual[s][1]+=p['cpu_request_cores']
        for s in self.ids:
            assert max(abs(a-b) for a,b in zip(actual[s],self.used[s]))<EPS
            h=self.hosts[s];u=self.used[s]
            assert -EPS<=u[0]<=h['gpu_count']+EPS and -EPS<=u[1]<=h['cpu_capacity_cores']+EPS

def choose_host(st,p,placement):
    fit=[s for s in st.ids if st.fits(p,s)]
    if not fit:return None
    if placement=='first_fit':return fit[0]
    return min(fit,key=lambda s:(st.free(s)[0]-p['gpu_request'],st.free(s)[1]-p['cpu_request_cores'],s))

def plans(st,req,cfg,max_moves=6,first_only=False):
    """Online target-evacuation heuristic, computed afresh using current resources.
    Backtracking over source subsets; deterministic best-fit destination assignment.
    Failure does NOT prove infeasibility. No frozen witness is used.
    """
    byhost=collections.defaultdict(list)
    for p in st.pods.values():
        if p.get('migratable',False):byhost[p['server_id']].append(p)
    candidates=[]
    for target in st.ids:
        if st.hosts[target]['gpu_spec_public']!=req['gpu_spec_public']:continue
        fg,fc=st.free(target);ps=sorted(byhost[target],key=lambda p:(-p['gpu_request'],p['pod_id']))
        if len(ps)>12:continue
        for n in range(1,min(max_moves,len(ps))+1):
            for subset in itertools.combinations(ps,n):
                if fg+sum(p['gpu_request'] for p in subset)+EPS<req['gpu_request'] or fc+sum(p['cpu_request_cores'] for p in subset)+EPS<req['cpu_request_cores']:continue
                free={s:list(st.free(s)) for s in st.ids};moves=[]
                for p in subset:
                    dest=[s for s in st.ids if s!=target and st.hosts[s]['gpu_spec_public']==p['gpu_spec_public'] and free[s][0]+EPS>=p['gpu_request'] and free[s][1]+EPS>=p['cpu_request_cores']]
                    if not dest:break
                    s=min(dest,key=lambda s:(free[s][0]-p['gpu_request'],free[s][1]-p['cpu_request_cores'],s))
                    free[s][0]-=p['gpu_request'];free[s][1]-=p['cpu_request_cores']
                    free[target][0]+=p['gpu_request'];free[target][1]+=p['cpu_request_cores']
                    moves.append({'pod_id':p['pod_id'],'source':target,'destination':s,'pause_sec':move_seconds(p,cfg),'gpu':p['gpu_request']})
                else:
                    cost=sum(m['pause_sec']*m['gpu'] for m in moves)
                    candidates.append({'target':target,'moves':moves,'gpu_pause_sec':cost})
                    if first_only:return candidates
    return sorted(candidates,key=lambda p:(p['gpu_pause_sec'],len(p['moves']),p['target'],str(p['moves'])))

def public_observation(st,queue,t):
    # Explicit allowlist: no service finish times, donor records, future requests,
    # source metadata, evaluator outputs or precomputed plans cross this boundary.
    keys=('pod_id','server_id','gpu_request','cpu_request_cores','gpu_spec_public','migratable')
    obs=State(list(st.hosts.values()),[])
    obs.used=copy.deepcopy(st.used)
    obs.pods={pid:{k:p[k] for k in keys if k in p} for pid,p in st.pods.items()}
    visible=[{k:p[k] for k in ('pod_id','request_id','gpu_request','cpu_request_cores','gpu_spec_public','arrival_time_sec')} for p in queue]
    return obs,visible

def simulate(servers,initial,arrivals,cfg,placement,migration):
    st=State(servers,initial); st.check(); horizon=cfg['horizon_sec'];dtick=cfg['decision_interval_sec']
    future=sorted(arrivals,key=lambda r:(r['arrival_time_sec'],r['request_id']))
    queue=[];i=0;t=0.;next_tick=0.;move_end=math.inf;current=None;operation=None
    logs=[];starts={};work=0.;capacity_area=0.;window_work=0.;window_occupied=0.;gpu_pause=0.;gpu_pause_accrued=0.;move_count=0;completed_moves=0;gpu_moved=0;first_classes={}
    block_seconds=collections.Counter();total_moves_budget=cfg['migration']['max_total_moves']
    def event(kind,**kwargs):logs.append(dict(time_sec=t,event=kind,**kwargs))
    def begin_next():
        nonlocal current,move_end,move_count,gpu_moved,gpu_pause
        while operation['moves']:
            m=operation['moves'].pop(0);pid=m['pod_id']
            if pid not in st.pods:
                event('planned_move_skipped_completed',pod_id=pid);continue
            p=st.pods[pid];s=m['destination']
            assert p['server_id']==m['source'] and st.fits(p,s) and st.reserved is None
            pause=move_seconds(p,cfg);p['finish']+=pause
            st.used[s][0]+=p['gpu_request'];st.used[s][1]+=p['cpu_request_cores'];st.reserved=(p,s)
            current=dict(m,pause_sec=pause);move_end=t+pause
            move_count+=1;gpu_moved+=p['gpu_request'];gpu_pause+=p['gpu_request']*pause
            event('migration_start',**current);st.check();return
        current=None;move_end=math.inf
    while t<horizon-EPS:
        # Completion precedes migration completion, arrivals and admission on ties.
        migrating=current['pod_id'] if current else None
        for pid in sorted(list(st.pods)):
            p=st.pods[pid]
            if pid!=migrating and p['finish']<=t+EPS:
                st.remove(pid);event('completion',pod_id=pid)
        if current and move_end<=t+EPS:
            m=current;p=st.remove(m['pod_id']);s=m['destination']
            st.used[s][0]-=p['gpu_request'];st.used[s][1]-=p['cpu_request_cores'];st.reserved=None
            st.add(p,s,p['finish']);completed_moves+=1
            event('migration_complete',**m);current=None;move_end=math.inf;st.check();begin_next()
        if operation and current is None:
            event('plan_complete',request_id=operation['request_id']);operation=None
        while i<len(future) and future[i]['arrival_time_sec']<=t+EPS:
            q=future[i];queue.append(q);event('arrival',request_id=q['request_id']);i+=1
        if t>=next_tick-EPS:next_tick=(math.floor((t+EPS)/dtick)+1)*dtick
        if operation is None:
            for q in list(queue):
                rid=q['request_id'];first_classes.setdefault(rid,st.classify(q))
                s=choose_host(st,q,placement)
                if s is None:continue
                st.add(q,s,t+q['duration_sec']);queue.remove(q);starts[rid]=t
                event('start',request_id=rid,pod_id=q['pod_id'],server_id=s,wait_sec=t-q['arrival_time_sec']);st.check()
            if queue and migration!='none' and move_count<total_moves_budget:
                obs,visible=public_observation(st,queue,t)
                for q in visible:
                    if obs.classify(q)!='fragmentation':continue
                    lim=1 if migration=='one_move' else cfg['migration']['max_plan_moves']
                    options=plans(obs,q,cfg,min(lim,total_moves_budget-move_count),first_only=migration=='first_multi')
                    if not options:continue
                    plan=options[0];operation=dict(request_id=q['request_id'],moves=list(plan['moves']))
                    event('plan_start',request_id=q['request_id'],target=plan['target'],planned_moves=plan['moves'],gpu_pause_sec=plan['gpu_pause_sec'])
                    begin_next();break
        times=[horizon,next_tick,move_end]
        if i<len(future):times.append(future[i]['arrival_time_sec'])
        times.extend(p['finish'] for pid,p in st.pods.items() if current is None or pid!=current['pod_id'])
        nt=min(x for x in times if x>t+EPS);elapsed=nt-t
        running=sum(p['gpu_request'] for pid,p in st.pods.items() if current is None or pid!=current['pod_id'])
        work+=elapsed*running;capacity_area+=elapsed*sum(u[0] for u in st.used.values())
        if current:gpu_pause_accrued+=elapsed*st.pods[current['pod_id']]['gpu_request']
        window_dt=max(0.,min(nt,cfg['allocation_window_sec'])-min(t,cfg['allocation_window_sec']))
        window_work+=window_dt*running;window_occupied+=window_dt*sum(u[0] for u in st.used.values())
        for q in queue:block_seconds[st.classify(q)]+=elapsed
        t=nt
    st.check()
    waits=[starts.get(q['request_id'],horizon)-q['arrival_time_sec'] for q in future]
    completed=sum(1 for q in future if q['pod_id'] not in st.pods and q['request_id'] in starts)
    # All arrivals finish before H-deadline, so late unstarted jobs count as misses.
    assert all(q['arrival_time_sec']<=horizon-cfg['start_deadline_sec'] for q in future)
    metrics={'placement':placement,'migration':migration,'requests':len(future),'started':len(starts),'completed':completed,
      'unstarted':len(future)-len(starts),'p95_wait_censored_sec':float(np.quantile(waits,.95)),
      'mean_wait_censored_sec':float(np.mean(waits)),
      'start_within_deadline_rate':sum(q['request_id'] in starts and starts[q['request_id']]-q['arrival_time_sec']<=cfg['start_deadline_sec']+EPS for q in future)/len(future),
      'gpu_effective_allocation_rate':window_work/(cfg['allocation_window_sec']*sum(h['gpu_count'] for h in servers)),
      'gpu_reserved_allocation_rate':window_occupied/(cfg['allocation_window_sec']*sum(h['gpu_count'] for h in servers)),
      'gpu_effective_allocation_full_horizon_rate':work/(horizon*sum(h['gpu_count'] for h in servers)),
      'migration_count':move_count,'migration_completed_count':completed_moves,'gpu_units_moved':gpu_moved,'migration_gpu_pause_sec':gpu_pause,
      'migration_gpu_pause_accrued_sec':gpu_pause_accrued,
      'migration_inflight_at_horizon':current is not None,'first_attempt_classes':dict(collections.Counter(first_classes.values())),
      'blocked_request_seconds':dict(block_seconds),'all_resource_checks_passed':True}
    return metrics,logs
