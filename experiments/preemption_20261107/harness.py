"""Explicit experimental policies and event-derived metrics, separate from production code."""
import math
from collections import defaultdict
from vmagent.gpu.environment import GpuSchedEnv, EPS

CORE_POLICIES=('first_fit','best_fit','worst_fit','random')
POLICIES=('legacy_spotgpu','no_preempt','paper_cost','random_victim','paper_baseline','no_history')+CORE_POLICIES

class ExperimentEnv(GpuSchedEnv):
    def __init__(self,*args,variant='paper_cost',**kwargs):
        if variant not in POLICIES: raise ValueError(variant)
        self.variant=variant
        super().__init__(*args,**kwargs)

    def _hp_candidates(self):
        result=[]
        for i,group in enumerate(self._hp_gpu_wait_groups):
            cursor=self._hp_gpu_wait_group_cursors[i]
            while cursor<len(group):
                seq,pid=group[cursor]
                if pid in self.waiting_set and self._queue_seq.get(pid)==seq: break
                cursor+=1
            self._hp_gpu_wait_group_cursors[i]=cursor
            if cursor<len(group): result.append(group[cursor])
        return sorted(result)

    def _next_ready(self):
        if self.variant=='legacy_spotgpu' or self.variant in CORE_POLICIES:
            return super()._next_ready()
        # Explicit experimental queue: earliest feasible HP GPU request first;
        # try its free placement, then preemption, before admitting ordinary tasks.
        # The paper does not specify a complete global queue discipline.
        for _,pid in self._hp_candidates():
            task=self.by_id[pid]
            if self._any_fits(task): return pid
            if self.preempt_enabled:
                cand=self._spotgpu_best_preemption(task)
                if cand:
                    node,victims,_=cand
                    self._evict(victims,node)
                    return pid
        return super()._next_ready()

    def _spotgpu_best_preemption(self,task):
        if self.variant=='legacy_spotgpu': return super()._spotgpu_best_preemption(task)
        plans=[]
        for j,node in enumerate(self.nodes):
            if node.gpu_type not in task.allowed_gpu_types or not self._fits(task,j,empty=True): continue
            if self._fits(task,j): continue
            victims=sorted(self.node_spot_pids[j])
            if self.variant in ('random_victim','paper_baseline'):
                self.rng.shuffle(victims)
            else:
                victims.sort(key=lambda p:(self._spotgpu_victim_cost(p),p))
            chosen=[]; fg=self.free_gpu[j]; fc=self.free_cpu[j]; cost=0.
            for pid in victims:
                chosen.append(pid); t=self.by_id[pid]; fg+=t.gpu; fc+=t.cpu
                cost+=self._spotgpu_victim_cost(pid)
                if fg+EPS>=task.gpu and fc+EPS>=task.cpu:
                    plans.append((j,chosen,cost)); break
        if not plans: return None
        if self.variant in ('random_victim','paper_baseline'):
            # Uniform eligible node + shuffled prefix; explicitly defined because
            # paper's baseline paragraph does not specify its node tie-breaking.
            return self.rng.choice(plans)
        return min(plans,key=lambda x:(x[2],x[0]))

    def _spotgpu_choose_action(self):
        if self.variant not in ('paper_baseline','no_history'): return super()._spotgpu_choose_action()
        task=self.by_id[self.pending_id]
        valid=[j for j in range(len(self.nodes)) if self._fits(task,j)]
        def rank(j):
            colocate=0
            if self.variant=='no_history':
                colocate=int((self.node_hp_running if task.priority=='HP' else self.node_lp_running)[j]==0)
            return self.free_gpu[j]-task.gpu,colocate,j
        return min(valid,key=rank)

    def run_experiment(self):
        if self.variant in CORE_POLICIES:
            self.preempt_enabled=False
            return self.run(self.variant)
        self.preempt_enabled=self.variant!='no_preempt'
        self.reset()
        while not self.done: self.step(self._spotgpu_choose_action())
        return self.summary()

def intersect(a,b,c,d): return max(0.,min(b,d)-max(a,c))
def percentile(xs,q):
    return sorted(xs)[max(0,math.ceil(len(xs)*q)-1)] if xs else None
def average(xs): return sum(xs)/len(xs) if xs else None

def evaluate(env):
    """Reconstruct actual run segments; never use overwritten start_time or flawed run totals."""
    a,b=env.measure_start,env.measure_end; duration=b-a
    segments=defaultdict(list); active={}; first={}; preempt_at=[]
    for e in env.events:
        pid=e['pod_id']; t=e['time_sec']; kind=e['event']
        if kind=='start':
            active[pid]=t; first.setdefault(pid,t)
        elif kind in ('preempt','complete'):
            start=active.pop(pid)
            segments[pid].append((start,t,kind))
            if kind=='preempt': preempt_at.append((pid,t))
    for pid,start in active.items(): segments[pid].append((start,env.current_time,'censored'))
    allocated=waste=0.; execution={}; loss={}; per_model=defaultdict(lambda:[0.,0.])
    per_task=[]
    for task in env.tasks:
        pid=task.pod_id; r=env.records[pid]; run=lost=0.
        for start,end,kind in segments[pid]:
            run+=end-start
            amount=task.gpu*intersect(start,end,a,b); allocated+=amount
            model=task.allowed_gpu_types[0] if task.gpu else 'CPU_ONLY'
            per_model[model][0]+=amount
            if kind=='preempt':
                ck=start+math.floor((end-start)/env.checkpoint_interval)*env.checkpoint_interval
                lost+=end-ck
                amount=task.gpu*intersect(ck,end,a,b); waste+=amount; per_model[model][1]+=amount
        execution[pid]=run; loss[pid]=lost
        finish=r['finish_time_sec']
        stop=finish if finish is not None else env.current_time
        wait=max(0.,stop-task.submit_time-run)
        # At a fixed horizon, include unfinished and not-yet-started tasks.
        run_b=sum(intersect(s,e,0,b) for s,e,_ in segments[pid])
        stop_b=min(b,finish) if finish is not None else b
        wait_b=max(0.,stop_b-task.submit_time-run_b)
        per_task.append({'pod_id':pid,'priority':task.priority,'gpu':task.gpu,
            'submit':task.submit_time,'service':task.service_time,'first_start':first.get(pid),
            'last_start':r['start_time_sec'],'finish':finish,'status':r['status'],
            'response_sec':first[pid]-task.submit_time if pid in first else None,
            'total_queue_sec':wait,'queue_sec_at_cutoff':wait_b,'running_sec':run,
            'wasted_sec':lost,'preemptions':r['preemption_count'],
            'jct_sec':finish-task.submit_time if finish is not None else None,
            'bounded_slowdown':max(1.,(finish-task.submit_time)/max(60.,task.service_time)) if finish is not None else None})
    groups={}
    for priority in ('HP','LP','Other','ALL'):
        cohort=[r for r in per_task if a<=r['submit']<b and (priority=='ALL' or r['priority']==priority)]
        completed=[r for r in cohort if r['finish'] is not None]
        started=[r for r in cohort if r['first_start'] is not None]
        at_b=[r for r in cohort if r['finish'] is not None and r['finish']<=b]
        m={'n':len(cohort),'completed_eventually':len(completed),
           'completed_at_cutoff':len(at_b),'unfinished_at_cutoff':len(cohort)-len(at_b),
           'completion_rate_at_cutoff':len(at_b)/len(cohort) if cohort else None,
           'completion_fraction_eventually':len(completed)/len(cohort) if cohort else None,
           'response_mean_sec_started':average([r['response_sec'] for r in started]),
           'response_p95_sec_started':percentile([r['response_sec'] for r in started],.95),
           'response_p99_sec_started':percentile([r['response_sec'] for r in started],.99),
           'queue_mean_sec':average([r['total_queue_sec'] for r in cohort]),
           'jct_mean_sec_completed':average([r['jct_sec'] for r in completed]),
           'jct_p95_sec_completed':percentile([r['jct_sec'] for r in completed],.95),
           'slowdown_p95_completed':percentile([r['bounded_slowdown'] for r in completed],.95),
           'queue_p95_sec_at_cutoff_all':percentile([r['queue_sec_at_cutoff'] for r in cohort],.95),
           'fraction_queue_over_24h_at_cutoff':sum(r['queue_sec_at_cutoff']>86400 for r in cohort)/len(cohort) if cohort else None}
        for slo in (60,300):
            mature=[r for r in cohort if r['submit']+slo<=b]
            m[f'response_slo_{slo}s_denominator']=len(mature)
            m[f'response_slo_{slo}s_violation_ratio']=sum(r['first_start'] is None or r['first_start']>r['submit']+slo for r in mature)/len(mature) if mature else None
        groups[priority]=m
    lp=[r for r in per_task if r['priority']=='LP' and a<=r['submit']<b]
    max_error=max((abs(r['running_sec']-r['wasted_sec']-r['service']) for r in per_task if r['status']=='completed'),default=0.)
    assert max_error<1e-5, max_error
    assert abs(allocated-env.gpu_area)<max(1e-5,allocated*1e-10)
    assert -1e-6<=waste<=allocated+1e-6
    gden=env.total_gpu*duration
    result={'measurement_window_sec':[a,b],'cohort':'arrivals in [start,end); completed JCT follows cohort to drain',
        'groups':groups,'allocated_gpu_ratio':allocated/gden,
        'wasted_gpu_ratio_window':waste/gden,
        'useful_gpu_ratio_retrospective':(allocated-waste)/gden,
        'useful_ratio_semantics':'allocated intervals minus lost tails, attributed to execution time; losses discovered after cutoff included; not SM utilization',
        'useful_ratio_final':env.end_reason=='drained',
        'lost_fraction_of_allocated':waste/allocated if allocated else 0.,
        'wasted_gpu_hours_window':waste/3600,
        'wasted_gpu_hours_full_run':sum(t.gpu*loss[t.pod_id] for t in env.tasks)/3600,
        'preemption_victims_window':sum(a<=t<b for _,t in preempt_at),
        'lp_preempted_fraction':sum(r['preemptions']>0 for r in lp)/len(lp) if lp else None,
        'lp_preemptions_p95':percentile([r['preemptions'] for r in lp],.95),
        'lp_preemptions_max':max([r['preemptions'] for r in lp],default=0),
        'mean_queue_length':env.queue_area/duration,
        'completed_during_window_per_hour':sum(r['finish'] is not None and a<=r['finish']<b for r in per_task)/(duration/3600),
        'task_makespan_sec':max([r['finish'] for r in per_task if r['finish'] is not None],default=0)-min(t.submit_time for t in env.tasks),
        'completed_work_conservation_max_error_sec':max_error,
        'per_model':{m:{'allocated_gpu_sec':v[0],'lost_gpu_sec':v[1]} for m,v in per_model.items() if m!='CPU_ONLY'}}
    return result,per_task
