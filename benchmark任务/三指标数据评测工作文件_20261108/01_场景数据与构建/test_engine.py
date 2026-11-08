"""Small adversarial regression checks for migration and censoring semantics."""
import copy
from engine import simulate,State,public_observation
from build_suite import config

def run():
    hosts=[dict(server_id=s,gpu_count=8,cpu_capacity_cores=32,gpu_spec_public='H20') for s in ('a','b')]
    initial=[dict(pod_id='p'+s,server_id=s,gpu_request=4,cpu_request_cores=4,gpu_spec_public='H20',migratable=True,remaining_duration_sec=1000.) for s in ('a','b')]
    arrival=[dict(pod_id='r',request_id='r',gpu_request=8,cpu_request_cores=8,gpu_spec_public='H20',migratable=True,arrival_time_sec=0.,duration_sec=100.)]
    cfg=config('TEST',1,10.,1.);cfg.update(horizon_sec=200.,allocation_window_sec=200.,start_deadline_sec=10.)
    row,events=simulate(hosts,initial,arrival,cfg,'first_fit','one_move')
    assert row['migration_count']==row['migration_completed_count']==1
    assert abs(next(e['time_sec'] for e in events if e['event']=='start')-52.4)<1e-8
    assert abs(row['gpu_effective_allocation_rate']-2190.4/3200)<1e-8
    assert abs(row['migration_gpu_pause_sec']-209.6)<1e-8
    cfg.update(horizon_sec=20.,allocation_window_sec=20.)
    row,_=simulate(hosts,initial,arrival,cfg,'first_fit','one_move')
    assert row['migration_count']==1 and row['migration_completed_count']==0 and row['migration_inflight_at_horizon']
    assert abs(row['migration_gpu_pause_sec']-209.6)<1e-8 and abs(row['migration_gpu_pause_accrued_sec']-80)<1e-8
    assert row['unstarted']==1 and row['p95_wait_censored_sec']==20. and row['start_within_deadline_rate']==0.
    obs,visible=public_observation(State(hosts,initial),arrival,0)
    assert all('finish' not in p and 'remaining_duration_sec' not in p for p in obs.pods.values())
    assert 'duration_sec' not in visible[0]
    # A model mismatch cannot be bypassed by capacity or migration.
    wrong=copy.deepcopy(arrival);wrong[0]['gpu_spec_public']='A100'
    row,_=simulate(hosts,initial,wrong,cfg,'first_fit','cheap_multi')
    assert row['started']==0 and row['migration_count']==0
    print('PASS: pause, double reservation, in-flight accounting, censoring, observation allowlist, GPU model checks')

if __name__=='__main__':run()
