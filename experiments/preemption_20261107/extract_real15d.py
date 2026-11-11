"""Read all 360 hours of an independently screened real window, with raw provenance."""
import argparse,csv,hashlib,json,random,sys
from pathlib import Path
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from build_scenarios import audit,read,write,sha,DAY,DEFAULT_OUT

ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['early_h20','late_h20','xpu_d'],default='late_h20');args=ap.parse_args()
PROFILES={'early_h20':('9089d80baea5f56a3c169801052daa1f','H20',48,'P6_real_H20_early15d'),
 'late_h20':('9089d80baea5f56a3c169801052daa1f','H20',158,'P7_real_H20_15d'),
 'xpu_d':('2f9611355f71fe804da2606899ddfb36','XPU-D',132,'P8_real_XPUD_15d')}
CID,MODEL,START,NAME=PROFILES[args.profile];END=START+15
ROOT=Path('D:/cluster-trace-gpu-v2026'); CACHE=Path('D:/vmagent_preemption_dataset_audit_20261107')/('real15d' if args.profile=='early_h20' else NAME)
CACHE.mkdir(exist_ok=True)
C=duckdb.connect();C.execute('SET threads=4');C.execute("SET memory_limit='2GB'")
# 30 nodes, stable inventory and capacity on every day of the selected window.
snapshots=[str(ROOT/f'asi_opensource_server_hourly/day={d}/hour=00/part-000.parquet') for d in range(START,END)]
C.execute("""CREATE TEMP TABLE stable AS SELECT server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores
 FROM read_parquet(?,union_by_name=true) WHERE cluster_id=? AND gpu_spec_public=? AND gpu_count=8
 GROUP BY ALL HAVING count(DISTINCT day)=15""",[snapshots,CID,MODEL])
first=str(ROOT/f'asi_opensource_pod_hourly/day={START}/hour=00/part-000.parquet')
C.execute("""CREATE TEMP TABLE selected AS WITH p AS (SELECT server_id,
 sum(CASE WHEN priority_class='HP' THEN used_gpu_hours ELSE 0 END) hp,
 sum(CASE WHEN priority_class='LP' THEN used_gpu_hours ELSE 0 END) lp
 FROM read_parquet(?) WHERE cluster_id=? AND gpu_spec_public=? AND gpu_request>0 GROUP BY server_id)
 SELECT s.* FROM stable s JOIN p USING(server_id) ORDER BY least(hp,lp) DESC,server_id LIMIT 30""",[first,CID,MODEL])
cur=C.execute('SELECT * FROM selected ORDER BY server_id'); nodes=[dict(zip([d[0] for d in cur.description],r)) for r in cur.fetchall()]
assert len(nodes)==30
write(CACHE/'nodes.csv',nodes)
files=[]
for hour_index in range(-1,360):
    day=START+hour_index//24;hour=hour_index%24
    dest=CACHE/f'h{hour_index+1:03d}.parquet';files.append(str(dest))
    if dest.exists():continue
    source=ROOT/f'asi_opensource_pod_hourly/day={day}/hour={hour:02d}/part-000.parquet'
    C.execute("""COPY (SELECT p.pod_id,p.server_id,p.cluster_id,p.gpu_spec_public,p.gpu_request,
     p.cpu_request_cores,p.priority_class,p.job_type_public,p.used_gpu_hours,p.avg_gpu_sm_util,
     ? AS relative_hour FROM read_parquet(?) p JOIN selected n USING(server_id)
     WHERE p.cluster_id=? AND (p.gpu_request>0 OR p.cpu_request_cores>0))
     TO ? (FORMAT PARQUET)""",[hour_index,str(source),CID,str(dest)])
    if hour_index%24==0:print('read raw day',day,flush=True)
C.execute('CREATE TEMP TABLE observations AS SELECT * FROM read_parquet(?)',[files])
C.execute("""CREATE TEMP TABLE shape AS WITH counted AS (SELECT pod_id,gpu_spec_public,gpu_request,cpu_request_cores,
 priority_class,job_type_public,count(*) n FROM observations GROUP BY ALL
 ) SELECT * FROM counted
 QUALIFY row_number() OVER(PARTITION BY pod_id ORDER BY n DESC,gpu_spec_public,gpu_request,cpu_request_cores,priority_class,job_type_public)=1""")
C.execute("""CREATE TEMP TABLE spans AS SELECT pod_id,min(relative_hour) first_h,max(relative_hour) last_h,
 count(DISTINCT relative_hour) seen_hours,avg(avg_gpu_sm_util) observed_sm_mean FROM observations GROUP BY pod_id""")
C.execute("""CREATE TEMP TABLE durations AS SELECT j.pod_id,median(j.duration_hours)*3600 service_sec,
 count(*) matched_execution_rows FROM read_parquet(?) j JOIN shape s USING(pod_id)
 WHERE j.gpu_spec_public=s.gpu_spec_public AND j.gpu_request=s.gpu_request AND j.duration_hours>0
 GROUP BY j.pod_id""",[str(ROOT/'asi_opensource_job_execution_summary/part-000.parquet')])
cur=C.execute('SELECT s.*,p.first_h,p.last_h,p.seen_hours,p.observed_sm_mean,d.service_sec,d.matched_execution_rows FROM shape s JOIN spans p USING(pod_id) LEFT JOIN durations d USING(pod_id) ORDER BY pod_id')
records=[dict(zip([d[0] for d in cur.description],r)) for r in cur.fetchall()]
tasks=[];excluded=0
for r in records:
    if r['last_h']<0:continue
    if not any((r['gpu_request']==0 or r['gpu_spec_public']==n['gpu_spec_public']) and r['gpu_request']<=n['gpu_count'] and r['cpu_request_cores']<=n['cpu_capacity_cores'] for n in nodes):excluded+=1;continue
    warm=r['first_h']<0
    seed=int(hashlib.sha256((r['pod_id']+'20261107').encode()).hexdigest()[:16],16)
    arrival=0 if warm else (r['first_h']+random.Random(seed).random())*3600
    if warm:
        service=(r['last_h']+1)*3600; match='presence_remaining_proxy'
    elif START>=109 and r['service_sec'] is not None:
        service=r['service_sec'];match='pod_model_request_matched_median_execution_span'
    else:
        service=(r['last_h']-r['first_h']+1)*3600;match='presence_span_proxy'
    tasks.append({'pod_id':r['pod_id'],'cluster_id':CID,'gpu_spec_public':r['gpu_spec_public'] if r['gpu_request'] else '[]',
         'gpu_request':r['gpu_request'],'cpu_request_cores':r['cpu_request_cores'],'priority_class':r['priority_class'] or 'Other',
         'job_type_public':r['job_type_public'] or 'unknown','submit_time_sec':arrival,'service_sec':service,
         'duration_match_status':match,'source_pod_id':r['pod_id'],'first_relative_hour':r['first_h'],
         'last_relative_hour':r['last_h'],'seen_hours':r['seen_hours'],
         'right_boundary_observed':r['last_h']==359,'observed_sm_mean_not_used_by_simulator':r['observed_sm_mean'],
         'matched_execution_rows':r['matched_execution_rows'] or 0})
tasks.sort(key=lambda r:(r['submit_time_sec'],r['pod_id']))
out=DEFAULT_OUT/NAME;out.mkdir(exist_ok=False)
write(out/'tasks.csv',tasks);write(out/'nodes.csv',nodes)
window=[DAY,15*DAY]
(out/'tasks.json').write_text(json.dumps({'measurement_window_sec':window,'tasks':tasks},ensure_ascii=False),encoding='utf-8')
meta={'kind':'continuous real-window reconstruction; no task duplication or changed demand tuples',
 'nodes_file':NAME+'/nodes.csv','measurement_window_sec':window,'arrival_horizon_sec':15*DAY,
 'max_sim_time_sec':90*DAY,'sha256':sha(out/'tasks.csv'),'source_days':[START,END-1],
 'warmup_source_hour':f'day{START-1} hour23','raw_partitions_read':361,'excluded_empty_node_infeasible':excluded,
 'node_selection':f'stable all15days; rank min(HP,LP) observed GPU-hours at day{START} hour00; top30',
 'execution_summary_used':START>=109,'early_window_duration_warning':START<109,
 'caution':'hourly observation, synthetic within-hour timestamps; warmup/presence spans and execution medians are proxies',
 **audit(tasks,nodes,window)}
manifest=json.loads((DEFAULT_OUT/'manifest.json').read_text(encoding='utf-8'))
manifest['scenarios'][NAME]=meta
(DEFAULT_OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
(CACHE/'construction_audit.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=True),flush=True)
