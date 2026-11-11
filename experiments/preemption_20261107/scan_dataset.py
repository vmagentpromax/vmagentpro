"""Audit whole-dataset coverage and raw daily snapshots; no raw data modifications.
The 185 midnight snapshots are explicitly a screen, not a scan of all pod rows.
Existing 4440-hour aggregate coverage is checked and reused as supporting evidence.
"""
import csv,json,time
from pathlib import Path
import duckdb

ROOT=Path('D:/cluster-trace-gpu-v2026')
OUT=Path('D:/vmagent_preemption_dataset_audit_20261107')
OUT.mkdir(exist_ok=True)
C=duckdb.connect(); C.execute("SET threads=4"); C.execute("SET memory_limit='2GB'")

def save_query(path,query,params=None):
    cur=C.execute(query,params or []); rows=cur.fetchall()
    with (OUT/path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow([x[0] for x in cur.description]); w.writerows(rows)
    return rows

tables={t:len(list((ROOT/t).rglob('*.parquet'))) for t in
        ['asi_opensource_pod_hourly','asi_opensource_server_hourly','asi_opensource_network_hourly','asi_opensource_job_execution_summary']}
(OUT/'coverage.json').write_text(json.dumps(tables,indent=2),encoding='utf-8')
save_query('job_summary_full.csv',"""SELECT gpu_spec_public,priority_class,job_type_public,
 count(*) records, sum(gpu_request*duration_hours) gpu_hours,
 median(duration_hours) median_hours, quantile_cont(duration_hours,.95) p95_hours,
 count(*) FILTER(WHERE duration_hours BETWEEN .5 AND 24) templates_half_to_24h,
 count(*) FILTER(WHERE gpu_request>1) multi_gpu_records
 FROM read_parquet(?) GROUP BY ALL ORDER BY gpu_hours DESC""",
 [str(ROOT/'asi_opensource_job_execution_summary/part-000.parquet')])
print('Full job summary complete',flush=True)
path=OUT/'daily_midnight_screen.csv'
done=set()
if path.exists():
    with path.open(encoding='utf-8-sig') as f: done={int(r['day']) for r in csv.DictReader(f)}
with path.open('a',encoding='utf-8',newline='') as f:
    w=csv.writer(f)
    if not done: w.writerow(['day','cluster_id','gpu_spec_public','capacity_gpu','capacity_cpu','hp_gpu_h','lp_gpu_h','other_gpu_h','hp_request','lp_request','cpu_request','gpu_rows','lp_rows'])
    for d in range(185):
        if d in done: continue
        pod=str(ROOT/f'asi_opensource_pod_hourly/day={d}/hour=00/part-000.parquet')
        node=str(ROOT/f'asi_opensource_server_hourly/day={d}/hour=00/part-000.parquet')
        rows=C.execute("""WITH n AS (SELECT cluster_id,gpu_spec_public,sum(gpu_count) capacity_gpu,
        sum(cpu_capacity_cores) capacity_cpu FROM read_parquet(?) GROUP BY ALL),
        p AS (SELECT cluster_id,gpu_spec_public,
        sum(CASE WHEN priority_class='HP' THEN used_gpu_hours ELSE 0 END) hp,
        sum(CASE WHEN priority_class='LP' THEN used_gpu_hours ELSE 0 END) lp,
        sum(CASE WHEN priority_class NOT IN ('HP','LP') OR priority_class IS NULL THEN used_gpu_hours ELSE 0 END) other,
        sum(CASE WHEN priority_class='HP' THEN gpu_request ELSE 0 END) hpr,
        sum(CASE WHEN priority_class='LP' THEN gpu_request ELSE 0 END) lpr,
        sum(cpu_request_cores) cr,count(*) ng,count(*) FILTER(WHERE priority_class='LP') nl
        FROM read_parquet(?) WHERE gpu_request>0 GROUP BY ALL)
        SELECT ?,n.*,p.hp,p.lp,p.other,p.hpr,p.lpr,p.cr,p.ng,p.nl FROM n JOIN p USING(cluster_id,gpu_spec_public)
        ORDER BY cluster_id,gpu_spec_public""",[node,pod,d]).fetchall()
        w.writerows(rows); f.flush()
        if d%10==0: print('screen day',d,flush=True)
save_query('candidate_15day_windows.csv',"""WITH x AS (
 SELECT cluster_id,gpu_spec_public,day AS end_day,
 count(*) OVER w AS n_days, avg(capacity_gpu) OVER w AS capacity_gpu,
 avg(hp_gpu_h/capacity_gpu) OVER w hp_ratio,avg(lp_gpu_h/capacity_gpu) OVER w lp_ratio,
 min((hp_gpu_h+lp_gpu_h+other_gpu_h)/capacity_gpu) OVER w min_total_ratio,
 avg((hp_gpu_h+lp_gpu_h+other_gpu_h)/capacity_gpu) OVER w total_ratio
 FROM read_csv_auto(?) WINDOW w AS(PARTITION BY cluster_id,gpu_spec_public ORDER BY day RANGE BETWEEN 14 PRECEDING AND CURRENT ROW))
 SELECT *,least(hp_ratio,lp_ratio)*total_ratio score FROM x
 WHERE n_days=15 AND capacity_gpu>=64 AND hp_ratio>.1 AND lp_ratio>.1
 ORDER BY score DESC""",[str(path)])
print('screen complete',flush=True)
