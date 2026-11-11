import argparse,csv,json,hashlib,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vmagent.gpu.environment import load_nodes,load_workload
from harness import ExperimentEnv,evaluate,POLICIES,CORE_POLICIES

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('D:/vmagent_scenarios_preemption_20261107'))
    p.add_argument('--scenarios',nargs='+');p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--checkpoints',nargs='+',type=float,default=[3600.])
    p.add_argument('--full-export',action='store_true');args=p.parse_args()
    manifest=json.loads((args.root/'manifest.json').read_text(encoding='utf-8'))
    summary_rows=[]
    for name in args.scenarios or manifest['scenarios']:
        meta=manifest['scenarios'][name]; tasks,_=load_workload(args.root/name/'tasks.csv')
        nodes_path=args.root/meta.get('nodes_file','nodes.csv')
        nodes=load_nodes(nodes_path)
        for ck in args.checkpoints:
            for seed in args.seeds:
                for policy in args.policies:
                    out=args.root/'outputs'/f'{name}-{policy}-s{seed}-ck{ck:g}'
                    if out.exists(): raise SystemExit(f'Refusing to overwrite {out}')
                    env=ExperimentEnv(nodes,tasks,variant=policy,clock='fast',queue_policy='backfill',
                        measurement_window=meta['measurement_window_sec'],max_sim_time=meta['max_sim_time_sec'],
                        seed=seed,checkpoint_interval=ck)
                    s=env.run_experiment(); extra,detail=evaluate(env);out.mkdir(parents=True)
                    if args.full_export:env.export(out)
                    else:(out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
                    (out/'metrics.json').write_text(json.dumps(extra,indent=2),encoding='utf-8')
                    with (out/'task_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
                    receipt={'scenario':name,'policy':policy,'seed':seed,'checkpoint_interval_sec':ck,
                        'queue':'original backfill' if policy=='legacy_spotgpu' or policy in CORE_POLICIES else 'HP GPU first, otherwise backfill',
                        'task_sha256':hashlib.sha256((args.root/name/'tasks.csv').read_bytes()).hexdigest(),
                        'node_sha256':hashlib.sha256(nodes_path.read_bytes()).hexdigest(),
                        'python':sys.version,'core_sha256':hashlib.sha256((Path(__file__).resolve().parents[2]/'vmagent/gpu/environment.py').read_bytes()).hexdigest(),
                        'harness_sha256':hashlib.sha256(Path(__file__).with_name('harness.py').read_bytes()).hexdigest(),
                        'queue_policy':'backfill','clock':'fast','cpu_background_fraction':0,
                        'measurement_window_sec':meta['measurement_window_sec'],'max_sim_time_sec':meta['max_sim_time_sec']}
                    (out/'run_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
                    row={'scenario':name,'policy':policy,'seed':seed,'checkpoint':ck,'end_reason':s['end_reason'],
                        'completed':s['completed'],'unfinished':s['unfinished'],'rejected':s['rejected'],
                        'allocated':extra['allocated_gpu_ratio'],'useful':extra['useful_gpu_ratio_retrospective'],
                        'lost_gpu_h':extra['wasted_gpu_hours_window'],'preemptions':s['preemptions'],
                        'hp_response_p95_sec':extra['groups']['HP']['response_p95_sec_started'],
                        'lp_jct_mean_sec':extra['groups']['LP']['jct_mean_sec_completed'],
                        'lp_completed_at_cutoff':extra['groups']['LP']['completion_rate_at_cutoff'],
                        'wall_sec':s['wall_time_sec']}
                    summary_rows.append(row);print(json.dumps(row),flush=True)
    dest=args.root/'outputs'/'comparison_latest.csv'
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)

if __name__=='__main__':main()
