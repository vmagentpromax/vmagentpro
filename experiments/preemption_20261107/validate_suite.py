import hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from vmagent.gpu.environment import load_nodes,load_workload
from build_scenarios import DEFAULT_OUT,read,SOURCE,make_p0_strict,P0_WARMUP_SEC

root=DEFAULT_OUT;m=json.loads((root/'manifest.json').read_text(encoding='utf-8'));result={}
for name,meta in m['scenarios'].items():
    path=root/name/'tasks.csv';nodes=load_nodes(root/meta.get('nodes_file','nodes.csv'));tasks,_=load_workload(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
    assert len(tasks)==meta['tasks'] and len({t.pod_id for t in tasks})==len(tasks)
    assert all(t.submit_time<meta['arrival_horizon_sec'] and any((t.gpu==0 or n.gpu_type in t.allowed_gpu_types)
           and t.gpu<=n.gpu_capacity and t.cpu<=n.cpu_capacity for n in nodes) for t in tasks)
    if (root/name/'tasks.json').exists():
        jt,jm=load_workload(root/name/'tasks.json');assert jt==tasks and jm['measurement_window_sec']==meta['measurement_window_sec']
    result[name]={'loaded_tasks':len(tasks),'nodes':len(nodes),'checksum_verified':True,'all_empty_node_feasible':True}
    print(name,len(tasks),'valid',flush=True)
original_f=read(SOURCE/'jobs_F.csv')
p0=read(root/'P0_real_F8d/tasks.csv')
assert p0==make_p0_strict(original_f)
assert m['scenarios']['P0_real_F8d']['measurement_window_sec'][0]==P0_WARMUP_SEC
assert sum(float(r['submit_time_sec'])==0 for r in p0)==sum(float(r['submit_time_sec'])==0 for r in original_f)
def identity(r,arrival=True):
    cols=['source_pod_id','gpu_request','cpu_request_cores','service_sec','priority_class','gpu_spec_public']
    if arrival:cols+=['submit_time_sec']
    return tuple(r.get(k,'') for k in cols)
p1=read(root/'P1_steady/tasks.csv');p2=read(root/'P2_burst/tasks.csv');p5=read(root/'P5_cpu_contention/tasks.csv')
assert Counter(identity(r,False) for r in p1)==Counter(identity(r,False) for r in p2)
assert Counter(identity(r) for r in p2)==Counter(identity(r) for r in p5 if float(r['gpu_request'])>0)
p7=read(root/'P7_real_H20_15d/tasks.csv');p9=read(root/'P9_H20_gpu_capacity_pressure/tasks.csv')
assert [r for r in p7 if float(r['gpu_request'])>0]==p9
result['paired_inputs']={'P1_P2_same_service_work':True,'P2_P5_identical_gpu_arrivals':True,'P7_P9_identical_gpu_rows':True,'P0_exact_transform_of_F_with_warmup':True}
(root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('all validations passed',flush=True)
