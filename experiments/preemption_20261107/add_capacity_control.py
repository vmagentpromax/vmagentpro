"""Preserve late H20 GPU tasks; change pool capacity as an explicit controlled factor."""
import json
from build_scenarios import DEFAULT_OUT,read,write,audit,sha,DAY
root=DEFAULT_OUT;source=root/'P7_real_H20_15d';name='P9_H20_gpu_capacity_pressure'
rows=[r for r in read(source/'tasks.csv') if float(r['gpu_request'])>0]
nodes=read(source/'nodes.csv')[:19]
out=root/name;out.mkdir(exist_ok=False);write(out/'tasks.csv',rows);write(out/'nodes.csv',nodes)
window=[DAY,15*DAY]
(out/'tasks.json').write_text(json.dumps({'measurement_window_sec':window,'tasks':rows},ensure_ascii=False),encoding='utf-8')
m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
m['scenarios'][name]={'kind':'controlled capacity contraction, same real GPU arrival sequence as P7',
 'nodes_file':name+'/nodes.csv','measurement_window_sec':window,'arrival_horizon_sec':15*DAY,
 'max_sim_time_sec':90*DAY,'sha256':sha(out/'tasks.csv'),
 'transformation':'all 12869 GPU tasks from P7, CPU-only excluded; source30 nodes -> deterministic first19 nodes, no task-time/resource/priority changes',
 **audit(rows,nodes,window)}
(root/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(m['scenarios'][name]),flush=True)
