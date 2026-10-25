"""Convert the all-pods warmup scenario suite to vmagentpro CSV inputs.

Usage: python prepare_inputs.py [path-to-vmagent_scenarios_warmup_allpods]
"""
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('vmagent_scenarios_warmup_allpods')
DST = Path(__file__).resolve().parent / 'inputs'
DST.mkdir(parents=True, exist_ok=True)

nodes = pd.read_csv(SRC / 'nodes.csv')
out_nodes = pd.DataFrame({
    'server_id': nodes.node_id,
    'cluster_id': nodes.source_cluster_id,
    'gpu_spec_public': nodes.gpu_type,
    'gpu_count': nodes.gpu_count,
    'cpu_capacity_cores': nodes.cpu_capacity_cores,
})
assert len(out_nodes) == 20
assert out_nodes.cluster_id.nunique() == 1
assert out_nodes.server_id.is_unique
out_nodes.to_csv(DST / 'nodes.csv', index=False)

meta = {}
for scen in 'ABCDE':
    df = pd.read_csv(SRC / f'jobs_scenario_{scen}.csv')
    out = pd.DataFrame({
        'pod_id': df.source_pod_id,
        'gpu_spec_public': 'H20',
        'gpu_request': df.gpu_request,
        'cpu_request_cores': df.cpu_request_cores,
        'submit_time_sec': df.submit_time_sec,
        'service_sec': df.service_sec,
        'cluster_id': out_nodes.cluster_id.iloc[0],
        'duration_match_status': df.service_source,
        'job_type_public': df.job_type_public,
        'priority_class': df.priority_class,
    })
    assert out.pod_id.is_unique
    assert (out.gpu_request >= 0).all() and (out.cpu_request_cores >= 0).all()
    assert (out.service_sec > 0).all()
    assert out.gpu_request.max() <= 8 and out.cpu_request_cores.max() <= 224
    out.to_csv(DST / f'jobs_{scen}.csv', index=False)
    meta[scen] = dict(
        tasks=len(out), gpu_tasks=int((out.gpu_request > 0).sum()),
        cpu_only_tasks=int((out.gpu_request == 0).sum()),
        max_submit_sec=float(out.submit_time_sec.max()),
        max_finish_sec=float((out.submit_time_sec + out.service_sec).max()),
        max_gpu=float(out.gpu_request.max()), max_cpu=float(out.cpu_request_cores.max()))

(DST / 'conversion_manifest.json').write_text(
    json.dumps(dict(source=str(SRC), nodes=len(out_nodes), cluster=out_nodes.cluster_id.iloc[0],
                    capacity_gpu=int(out_nodes.gpu_count.sum()),
                    capacity_cpu=int(out_nodes.cpu_capacity_cores.sum()),
                    scenarios=meta), ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
print('wrote', sorted(p.name for p in DST.iterdir()))
