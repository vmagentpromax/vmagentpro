"""Rebuild complete suite or a single template instance; compare saved results."""
from __future__ import annotations
import argparse,shutil,json
from pathlib import Path
import pandas as pd
import source_adapter as src
from build_suite import build,workload,layout,source_checks,evaluate,config,load,dump

def instance(root,template,output):
    if output.exists():raise RuntimeError('Output must be a new path')
    cfg=load(template/'config.json');cache=root/'source_audit';e=cache/'evaluator_only'
    h,i,a,s4,factor=workload(pd.read_parquet(e/'source_server_snapshot.parquet'),pd.read_parquet(e/'source_eligible_pods.parquet'),pd.read_parquet(e/'source_summary_matches.parquet'),cfg['seed'],cfg['arrival_mean_interval_sec_S1_S3'])
    s=cfg['scenario'];initial=layout(h,i,'S2' if s=='S4' else s,cfg['seed']);arrivals=s4 if s=='S4' else a
    source_checks(cache,h,initial,arrivals);output.mkdir(parents=True)
    for name,frame in [('servers',h),('initial_pods',initial),('arrivals',arrivals)]:
        frame.to_parquet(output/f'{name}.parquet',index=False);pd.testing.assert_frame_equal(frame,pd.read_parquet(template/f'{name}.parquet'))
    dump(output/'config.json',cfg);(output/'evaluator_only').mkdir()
    layout(h,i,'S1',cfg['seed']).to_parquet(output/'evaluator_only'/'reference_initial.parquet',index=False)
    v,rows=evaluate(output)
    for result in (output/'baselines').glob('*.json'):
        assert load(result)==load(template/'baselines'/result.name),result.name
    report={'status':'PASS','input_frames_equal':True,'baseline_metrics_and_all_events_equal':True,'scenario':s,'seed':cfg['seed']}
    dump(output/'reproduction_check.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--instance',type=Path);ap.add_argument('--raw-source',type=Path);a=ap.parse_args();root=Path(__file__).parent
    if a.instance:instance(root,a.instance.resolve(),a.output.resolve())
    else:
        cache=root/'source_audit'
        if a.raw_source:
            cache=a.output.resolve().parent/(a.output.name+'_fresh_source_cache')
            if cache.exists():raise RuntimeError('Fresh-source cache path already exists')
            cache.mkdir(parents=True);s,p,d,prov=src.load_source(a.raw_source,cache);dump(cache/'provenance.json',prov)
        build(argparse.Namespace(output=a.output,source_cache=cache))
