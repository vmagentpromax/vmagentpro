from pathlib import Path
import csv,hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.sha256.json').read_text(encoding='utf8'))
for relative,digest in manifest['files'].items():
    p=(root/relative).resolve();p.relative_to(root)
    assert p.is_file(),relative
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,relative
for s in ['S1','S2','S3','S4']:
    folder=root/'01_场景数据与构建'/(s+'_selected')
    for stem,count in [('servers',100),('initial_pods',293),('arrivals',120 if s=='S4' else 160)]:
        with (folder/(stem+'.csv')).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==count,(s,stem,len(rows))
        assert [int(r['sequence_index']) for r in rows]==list(range(1,count+1))
    frozen=json.loads((root/'02_三指标评测与复现'/(s+'.json')).read_text(encoding='utf8'))
    assert len(frozen['servers'])==100 and len(frozen['initial_pods'])==293
    assert len(frozen['arrivals'])==(120 if s=='S4' else 160)
assert len(list((root/'02_三指标评测与复现/logs_py311').rglob('*.json')))==160
print('PASS: '+str(len(manifest['files']))+' files match; selected server/task sequences and 160 logs present.')
