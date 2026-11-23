"""Coarse date search on fixed servers; candidates require full-window audit."""
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "two_gpu_scenes"
BASE = Path(r"E:\cluster-trace-gpu-v2026\data")
ids = pa.array(pd.read_csv(HERE / "servers_20_H20.csv").server_id.tolist())
rows = []
dest = OUT / "daily_search_samples.csv"
if dest.exists():
    rows = pd.read_csv(dest).to_dict("records")
done = {int(r["day"]) for r in rows}
for day in range(185):
    if day in done:
        continue
    folder = BASE / "asi_opensource_pod_hourly" / "asi_opensource_pod_hourly" / f"day={day}" / "hour=12"
    parts = []
    for file in sorted(folder.glob("*.parquet")):
        table = pq.ParquetFile(file).read(columns=["server_id", "used_gpu_hours", "gpu_request", "cpu_request_cores"])
        parts.append(table.filter(pc.is_in(table.column("server_id"), value_set=ids)).to_pandas())
    if not parts:
        continue
    frame = pd.concat(parts, ignore_index=True)
    row = dict(day=day, hour=12, observed_servers=frame.server_id.nunique(),
        gpu_ratio=frame.used_gpu_hours.sum()/160, gpu_request_ratio=frame.gpu_request.sum()/160,
        cpu_request_ratio=frame.cpu_request_cores.sum()/3840, pod_rows=len(frame))
    rows.append(row)
    pd.DataFrame(rows).to_csv(dest, index=False)
    if day % 10 == 0 or row["gpu_ratio"] < .8:
        print(row, flush=True)
print(pd.DataFrame(rows).sort_values("gpu_ratio").head(12).to_string(index=False), flush=True)
