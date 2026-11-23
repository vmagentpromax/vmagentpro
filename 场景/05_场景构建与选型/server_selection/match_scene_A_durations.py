"""Read execution summary spans for the existing 50 scene-A pod IDs."""
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "real_pod_sequences" / "scene_A_timing"
SUMMARY = Path(r"E:\cluster-trace-gpu-v2026\data\asi_opensource_job_execution_summary\asi_opensource_job_execution_summary\part-000.parquet")


def main():
    tasks = pd.read_csv(HERE / "real_pod_sequences" / "scene_A_real_H20_pods.csv")
    assert len(tasks) == 50 and tasks.pod_id.is_unique
    ids = pa.array(tasks.pod_id.tolist())
    columns = ["pod_id", "workload_id", "server_id", "gpu_spec_public", "gpu_request", "duration_hours", "job_type_public", "priority_class"]
    pieces = []
    offset = 0
    for batch in pq.ParquetFile(SUMMARY).iter_batches(batch_size=262144, columns=columns):
        mask = pc.is_in(batch.column("pod_id"), value_set=ids)
        matched = batch.filter(mask)
        if matched.num_rows:
            source_indices = np.flatnonzero(mask.to_numpy(zero_copy_only=False)) + offset
            matched = pa.RecordBatch.from_arrays(list(matched.columns) + [pa.array(source_indices)],
                names=matched.schema.names + ["execution_summary_row_index_0based"])
            pieces.append(matched)
        offset += batch.num_rows
    matches = pa.Table.from_batches(pieces).to_pandas() if pieces else pd.DataFrame(columns=columns)
    OUT.mkdir(parents=True, exist_ok=True)
    matches.to_csv(OUT / "matched_original_execution_spans.csv", index=False, encoding="utf-8-sig")
    counts = matches.groupby("pod_id").size()
    print("Matched execution rows:", len(matches), "Matched IDs:", len(counts))
    print("Rows per ID distribution:", counts.value_counts().to_dict())
    print("Missing IDs:", sorted(set(tasks.pod_id) - set(matches.pod_id)))
    joined = tasks.merge(matches, on="pod_id", how="left", suffixes=("_pod", "_span"))
    for col in ["gpu_spec_public", "gpu_request", "server_id", "job_type_public", "priority_class"]:
        print(col, "exact matches", int((joined[col+"_pod"] == joined[col+"_span"]).sum()), "of", len(joined))
    print(matches[["pod_id", "duration_hours", "gpu_request", "job_type_public"]].to_string(index=False))
    print("Duration quantiles:", matches.duration_hours.quantile([0,.25,.5,.75,.95,1]).to_dict())


if __name__ == "__main__":
    main()
