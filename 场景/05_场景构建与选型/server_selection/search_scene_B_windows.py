"""Exhaust all contiguous, whole-hour windows in the observed 72 hours."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "real_pod_sequences" / "scene_B_continuous_72h"


def main():
    hourly = pd.read_csv(OUT / "hourly_load_proxy.csv").sort_values("window_hour_index")
    assert hourly.window_hour_index.tolist() == list(range(72))
    gpu = np.r_[0, hourly.gpu_request_sum.cumsum()]
    cpu = np.r_[0, hourly.cpu_request_cores_sum.cumsum()]
    rows = []
    for start in range(72):
        for end in range(start + 1, 73):
            hours = end - start
            gh, ch = gpu[end] - gpu[start], cpu[end] - cpu[start]
            gr, cr = gh / (160 * hours), ch / (3840 * hours)
            rows.append(dict(start_day=109 + start // 24, start_hour=start % 24,
                last_included_day=109 + (end - 1) // 24, last_included_hour=(end - 1) % 24,
                window_hours=hours, gpu_hours_proxy=gh, cpu_core_hours_proxy=ch,
                gpu_ratio_proxy=gr, cpu_ratio_proxy=cr,
                exact_target_satisfied=bool(np.isclose(gr, .9, rtol=0, atol=1e-8) and
                                            np.isclose(cr, .3, rtol=0, atol=1e-8)),
                high_gpu_low_cpu_range_satisfied=bool(.8 <= gr <= .95 and 0 <= cr <= .4),
                squared_target_distance=(gr - .9)**2 + (cr - .3)**2))
    windows = pd.DataFrame(rows)
    windows.to_csv(OUT / "all_contiguous_window_checks.csv", index=False, encoding="utf-8-sig")
    summary = dict(windows_checked=len(windows), window_duration_hours_range=[1, 72],
        exact_90_30_matches=int(windows.exact_target_satisfied.sum()),
        high_gpu_low_cpu_range_matches=int(windows.high_gpu_low_cpu_range_satisfied.sum()),
        minimum_hourly_gpu_ratio=float(hourly.gpu_ratio_proxy.min()),
        minimum_hourly_cpu_ratio=float(hourly.cpu_ratio_proxy.min()),
        closest_by_squared_ratio_distance=windows.sort_values("squared_target_distance").iloc[0].to_dict(),
        conclusion="No qualifying window within days109-111. Every contiguous hourly average is a weighted average of its hourly values; changing duration cannot reduce GPU below the minimum hourly GPU ratio, or CPU below its corresponding hourly minimum. Sub-hour boundaries cannot be recovered from hourly observations.")
    (OUT / "window_search_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
