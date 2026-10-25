"""Check the methodological boundaries of natural-window scenario construction."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs/server_selection"))
from construct_request_scenes import extract_templates, timing
from rebuild_request_scenes import windows
from run_request_scenes import metrics, interval
from vmagent.gpu.environment import GpuSchedEnv, NodeSpec, TaskSpec


class RequestSceneTests(unittest.TestCase):
    def test_request_changes_including_zero_gpu_are_excluded(self):
        rows = []
        for h in range(6):
            for pid in ("stable", "changing", "zero", "cpu-changing"):
                rows.append(dict(pod_id=pid, absolute_hour=h, gpu_spec_public="H20", cluster_id="c",
                    server_id="n", gpu_request=(2 if pid=="changing" and h==3 else
                    0 if pid=="zero" and h==3 else 1),
                    cpu_request_cores=3 if pid=="cpu-changing" and h==3 else 2))
        tasks, excluded, _ = extract_templates(pd.DataFrame(rows),0,6)
        self.assertEqual(tasks.pod_id.tolist(),["stable"])
        self.assertEqual(set(excluded.pod_id),{"changing","zero","cpu-changing"})

    def test_missing_hours_do_not_make_contiguous_windows(self):
        obs = pd.DataFrame([dict(absolute_hour=h,pod_id=str(i),gpu_request=1)
                            for h in (0,1,2,4,5,6,7,8,9) for i in range(50)])
        hours = pd.DataFrame([dict(absolute_hour=h,rho_gpu=.9,rho_cpu=.7,used_gpu_ratio=.8)
                              for h in sorted(obs.absolute_hour.unique())])
        with tempfile.TemporaryDirectory() as folder:
            result = windows(hours,obs,Path(folder))
        self.assertEqual(len(result),1)
        self.assertEqual(result.iloc[0].start_hour,4)
        self.assertEqual(result.iloc[0].hours,6)
        self.assertTrue(result.iloc[0].high_candidate)
        self.assertFalse(result.iloc[0].low_candidate)

    def test_rate_calibration_and_unit_scaling_are_consistent(self):
        frame = pd.DataFrame([dict(pod_id=str(i),duration_hours=i+1,gpu_request=(i%3)+.25,cpu_request_cores=4)
                              for i in range(50)])
        a, meta = timing(frame,.85,42,3600.)
        b, slower = timing(frame,.85,42,1800.)
        low, _ = timing(frame,.35,42,3600.)
        self.assertAlmostEqual(meta["expected_arrival_span_sec"],np.dot(a.gpu_request,a.service_sec)/(.85*160))
        self.assertEqual(a.pod_id.tolist(),b.pod_id.tolist())
        self.assertTrue(np.allclose(a.submit_time_sec*2,b.submit_time_sec))
        self.assertTrue(np.allclose(a.service_sec*2,b.service_sec))
        self.assertTrue(np.allclose(low.submit_time_sec,a.submit_time_sec*.85/.35))
        self.assertEqual(a.submit_time_sec.iloc[0],0)
        self.assertTrue((np.diff(a.submit_time_sec)>0).all())
        self.assertNotAlmostEqual(meta["actual_arrival_span_sec"],meta["expected_arrival_span_sec"])

    def env(self,tasks):
        return GpuSchedEnv([NodeSpec("n1","c","H20",8,8),NodeSpec("n2","c","H20",8,8)],
                           tasks,clock="fast",max_sim_time=100)

    def task(self,pid,gpu,cpu,arrival=0,service=10):
        return TaskSpec(pid,gpu,cpu,("H20",),arrival,service)

    def test_gpu_fragmentation_integrates_only_positive_intervals(self):
        env = self.env([self.task("a",5,1),self.task("b",5,1),self.task("c",4,1,1,1)])
        env.run("first_fit")
        result = metrics(env,.1)
        self.assertAlmostEqual(result["afr_joint"],9/11)
        self.assertAlmostEqual(result["gpu_fragmentation_fraction"],9/11)
        self.assertAlmostEqual(result["agu_reserved"],104/(16*11))
        self.assertAlmostEqual(result["mean_jct_sec"],10)

    def test_aggregate_cpu_shortage_is_not_called_fragmentation(self):
        env = self.env([self.task("a",1,8),self.task("b",1,8),self.task("c",1,1,1,1)])
        env.run("first_fit")
        result = metrics(env,.1)
        self.assertEqual(result["afr_joint"],0)
        self.assertEqual(result["gpu_fragmentation_fraction"],0)
        self.assertGreater(result["pending_time_fraction"],0)

    def test_worst_fit_prefers_more_remaining_joint_capacity(self):
        env = self.env([self.task("a",2,2),self.task("b",1,1)])
        env.reset()
        env.step(0)
        self.assertEqual(env.choose_action("best_fit"),0)
        self.assertEqual(env.choose_action("worst_fit"),1)

    def test_paired_confidence_interval_preserves_constant_effect(self):
        result = interval(np.array([2.,3.,4.])-np.array([1.,2.,3.]))
        self.assertEqual(result["mean"],1)
        self.assertEqual(result["ci95_low"],1)
        self.assertEqual(result["ci95_high"],1)


if __name__ == "__main__":
    unittest.main()
