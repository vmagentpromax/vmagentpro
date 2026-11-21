"""Tests for the parallel policy matrix and the tolerance-based result comparer."""
import json
import tempfile
import unittest
from pathlib import Path

from vmagent.gpu import compare
from vmagent.gpu.environment import GpuSchedEnv, NodeSpec, TaskSpec
from vmagent.gpu.matrix import PLACEMENTS, QUEUES, _run_one, build_jobs, run_matrix


def task(pid, gpu, cpu, arrival=0., service=1., model="H20", priority="Other"):
    return TaskSpec(pid, gpu, cpu, (model,), arrival, service, priority)


class CompareTests(unittest.TestCase):
    def test_last_ulp_noise_is_not_reported_as_a_difference(self):
        # The exact pair measured across CPython 3.11 and 3.13.
        ref = {"per_model": {"XPU-A": {"offered_gpu_ratio": 1.040616417218244}}}
        cand = {"per_model": {"XPU-A": {"offered_gpu_ratio": 1.0406164172182435}}}
        self.assertEqual(compare.diff_summaries(ref, cand), [])

    def test_real_value_change_is_reported(self):
        ref = {"completed": 119495, "reserved_gpu_ratio": 0.7447563657407585}
        cand = {"completed": 119495, "reserved_gpu_ratio": 0.70}
        diffs = compare.diff_summaries(ref, cand)
        self.assertEqual([d[0] for d in diffs], ["reserved_gpu_ratio"])

    def test_missing_field_is_reported_as_presence(self):
        diffs = compare.diff_summaries({"a": 1}, {})
        self.assertEqual(diffs[0][3], "presence")

    def test_volatile_fields_are_ignored_by_default(self):
        ref = {"wall_time_sec": 42.1, "sim_time_sec": 3628230.0, "completed": 5}
        cand = {"wall_time_sec": 118.2, "sim_time_sec": 3628231.5, "completed": 5}
        self.assertEqual(compare.diff_summaries(ref, cand), [])
        self.assertEqual(len(compare.diff_summaries(ref, cand, ignore_volatile=False)), 2)

    def test_nested_dict_presence_is_reported(self):
        ref = {"per_model": {"XPU-A": {"gpu_capacity": 160.0}, "H20": {"gpu_capacity": 80.0}}}
        cand = {"per_model": {"XPU-A": {"gpu_capacity": 160.0}}}
        self.assertEqual([d[0] for d in compare.diff_summaries(ref, cand)],
                         ["per_model.H20"])

    def test_booleans_are_compared_exactly_not_numerically(self):
        self.assertEqual(len(compare.diff_summaries({"drained": True}, {"drained": False})), 1)
        self.assertEqual(compare.diff_summaries({"drained": True}, {"drained": True}), [])


class MatrixTests(unittest.TestCase):
    def nodes(self):
        return [NodeSpec("n1", "c", "H20", 8., 8.), NodeSpec("n2", "c", "H20", 8., 8.)]

    def write_inputs(self, folder):
        nodes = Path(folder)/"nodes.csv"
        nodes.write_text("server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores\n"
                         "n1,c,H20,8,8\nn2,c,H20,8,8\n", encoding="utf-8")
        tasks = Path(folder)/"jobs.csv"
        tasks.write_text("pod_id,gpu_spec_public,gpu_request,cpu_request_cores,submit_time_sec,"
                         "service_sec,priority_class\n"
                         "p1,H20,4,4,0,5,LP\np2,H20,4,4,1,5,LP\np3,H20,6,6,2,5,HP\n", encoding="utf-8")
        return nodes, tasks

    def test_build_jobs_covers_every_combination(self):
        jobs = build_jobs("n.csv", "j.csv", ("first_fit", "best_fit"), ("fcfs", "backfill"),
                          {}, Path("/tmp/out"))
        self.assertEqual(len(jobs), 4)
        self.assertEqual({(j[2], j[3]) for j in jobs},
                         {("first_fit", "fcfs"), ("first_fit", "backfill"),
                          ("best_fit", "fcfs"), ("best_fit", "backfill")})

    def test_default_matrix_shape_is_placements_times_queues(self):
        jobs = build_jobs("n", "j", PLACEMENTS, QUEUES, {}, Path("/tmp/out"))
        self.assertEqual(len(jobs), len(PLACEMENTS)*len(QUEUES))

    def test_worker_cell_matches_serial_run_field_by_field(self):
        # The core claim of M7: running a cell inside a worker process must not
        # change any number relative to running it in-process.
        with tempfile.TemporaryDirectory() as folder:
            nodes, tasks = self.write_inputs(folder)
            options = dict(max_sim_time=100000., cpu_background_fraction=0.,
                           checkpoint_interval=3600., seed=42, measurement_window=[0., 100.])
            result = _run_one((nodes, tasks, "first_fit", "fcfs", options, Path(folder)/"cell"))
            self.assertTrue(result["ok"], result.get("error"))

            # Must mirror write_inputs() exactly: service 5, not the helper default.
            env = GpuSchedEnv(self.nodes(), [task("p1", 4, 4, service=5.), task("p2", 4, 4, arrival=1., service=5.),
                                             task("p3", 6, 6, arrival=2., service=5., priority="HP")],
                              clock="fast", queue_policy="fcfs", measurement_window=(0., 100.),
                              max_sim_time=100000., seed=42)
            serial = env.run("first_fit")
            diffs = compare.diff_summaries(serial, result["summary"])
            self.assertEqual(diffs, [], compare.format_diff(diffs))

    def test_matrix_writes_one_directory_per_cell_and_an_index(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes, tasks = self.write_inputs(folder)
            out = Path(folder)/"matrix"
            results = run_matrix(nodes, tasks, ("first_fit", "best_fit"), ("fcfs",), out,
                                 workers=2, max_sim_time=100000., seed=42,
                                 measurement_window=[0., 100.])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r["ok"] for r in results))
            self.assertTrue((out/"F-first_fit-fcfs"/"summary.json").exists())
            self.assertTrue((out/"F-best_fit-fcfs"/"summary.json").exists())
            index = json.loads((out/"matrix_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["cells"], 2)
            self.assertIn("environment", json.loads(
                (out/"F-first_fit-fcfs"/"run_manifest.json").read_text(encoding="utf-8")))

    def test_a_failing_cell_does_not_abort_the_matrix(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = Path(folder)/"missing.csv"
            tasks = self.write_inputs(folder)[1]
            results = run_matrix(nodes, tasks, ("first_fit", "best_fit"), ("fcfs",),
                                 Path(folder)/"m", workers=2, max_sim_time=1000., seed=42,
                                 measurement_window=[0., 100.])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(not r["ok"] for r in results))
            self.assertIn("error", results[0])


if __name__ == "__main__":
    unittest.main()
