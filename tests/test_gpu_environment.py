import csv
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from vmagent.gpu.environment import GpuSchedEnv, NodeSpec, TaskSpec, DEFAULT_TASKS, load_workload, load_nodes
from vmagent.gpu.run import make_environment
from vmagent.gpu.environment import DEFAULT_NODES


def task(pid, gpu, cpu, arrival=0., service=1., model="H20"):
    return TaskSpec(pid, gpu, cpu, (model,), arrival, service)


class GpuEnvironmentTests(unittest.TestCase):
    def env(self, tasks, **options):
        return GpuSchedEnv([NodeSpec("n1", "cluster", "H20", 8., 8.)], tasks,
                           clock="fast", measurement_window=(0., 10.), **options)

    def test_waiting_uses_actual_start_for_completion(self):
        env = self.env([task("a", 8, 8, service=5), task("b", 1, 1, arrival=1)], queue_policy="fcfs")
        result = env.run()
        self.assertEqual(result["completed"], 2)
        self.assertEqual(env.records["b"]["start_time_sec"], 5.)
        self.assertEqual(env.records["b"]["finish_time_sec"], 6.)
        self.assertEqual(env.records["b"]["wait_sec"], 4.)
        self.assertEqual(env.free_gpu, [8.])
        self.assertEqual(env.free_cpu, [8.])

    def test_cpu_blocks_task_while_gpu_available(self):
        env = self.env([task("a", 1, 8, service=3), task("b", 1, 1, arrival=1)])
        env.run()
        self.assertEqual(env.records["b"]["start_time_sec"], 3.)
        self.assertLess(env.summary()["peak_gpu_reserved"], 8)

    def test_rejections_do_not_block_following_tasks(self):
        env = self.env([task("wrong-model", 1, 1, model="A10"), task("too-big", 9, 1),
                        task("too-much-cpu", 1, 9), task("valid", .19, 1.25)])
        result = env.run()
        self.assertEqual(result["rejected"], 3)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["unfinished"], 0)
        self.assertEqual(env.records["wrong-model"]["rejection_reason"], "gpu_model_unavailable")

    def test_fcfs_and_backfill_have_different_waits(self):
        tasks = [task("a", 6, 6, service=5), task("b", 4, 4, arrival=1, service=2),
                 task("c", 1, 1, arrival=2)]
        backfill = self.env(tasks, queue_policy="backfill")
        strict = self.env(tasks, queue_policy="fcfs")
        backfill.run(); strict.run()
        self.assertEqual(backfill.records["c"]["start_time_sec"], 2.)
        self.assertEqual(strict.records["c"]["start_time_sec"], 5.)
        self.assertEqual(backfill.records["b"]["finish_time_sec"], 7.)

    def test_completion_precedes_arrival_at_same_time(self):
        env = self.env([task("a", 8, 8, service=1), task("b", 8, 8, arrival=1)])
        env.run()
        events = [e["event"] for e in env.events if e["time_sec"] == 1.]
        self.assertLess(events.index("complete"), events.index("arrival"))
        self.assertEqual(env.records["b"]["wait_sec"], 0.)

    def test_cutoff_retains_running_and_future_tasks(self):
        env = self.env([task("a", 1, 1, service=10), task("b", 1, 1, arrival=20)], max_sim_time=5)
        result = env.run()
        self.assertEqual(result["end_reason"], "max_sim_time")
        self.assertEqual(result["sim_time_sec"], 5.)
        self.assertEqual(result["running"], 1)
        self.assertEqual(result["not_arrived"], 1)
        self.assertEqual(result["unfinished"], 2)
        self.assertEqual(result["measurement_denominator_sec"], 5.)

    def test_background_counted_once_and_can_reject(self):
        env = self.env([task("a", 1, 5), task("b", 1, 4)], cpu_background_fraction=.5)
        result = env.run()
        self.assertEqual(result["rejected"], 1)
        self.assertEqual(result["completed"], 1)
        self.assertAlmostEqual(result["reserved_cpu_ratio"], .55)
        self.assertEqual(env.free_cpu, [4.])

    def test_invalid_action_has_no_allocation_and_observation_hides_duration(self):
        env = self.env([task("a", .19, 1.25)])
        obs = env.reset()
        self.assertNotIn("service_sec", obs["task"])
        self.assertEqual(obs["avail"], [1])
        with self.assertRaises(ValueError):
            env.step(-1)
        self.assertEqual(env.free_gpu, [8.])
        env.step(0)
        self.assertAlmostEqual(env.free_gpu[0], 8.)

    def test_duplicate_and_bad_numeric_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"tasks.json"
            row = {"pod_id": "a", "gpu_request": .19, "cpu_request_cores": 1.25,
                   "gpu_spec_public": "H20", "submit_time_sec": 0, "service_sec": 1}
            path.write_text(json.dumps({"tasks": [row, row]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate pod_id"):
                load_workload(path)
            row["service_sec"] = float("nan")
            path.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_workload(path)
            path.write_text("123", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_workload(path)

    def test_full_real_scene_A_and_csv_input(self):
        for policy in ("first_fit", "best_fit", "random"):
            env, _ = make_environment(DEFAULT_NODES, DEFAULT_TASKS, clock="fast")
            result = env.run(policy)
            self.assertEqual(result["completed"], 50)
            self.assertEqual(result["rejected"], 0)
            self.assertEqual(result["unfinished"], 0)
            self.assertAlmostEqual(result["sim_time_sec"], 105.74080476059278)
            self.assertAlmostEqual(result["reserved_gpu_ratio"], result["offered_gpu_ratio"])
            self.assertAlmostEqual(result["reserved_cpu_ratio"], result["offered_cpu_ratio"])
            self.assertTrue(all(abs(g-n.gpu_capacity) < 1e-8 for g,n in zip(env.free_gpu,env.nodes)))
            with tempfile.TemporaryDirectory() as folder:
                env.export(folder)
                with (Path(folder)/"tasks.csv").open(encoding="utf-8-sig") as output:
                    self.assertEqual(len(list(csv.DictReader(output))), 50)
        json_tasks, _ = load_workload(DEFAULT_TASKS)
        csv_tasks, _ = load_workload(DEFAULT_TASKS.with_suffix(".csv"))
        self.assertEqual([(t.pod_id,t.gpu,t.cpu,t.submit_time,t.service_time) for t in json_tasks],
                         [(t.pod_id,t.gpu,t.cpu,t.submit_time,t.service_time) for t in csv_tasks])

    def test_realtime_can_be_cancelled_while_waiting(self):
        env = GpuSchedEnv([NodeSpec("n", "c", "H20", 8, 8)], [task("a", 1, 1, service=10)], clock="realtime")
        worker = threading.Thread(target=env.run)
        worker.start()
        deadline = time.monotonic()+2
        while not env.initialized and time.monotonic() < deadline:
            time.sleep(.001)
        time.sleep(.02)
        env.stop_requested.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(env.end_reason, "cancelled")
        self.assertEqual(env.summary()["unfinished"], 1)
        self.assertGreater(env.current_time, .005)
        self.assertLess(env.current_time, 2.)

    def test_finished_wall_time_does_not_keep_growing(self):
        env = self.env([task("a", 1, 1)])
        env.run()
        elapsed = env.summary()["wall_time_sec"]
        time.sleep(.002)
        self.assertEqual(env.summary()["wall_time_sec"], elapsed)


if __name__ == "__main__":
    unittest.main()
