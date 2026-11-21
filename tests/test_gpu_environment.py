import csv
import json
import math
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

    def test_backfill_keeps_gpu_waiters_in_order_with_identical_demands(self):
        tasks = [task("blocker", 8, 8, service=5),
                 task("first", 5, 5, arrival=1), task("second", 5, 5, arrival=2)]
        env = self.env(tasks)
        env.run()
        self.assertEqual(env.records["first"]["start_time_sec"], 5.)
        self.assertEqual(env.records["second"]["start_time_sec"], 6.)

    def test_spotgpu_preempts_lp_and_resumes_after_checkpoint(self):
        spot = task("spot", 8, 8, service=10)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        urgent = task("urgent", 8, 8, arrival=1, service=1)
        urgent = TaskSpec(urgent.pod_id, urgent.gpu, urgent.cpu, urgent.allowed_gpu_types,
                          urgent.submit_time, urgent.service_time, "HP")
        env = self.env([spot, urgent])
        result = env.run("spotgpu")
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["preemptions"], 1)
        self.assertEqual(env.records["spot"]["preemption_count"], 1)
        self.assertEqual(env.records["urgent"]["start_time_sec"], 1.)
        self.assertEqual(env.records["spot"]["finish_time_sec"], 12.)

    def test_spotgpu_fcfs_queue_still_indexes_hp_preemption_candidates(self):
        spot = task("spot", 8, 8, service=10)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        urgent = task("urgent", 8, 8, arrival=1, service=1)
        urgent = TaskSpec(urgent.pod_id, urgent.gpu, urgent.cpu, urgent.allowed_gpu_types,
                          urgent.submit_time, urgent.service_time, "HP")
        env = self.env([spot, urgent], queue_policy="fcfs")
        result = env.run("spotgpu")
        self.assertEqual(result["preemptions"], 1)
        self.assertEqual(env.records["urgent"]["start_time_sec"], 1.)
        self.assertEqual(env.records["spot"]["finish_time_sec"], 12.)

    def test_spotgpu_retries_preemption_when_resources_are_released(self):
        blocker = task("blocker", 4, 4, service=2)
        spot = task("spot", 4, 4, service=10)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        urgent = task("urgent", 8, 8, arrival=1, service=1)
        urgent = TaskSpec(urgent.pod_id, urgent.gpu, urgent.cpu, urgent.allowed_gpu_types,
                          urgent.submit_time, urgent.service_time, "HP")
        env = self.env([blocker, spot, urgent])
        result = env.run("spotgpu")
        self.assertEqual(result["preemptions"], 1)
        self.assertEqual(env.records["urgent"]["start_time_sec"], 2.)
        self.assertEqual(env.records["spot"]["finish_time_sec"], 13.)

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


class TopologyAndFragmentationMetricTests(unittest.TestCase):
    """Phase A adds observational fields only: asw_id, workload_id, entropy and
    partially-occupied node time. None of them may change scheduling decisions."""

    def stranded_env(self, tasks, nodes=None, **options):
        nodes = nodes or [NodeSpec("n1", "cluster", "H20", 8., 8.)]
        return GpuSchedEnv(nodes, tasks, clock="fast", measurement_window=(0., 10.), **options)

    def test_absent_labels_degrade_to_zero_without_error(self):
        env = self.stranded_env([task("a", 2, 2, service=5)])
        result = env.run()
        self.assertEqual(result["topology"]["nodes_with_asw"], 0)
        self.assertEqual(result["topology"]["tasks_with_workload_id"], 0)
        self.assertEqual(result["topology"]["workloads_placed"], 0)
        self.assertEqual(result["topology"]["gpu_entropy_nats_mean"], 0.)
        self.assertEqual(result["topology"]["gpu_entropy_nats_max"], 0.)

    def test_partial_and_stranded_node_time_matches_hand_computation(self):
        # t in [0,5): 8/8 used -> fully occupied, contributes neither.
        # t in [5,10]: 2/8 used -> partial, 6 GPUs stranded.
        env = self.stranded_env([task("a", 8, 8, service=5), task("b", 2, 2, service=10)],
                                queue_policy="fcfs")
        frag = env.run()["fragmentation"]
        self.assertEqual(frag["partial_node_time_sec"], 5.)
        self.assertEqual(frag["stranded_gpu_time_sec"], 30.)
        self.assertAlmostEqual(frag["partial_node_time_ratio"], 5./(1*10.))
        self.assertAlmostEqual(frag["stranded_gpu_time_ratio"], 30./(8*10.))

    def test_fully_idle_and_fully_used_nodes_are_not_counted_as_partial(self):
        env = self.stranded_env([task("a", 8, 8, service=5)])
        frag = env.run()["fragmentation"]
        self.assertEqual(frag["partial_node_time_sec"], 0.)
        self.assertEqual(frag["stranded_gpu_time_sec"], 0.)
        self.assertEqual(frag["nodes_partially_occupied_end"], 0)
        self.assertEqual(frag["nodes_idle_end"], 1)

    def entropy_env(self):
        from dataclasses import replace
        nodes = [NodeSpec("n1", "cluster", "H20", 8., 8., "A"),
                 NodeSpec("n2", "cluster", "A100", 8., 8., "B")]
        # Model whitelist pins each pod to one node, so placement is deterministic.
        specs = [task("p1", 4, 4, model="H20"), task("p2", 4, 4, model="A100"),
                 task("p3", 2, 1, model="H20"), task("p4", 2, 1, model="H20")]
        groups = ["w1", "w1", "w2", "w2"]
        tasks = [replace(t, workload_id=g) for t, g in zip(specs, groups)]
        return GpuSchedEnv(nodes, tasks, clock="fast", measurement_window=(0., 10.),
                           queue_policy="fcfs")

    def test_entropy_is_natural_log_over_asws_per_workload(self):
        topo = self.entropy_env().run()["topology"]
        self.assertEqual(topo["nodes_with_asw"], 2)
        self.assertEqual(topo["tasks_with_workload_id"], 4)
        self.assertEqual(topo["workloads_placed"], 2)
        # w1 puts 4 GPUs in ASW A and 4 in ASW B -> uniform over two -> ln 2.
        self.assertAlmostEqual(topo["gpu_entropy_nats_max"], math.log(2.))
        # w2 keeps all 4 GPUs inside ASW A -> entropy 0; mean is half of ln 2.
        self.assertAlmostEqual(topo["gpu_entropy_nats_mean"], math.log(2.)/2)
        self.assertEqual(topo["asw_span_histogram"], {"1": 1, "2": 1})
        self.assertEqual(topo["workload_gpu_in_measurable_groups"], 12.)
        self.assertEqual(topo["workload_gpu_sec_in_multi_asw_workloads"], 8.)

    def test_single_pod_workload_has_zero_entropy_by_construction(self):
        # This is why phase A alone cannot demonstrate the entropy policy: one pod
        # always lands on one node, hence in one ASW, whatever scenario is chosen.
        from dataclasses import replace
        nodes = [NodeSpec("n1", "cluster", "H20", 8., 8., "A"),
                 NodeSpec("n2", "cluster", "H20", 8., 8., "B")]
        env = GpuSchedEnv(nodes, [replace(task("solo", 4, 4), workload_id="w1")],
                          clock="fast", measurement_window=(0., 10.))
        topo = env.run()["topology"]
        self.assertEqual(topo["asw_span_histogram"], {"1": 1})
        self.assertEqual(topo["gpu_entropy_nats_max"], 0.)

    def test_loaders_read_optional_labels_and_tolerate_missing_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = Path(folder, "nodes.csv")
            nodes.write_text("server_id,cluster_id,gpu_spec_public,gpu_count,cpu_capacity_cores,asw_id\n"
                             "n1,c1,H20,8,192,  A  \nn2,c1,H20,8,192,\n", encoding="utf-8")
            loaded = load_nodes(nodes)
            self.assertEqual([n.asw_id for n in loaded], ["A", ""])

            tasks = Path(folder, "jobs.csv")
            tasks.write_text("pod_id,gpu_spec_public,gpu_request,cpu_request_cores,submit_time_sec,"
                             "service_sec,workload_id\n"
                             "p1,H20,1,1,0,1,w1\np2,H20,1,1,0,1,nan\np3,H20,1,1,0,1,\n", encoding="utf-8")
            loaded, _ = load_workload(tasks)
            self.assertEqual([t.workload_id for t in loaded], ["w1", "", ""])


if __name__ == "__main__":
    unittest.main()
