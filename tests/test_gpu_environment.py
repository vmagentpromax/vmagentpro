import csv
import json
import math
import tempfile
import threading
import time
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

from vmagent.gpu.environment import GpuSchedEnv, NodeSpec, TaskSpec, DEFAULT_TASKS, EPS, ROOT, load_workload, load_nodes
from vmagent.gpu.metrics import (integrate_events, interval_overlap, percentile_linear,
                                 percentile_nearest_rank, run_segments)
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

    def lp_and_hp(self):
        """The one-node case where SpotGPU must preempt: 8 GPUs wanted, 8 exist."""
        spot = task("spot", 8, 8, service=10)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        urgent = task("urgent", 8, 8, arrival=1, service=1)
        urgent = TaskSpec(urgent.pod_id, urgent.gpu, urgent.cpu, urgent.allowed_gpu_types,
                          urgent.submit_time, urgent.service_time, "HP")
        return [spot, urgent]

    def test_event_integrals_reproduce_the_ledger_areas_including_under_preemption(self):
        # `integrate_events` is the primitive external re-derivations are supposed to use
        # instead of rebuilding the ledger from records, so it has to agree with the
        # engine's own incremental accounting on exactly the run where records lie.
        env = self.env(self.lp_and_hp())
        summary = env.run("spotgpu")
        self.assertEqual(1, summary["preemptions"])
        self.assertAlmostEqual(env.gpu_area,
                               integrate_events(env.events, "gpu_reserved", (0., 10.)))
        self.assertAlmostEqual(env.gpu_area_full,
                               integrate_events(env.events, "gpu_reserved"))
        self.assertAlmostEqual(env.cpu_area,
                               integrate_events(env.events, "cpu_reserved_cores", (0., 10.)))

    def test_run_segments_split_a_preempted_task_that_the_record_reading_cannot_see(self):
        env = self.env(self.lp_and_hp())
        env.run("spotgpu")
        segments = run_segments(env.events)
        self.assertEqual([1, 2], sorted(len(parts) for parts in segments.values()))
        self.assertEqual(2, len(segments["spot"]), "one preemption, two occupied intervals")
        self.assertEqual(1, len(segments["urgent"]))
        # The victim held the node for 11s to finish 10s of work: evicted at t=1 with a
        # 3600s checkpoint interval, none of that first second survived, so it runs again.
        # Occupancy and service are different quantities once preemption exists, and only
        # the event log can tell them apart.
        self.assertAlmostEqual(11., sum(e-s for s, e, _ in segments["spot"]))
        self.assertAlmostEqual(10., env.by_id["spot"].service_time)
        # records only ever holds the *last* placement, which is the whole problem: read
        # this way the first interval disappears and the eviction itself is invisible.
        self.assertEqual(1, env.records["spot"]["preemption_count"])
        self.assertAlmostEqual(2., env.records["spot"]["start_time_sec"])
        self.assertAlmostEqual(12., env.records["spot"]["finish_time_sec"])

    def test_reusing_an_environment_does_not_carry_preemption_into_another_policy(self):
        # `run("spotgpu")` turned preemption on and nothing turned it back off, so a
        # second run on the same object under a non-preempting policy silently evicted
        # LP tasks. Every recorded pipeline builds a fresh environment, which is exactly
        # why this survived: the bug only appears when the object is reused, and the
        # Gym-style reset()/step() API invites reuse.
        def build():
            spot = TaskSpec("spot", 8., 8., ("H20",), 0., 20., "LP", {})
            urgent = TaskSpec("hp", 8., 8., ("H20",), 1., 1., "HP", {})
            return GpuSchedEnv([NodeSpec("n1", "cluster", "H20", 8., 8.)], [spot, urgent],
                               clock="fast", queue_policy="fcfs", checkpoint_interval=3600.,
                               measurement_window=(0., 60.), max_sim_time=1000.)
        pristine = build().run("first_fit")
        shared = build()
        shared.run("spotgpu")
        reused = shared.run("first_fit")
        self.assertEqual(0, pristine["preemptions"])
        self.assertGreater(shared.run("spotgpu")["preemptions"], 0)
        self.assertEqual(pristine["preemptions"], reused["preemptions"])
        self.assertEqual(pristine["wasted_gpu_sec"], reused["wasted_gpu_sec"])
        self.assertEqual(pristine["reserved_gpu_ratio"], reused["reserved_gpu_ratio"])

    def test_running_total_sec_counts_the_segment_that_started_at_zero(self):
        # `record["start_time_sec"] or self.current_time` looks like a null guard and is
        # a falsy-zero bug: a task whose last placement was at t=0 has start_time_sec
        # 0.0, so the whole final segment was added as zero. Every warm-up task of
        # scenario F and every pod of an --initial-placement layout hit it. The field is
        # published in tasks.csv, where it is the only way to see how long a task really
        # held its resources as opposed to how long it was asked to run.
        env = self.env([task("a", 4, 4, service=10), task("b", 4, 4, service=10)])
        env.run("first_fit")
        for pid in ("a", "b"):
            self.assertEqual(0., env.records[pid]["start_time_sec"])
            self.assertEqual(10., env.records[pid]["running_total_sec"])

    def test_completed_task_has_run_exactly_its_service_time_once_per_segment(self):
        # Cross-check the record field against the event log on a run where the two
        # could disagree: preemption at t=0+epsilon would expose a mis-summed segment.
        spot = task("spot", 8, 8, service=20)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        urgent = task("hp", 8, 8, arrival=1, service=1)
        urgent = TaskSpec(urgent.pod_id, urgent.gpu, urgent.cpu, urgent.allowed_gpu_types,
                          urgent.submit_time, urgent.service_time, "HP")
        env = self.env([spot, urgent])
        env.run("spotgpu")
        occupied = sum(end-start for start, end, _ in run_segments(env.events)["spot"])
        self.assertAlmostEqual(env.records["spot"]["running_total_sec"], occupied)
        # 20 s of service plus the one second lost to the eviction, in a run that
        # started at t=0 - the case the falsy-zero bug used to report as 20 rather than 21.
        self.assertAlmostEqual(21., occupied)

    def test_an_eviction_that_cost_nothing_is_not_counted_as_a_loss(self):
        # Under FCFS the admission path puts a waiting LP task onto capacity freed at this
        # instant and the preemption path takes it back at the same instant, so
        # `preemptions` counts an event that disturbed nobody. Scenario F measured 244
        # evictions of which only 22 lost any work, so the two figures have to be
        # reportable separately rather than reconciled by quietly changing the old one.
        spot = task("spot", 8, 8, service=20)
        spot = TaskSpec(spot.pod_id, spot.gpu, spot.cpu, spot.allowed_gpu_types,
                        spot.submit_time, spot.service_time, "LP")
        first = task("hp1", 8, 8, arrival=1, service=1)
        first = TaskSpec(first.pod_id, first.gpu, first.cpu, first.allowed_gpu_types,
                         first.submit_time, first.service_time, "HP")
        second = task("hp2", 8, 8, arrival=2, service=1)
        second = TaskSpec(second.pod_id, second.gpu, second.cpu, second.allowed_gpu_types,
                          second.submit_time, second.service_time, "HP")
        env = self.env([spot, first, second])
        result = env.run("spotgpu")
        self.assertEqual(2, result["preemptions"])
        self.assertEqual(1, result["preemptions_with_loss"])
        self.assertEqual(1, result["disturbed_task_count"])
        self.assertEqual(1, result["preempted_task_count"], "the churned task was still "
                         "evicted twice, so the old count keeps counting it")
        self.assertEqual(1, env.records["spot"]["costly_preemptions"])

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


class BlockClassificationTests(unittest.TestCase):
    """_classify_block explains why a waiting task cannot start right now.

    The fragmentation case is the one that matters: aggregate free capacity is
    sufficient but no single node jointly fits, so only a migration or a
    defragmentation sweep can unblock the task.

    These unit tests set the resource ledger directly rather than driving step():
    under clock="fast" a step jumps to the next decision point, so a task that
    can never be placed would fast-forward past the occupants' completion and the
    state under test would already be gone.
    """

    def two_nodes(self):
        return [NodeSpec("n1", "c", "H20", 8., 8.), NodeSpec("n2", "c", "H20", 8., 8.)]

    def staged(self, free_gpu, free_cpu, nodes=None):
        env = GpuSchedEnv(nodes or self.two_nodes(),
                          [task("probe", 1, 1)], clock="fast", measurement_window=(0., 20.))
        env.reset()
        env.free_gpu, env.free_cpu = list(free_gpu), list(free_cpu)
        return env

    def test_direct_when_a_node_fits_now(self):
        env = self.staged([8., 8.], [8., 8.])
        self.assertEqual(env._classify_block(task("a", 2, 2)), "direct")

    def test_model_unavailable_when_no_node_has_the_model(self):
        env = self.staged([8., 8.], [8., 8.])
        self.assertEqual(env._classify_block(task("a", 1, 1, model="A100")), "model_unavailable")

    def test_gpu_capacity_shortage_is_aggregate_not_per_node(self):
        # 1 GPU free on n1 and none on n2: the aggregate is 1 but 2 are needed, so
        # this is a real capacity shortage, not something defragmentation can fix.
        env = self.staged([1., 0.], [8., 8.])
        self.assertEqual(env._classify_block(task("c", 2, 1)), "gpu_capacity_shortage")

    def test_cpu_capacity_shortage_when_gpu_aggregate_suffices(self):
        # Plenty of GPU in aggregate, no CPU left anywhere: the paper's third
        # fragmentation cause must not be mislabelled as GPU fragmentation.
        env = self.staged([4., 4.], [0., 0.])
        self.assertEqual(env._classify_block(task("c", 2, 2)), "cpu_capacity_shortage")

    def test_fragmentation_when_aggregate_fits_but_no_single_node_does(self):
        # n1 and n2 each keep 1 GPU and 1 CPU free: 2+2 aggregate is enough for c,
        # yet neither node can take it. This is the case defragmentation targets.
        env = self.staged([1., 1.], [1., 1.])
        probe = task("c", 2, 2)
        self.assertEqual(sum(env.free_gpu), 2.)
        self.assertEqual(max(env.free_gpu), 1.)
        self.assertFalse(any(env._fits(probe, j) for j in range(2)))
        self.assertEqual(env._classify_block(probe), "fragmentation")

    def test_fragmentation_is_scoped_to_model_compatible_nodes(self):
        # An A100 node with free capacity must not rescue an H20-only request, and
        # the H20 pool alone is what should be judged short.
        nodes = [NodeSpec("n1", "c", "H20", 8., 8.), NodeSpec("n2", "c", "A100", 8., 8.)]
        env = self.staged([1., 8.], [1., 8.], nodes)
        self.assertEqual(env._classify_block(task("c", 2, 2, model="H20")), "gpu_capacity_shortage")

    def test_background_cpu_is_excluded_before_shortage_is_claimed(self):
        env = GpuSchedEnv([NodeSpec("n1", "c", "H20", 8., 100.)],
                          [task("a", 1, 1)], clock="fast", measurement_window=(0., 20.),
                          cpu_background_fraction=0.5)
        env.reset()
        # 100 cores with half reserved: a 60-core request is a real shortage, not
        # something to blame on fragmentation.
        self.assertEqual(env.free_cpu, [50.])
        self.assertEqual(env._classify_block(task("big", 1, 60)), "cpu_capacity_shortage")

    def test_summary_reports_first_attempt_classes_and_coverage(self):
        occupied = [task("h1", 7, 7, service=5), task("h2", 7, 7, service=5)]
        env = GpuSchedEnv(self.two_nodes(), occupied + [task("c", 2, 2, arrival=1)],
                          clock="fast", measurement_window=(0., 20.), queue_policy="fcfs")
        block = env.run()["block_classification"]
        self.assertEqual(block["tasks_classified"], 3)
        self.assertEqual(block["first_attempt_classes"]["direct"], 2)
        self.assertEqual(block["first_attempt_classes"]["fragmentation"], 1)
        self.assertEqual(block["task_count"], 3)
        self.assertEqual(block["tasks_never_reaching_a_decision"], 0)
        self.assertAlmostEqual(block["fragmentation_share_of_classified"], 1/3)


class SpotGpuVictimTieBreakTests(unittest.TestCase):
    """Equal-cost victims must be chosen by pod_id, not by set iteration order.

    Regression test for a reproducibility bug found by tools/verify_golden_runs.py:
    node_spot_pids is a set of strings, CPython randomises string hashes per process,
    and sorted() is stable, so the old `sorted(pids, key=cost)` returned victims in an
    order that changed between processes. Measured on the full scenario F input:
    preemptions 232 under PYTHONHASHSEED=0 versus 242 under PYTHONHASHSEED=1, with
    wasted_gpu_sec 6,417 GPU·s apart and reserved_gpu_ratio 3.2 points apart, on
    identical code and identical input. Warmup tasks that all start at t=0 make ties
    the common case, not an edge case.
    """
    def stage(self, pids):
        node = NodeSpec("n1", "cluster", "H20", 8., 8.)
        tasks = [TaskSpec(pid, 2., 2., ("H20",), 0., 100., "LP") for pid in pids]
        tasks.append(TaskSpec("hp", 5., 5., ("H20",), 0., 1., "HP"))
        env = GpuSchedEnv([node], tasks, clock="fast", measurement_window=(0., 10.))
        env.reset()
        # Hand-stage four LP tasks sharing one node, all started at the same instant so
        # every victim cost is identical and only the tie-break can distinguish them.
        env.current_time = 4.
        env.free_gpu, env.free_cpu = [0.], [0.]
        env.node_gpu_used, env.node_cpu_used, env.node_running_count = [8.], [8.], [4]
        env.running = {pid: 0 for pid in pids}
        env.running_start = {pid: 0. for pid in pids}
        env.node_spot_pids = [set(pids)]
        return env

    def test_equal_cost_victims_are_chosen_in_pod_id_order(self):
        env = self.stage(["lp_d", "lp_c", "lp_b", "lp_a"])
        node, victims, cost = env._spotgpu_best_preemption(env.by_id["hp"])
        self.assertEqual(0, node)
        self.assertEqual(["lp_a", "lp_b", "lp_c"], victims)
        self.assertEqual(24., cost)

    def test_the_choice_does_not_depend_on_how_the_set_was_built(self):
        first = self.stage(["lp_a", "lp_b", "lp_c", "lp_d"])
        second = self.stage(["lp_d", "lp_c", "lp_b", "lp_a"])
        self.assertEqual(first._spotgpu_best_preemption(first.by_id["hp"])[1],
                         second._spotgpu_best_preemption(second.by_id["hp"])[1])

    def test_a_cheaper_task_is_still_chosen_over_a_lower_pod_id(self):
        # pod_id only breaks ties: the cost ordering from the paper must still win,
        # or the fix would have replaced one arbitrary rule with a worse one.
        env = self.stage(["lp_a", "lp_b", "lp_c", "lp_d"])
        env.running_start["lp_d"] = 3.9          # lp_d has just checkpointed
        node, victims, cost = env._spotgpu_best_preemption(env.by_id["hp"])
        self.assertIn("lp_d", victims)
        self.assertEqual(0, node)


class InitialPlacementTests(unittest.TestCase):
    """A prescribed t=0 layout is an input, not a decision.

    The defragmentation benchmark starts from a deliberately fragmented layout. If
    the policy re-derived placement for those pods they would compact, and the
    experiment would measure something else - so the layout has to survive, and every
    contradiction between layout and workload has to raise rather than be repaired.
    """
    NODES = [NodeSpec("s1", "cluster", "H20", 8., 8.), NodeSpec("s2", "cluster", "H20", 8., 8.)]

    def env(self, tasks, placement):
        return GpuSchedEnv(self.NODES, tasks, clock="fast", measurement_window=(0., 100.),
                           max_sim_time=500., initial_placement=placement)

    def test_prescribed_node_wins_over_the_policy(self):
        # best_fit would fill s1 first (both are empty, so it takes the lowest index);
        # the layout says s2, and the layout has to be what runs.
        tasks = [TaskSpec("p1", 4., 4., ("H20",), 0., 10., "Other")]
        env = self.env(tasks, {"p1": "s2"})
        env.run("best_fit")
        # The reset event is recorded after the layout is loaded, so it already shows
        # 4 GPU reserved. Asserting live free_gpu here would not work: clock="fast"
        # advances through the only completion event before anything can be observed.
        reset = [e for e in env.events if e["event"] == "reset"][0]
        self.assertEqual(4., reset["gpu_reserved"])
        self.assertEqual(1, reset["running"])
        self.assertEqual("s2", env.records["p1"]["server_id"])
        self.assertEqual(0., env.records["p1"]["start_time_sec"])
        self.assertEqual(0., env.records["p1"]["wait_sec"])

    def test_placed_pods_never_emit_an_arrival_event(self):
        tasks = [TaskSpec("p1", 4., 4., ("H20",), 0., 10., "Other"),
                 TaskSpec("p2", 1., 1., ("H20",), 5., 1., "Other")]
        env = self.env(tasks, {"p1": "s1"})
        env.run("first_fit")
        arrivals = [e for e in env.events if e["event"] == "arrival"]
        self.assertEqual(["p2"], [e["pod_id"] for e in arrivals])

    def test_layout_capacity_is_summed_across_pods_on_one_node(self):
        tasks = [TaskSpec("a", 5., 1., ("H20",), 0., 10., "Other"),
                 TaskSpec("b", 5., 1., ("H20",), 0., 10., "Other")]
        with self.assertRaises(ValueError):
            self.env(tasks, {"a": "s1", "b": "s1"}).run("first_fit")

    def test_contradictions_raise_instead_of_re_placing(self):
        late = [TaskSpec("p1", 4., 4., ("H20",), 30., 10., "Other")]
        with self.assertRaises(ValueError):
            self.env(late, {"p1": "s1"}).run("first_fit")
        with self.assertRaises(ValueError):
            self.env(late, {"p1": "nosuch"}).run("first_fit")
        with self.assertRaises(ValueError):
            self.env(late, {"nosuch": "s1"}).run("first_fit")

    def test_spot_tasks_from_the_layout_are_still_preemptible(self):
        # LP pods that arrived through the layout must join node_spot_pids, or an HP
        # task could never evict them and the benchmark's spot capacity would be
        # silently immune to preemption. One node, so there is nowhere to put HP
        # without preempting.
        tasks = [TaskSpec("lp1", 4., 4., ("H20",), 0., 40., "LP"),
                 TaskSpec("lp2", 4., 4., ("H20",), 0., 40., "LP"),
                 TaskSpec("hp", 4., 4., ("H20",), 1., 1., "HP")]
        env = GpuSchedEnv([NodeSpec("s1", "cluster", "H20", 8., 8.)], tasks, clock="fast",
                          measurement_window=(0., 100.), max_sim_time=500.,
                          initial_placement={"lp1": "s1", "lp2": "s1"})
        result = env.run("spotgpu")
        self.assertEqual(1, result["preemptions"])
        self.assertEqual(1., env.records["hp"]["start_time_sec"])
        self.assertGreater(env.records["lp1"]["preemption_count"] + env.records["lp2"]["preemption_count"], 0)


class ReactiveMigrationTests(unittest.TestCase):
    """The ported defragmentation policy, tested on a hand-built fragmented pool.

    Three nodes of 8 GPU each holding 4 GPU apiece: 12 GPU are free in total but no
    single node can host a 5-GPU request, which is fragmentation rather than a
    capacity shortage - the only condition the policy is allowed to react to.
    Evacuating one 4-GPU pod onto a neighbour frees a whole node.
    """
    NODES = [NodeSpec("n%d" % i, "cluster", "H20", 8., 192.) for i in range(1, 4)]
    PAUSE = 20. + 8.*4 + 0.1*4          # one 4-GPU / 4-core pod, her cost model

    def env(self, **options):
        pods = [TaskSpec("A", 4., 4., ("H20",), 0., 1000., "Other", {"migratable": True}),
                TaskSpec("B", 4., 4., ("H20",), 0., 1000., "Other", {"migratable": True}),
                TaskSpec("C", 4., 4., ("H20",), 0., 1000., "Other", {"migratable": True}),
                TaskSpec("big", 5., 5., ("H20",), 1., 10., "Other", {"migratable": False})]
        settings = dict(clock="fast", measurement_window=(0., 2000.), max_sim_time=4000.,
                        initial_placement={"A": "n1", "B": "n2", "C": "n3"})
        settings.update(options)
        return GpuSchedEnv(self.NODES, pods, **settings)

    def lp_env(self, **options):
        """The same fragmented pool, but with the resident pods marked LP.

        Every existing migration test uses priority "Other", which is precisely why the
        bookkeeping drift below was never observed: `node_lp_running` and
        `node_spot_pids` are only maintained for HP/LP pods.
        """
        pods = [TaskSpec("A", 4., 4., ("H20",), 0., 1000., "LP", {"migratable": True}),
                TaskSpec("B", 4., 4., ("H20",), 0., 1000., "LP", {"migratable": True}),
                TaskSpec("C", 4., 4., ("H20",), 0., 1000., "LP", {"migratable": True}),
                TaskSpec("big", 5., 5., ("H20",), 1., 10., "HP", {"migratable": False})]
        settings = dict(clock="fast", measurement_window=(0., 2000.), max_sim_time=4000.,
                        initial_placement={"A": "n1", "B": "n2", "C": "n3"})
        settings.update(options)
        return GpuSchedEnv(self.NODES, pods, **settings)

    def test_a_move_carries_the_priority_and_spot_bookkeeping_to_the_new_node(self):
        env = self.lp_env(migration="cheap_multi")
        env.run("first_fit")
        # Every pod has completed by now, so all three per-node counters must be back to
        # zero. They were not: a move left a phantom +1 on the source node and a -1 on the
        # destination, because only the resource ledger followed the pod.
        self.assertEqual([0, 0, 0], env.node_lp_running,
                         "non-zero after everything drained means the two nodes kept "
                         "different books")
        self.assertEqual([0, 0, 0], env.node_hp_running)
        self.assertEqual([set(), set(), set()], [set(s) for s in env.node_spot_pids])

    def assert_node_books_match_the_running_set(self, env):
        """The per-node priority and spot counters must agree with `env.running`.

        Asserted as a distribution rather than `[1, 1, 1]`: after a successful evacuation
        the three LP pods are deliberately not one-per-node any more, and the whole point
        is that the books follow the pods.
        """
        lp = [0]*len(self.NODES)
        hp = [0]*len(self.NODES)
        spot = [set() for _ in self.NODES]
        for pid, j in env.running.items():
            priority = env.task_priority.get(pid)
            if priority == "LP":
                lp[j] += 1
                if env.by_id[pid].gpu > 0:
                    spot[j].add(pid)
            elif priority == "HP":
                hp[j] += 1
        self.assertEqual(lp, env.node_lp_running, "node_lp_running disagrees with running")
        self.assertEqual(hp, env.node_hp_running, "node_hp_running disagrees with running")
        self.assertEqual([set(s) for s in spot], [set(s) for s in env.node_spot_pids],
                         "a preemptible pod is listed on a node it is not on")

    def test_spot_membership_moves_so_a_later_eviction_releases_the_right_node(self):
        # Mid-run, with A relocated and the others still up: the spot sets must name the
        # node each pod is actually on. Getting this wrong makes SpotGPU evict a phantom
        # and `release()` a node the pod left.
        env = self.lp_env(migration="cheap_multi", max_sim_time=400.)
        env.run("first_fit")
        self.assertGreater(env.migration_completed_moves, 0)
        self.assert_node_books_match_the_running_set(env)

    def test_preemption_and_migration_together_still_satisfy_the_ledger(self):
        # The documented CLI allows `--policy spotgpu --migration cheap_multi`. The
        # per-node books used to drift under exactly that combination. Nothing here
        # invents a combined semantics - admission is frozen during a plan by default,
        # and an in-flight pod is never a victim - the point is that the combination
        # must not corrupt the accounting. The same scenario under first_fit, where the
        # move does happen, is the test above.
        env = self.lp_env(migration="cheap_multi", max_sim_time=400.)
        result = env.run("spotgpu")
        # Not "a move must happen": with preemption switched on the HP request is usually
        # satisfied by an eviction, and `_advance_until_ready` tries preemption before
        # defragmentation, so the migration path can legitimately stay idle. What is not
        # acceptable is either mechanism leaving the per-node books inconsistent, which is
        # what this asserts. The same scenario under first_fit, where the move does happen,
        # is the test above.
        self.assertTrue(result["migration"]["count"] or result["preemptions"],
                        "neither mechanism acted, so the case tests nothing")
        self.assert_node_books_match_the_running_set(env)
        for j, node in enumerate(self.NODES):
            self.assertLessEqual(env.node_gpu_used[j], node.gpu_capacity+1e-9)
            self.assertGreaterEqual(env.free_gpu[j], -1e-9)
            self.assertGreaterEqual(env.free_cpu[j], -1e-9)

    def test_without_migration_the_fragmented_request_waits(self):
        env = self.env()
        result = env.run("first_fit")
        self.assertEqual(0, result["migration"]["count"])
        self.assertGreater(env.records["big"]["start_time_sec"], 900.)

    def test_one_move_evacuates_a_pod_and_unblocks_the_request(self):
        env = self.env(migration="one_move")
        result = env.run("first_fit")
        self.assertEqual(1, result["migration"]["count"])
        self.assertEqual(1, result["migration"]["completed_count"])
        self.assertEqual(4., result["migration"]["gpu_units_moved"])
        self.assertAlmostEqual(result["migration"]["gpu_pause_sec"], 4.*self.PAUSE, places=9)
        self.assertNotEqual("n1", env.records["A"]["server_id"])
        self.assertEqual("n1", env.records["big"]["server_id"])
        # The request starts one pause after the move begins, not instantly.
        self.assertAlmostEqual(1.+self.PAUSE, env.records["big"]["start_time_sec"], places=9)

    def test_pause_service_extends_the_moved_task(self):
        # The pause is service downtime: with pause_service the moved task finishes
        # one pause later than it would have; without it the pause is free.
        paused = self.env(migration="one_move", pause_service=True)
        paused.run("first_fit")
        free = self.env(migration="one_move", pause_service=False)
        free.run("first_fit")
        self.assertAlmostEqual(1000.+self.PAUSE, paused.records["A"]["finish_time_sec"], places=9)
        self.assertEqual(1000., free.records["A"]["finish_time_sec"])

    def test_reserve_destination_double_books_only_when_asked(self):
        # make-before-break charges the pod at both ends for the pause, so the
        # reservation integral is exactly one pod-times-pause of GPU-seconds higher
        # than break-before-make. That is the whole quantity M11 is about.
        held = self.env(migration="one_move").run("first_fit")
        loose = self.env(migration="one_move", reserve_destination=False).run("first_fit")
        self.assertGreater(held["reserved_gpu_ratio"], loose["reserved_gpu_ratio"])
        extra = (held["reserved_gpu_ratio"]-loose["reserved_gpu_ratio"]) \
            * held["gpu_capacity"] * held["measurement_denominator_sec"]
        self.assertAlmostEqual(extra, 4.*self.PAUSE, places=6)

    def test_a_pod_that_may_not_move_is_never_evacuated(self):
        env = self.env(migration="one_move")
        for task in env.tasks[:3]:
            task.metadata["migratable"] = "False"
        result = env.run("first_fit")
        self.assertEqual(0, result["migration"]["count"])

    def stage_move(self):
        # reset() under clock="fast" runs the whole episode, so the state has to be
        # restored by hand to the moment right after the layout was loaded: A running
        # on n1 with its finish still ahead.
        env = self.env()
        env.reset()
        env.current_time = 0.
        env.done = False
        env.running = {"A": 0, "B": 1, "C": 2}
        env.running_start = {pid: 0. for pid in env.running}
        env.finish_at = {pid: 1000. for pid in env.running}
        env.free_gpu, env.free_cpu = [4., 4., 4.], [188., 188., 188.]
        env.node_gpu_used, env.node_cpu_used, env.node_running_count = [4.]*3, [4.]*3, [1, 1, 1]
        env.waiting, env.waiting_set = OrderedDict(), set()
        env.heap, env.counter = [], 0
        for task in env.tasks:
            if task.pod_id in env.running:
                env.completion_counter[task.pod_id] = env._push(1000., 0, "complete", task.pod_id)
            else:
                env._push(task.submit_time, 1, "arrival", task.pod_id)
        env._check_resources()
        return env, env.by_id["A"], env._move_pause(env.by_id["A"])

    def plan(self, env, pod_id, source, destination, pause):
        return {"pod_id": "trigger", "moves": [{"pod_id": pod_id, "source": source,
                                                "destination": destination,
                                                "gpu": env.by_id[pod_id].gpu,
                                                "pause_sec": pause}]}

    def test_two_moves_of_one_pod_pay_two_pauses(self):
        # Regression for the residual that kept the two engines apart: the pause has
        # to extend the pod's *current* finish. Deriving it from
        # running_start + service - preserved instead drops the first pause when a
        # pod is evacuated a second time, and S1/cheap_multi showed three pods
        # finishing exactly one pause early against the benchmark engine.
        env, task, pause = self.stage_move()
        original = env.finish_at["A"]
        env.migration_operation = self.plan(env, "A", "n1", "n2", pause)
        env._begin_next_move()
        self.assertAlmostEqual(original+pause, env.finish_at["A"], places=9)
        env.current_time = pause
        env._complete_migration("A", env.completion_counter["A"])
        env.migration_operation = self.plan(env, "A", "n2", "n3", pause)
        env._begin_next_move()
        self.assertAlmostEqual(original+2*pause, env.finish_at["A"], places=9)

    def test_a_stale_move_event_is_recorded_only_once(self):
        # The heap keeps the move event even when the pod has been re-planned out of
        # it. Recording that stale event put two migration_complete rows in the trace
        # for one move.
        env, task, pause = self.stage_move()
        before = len(env.events)
        env.migration_operation = self.plan(env, "A", "n1", "n2", pause)
        env._begin_next_move()
        env.migration_current = None            # the move was invalidated
        env.current_time = pause
        env._process_due()
        self.assertEqual(0, sum(1 for e in env.events[before:] if e["event"] == "migration_complete"))

    def test_effective_allocation_discounts_the_paused_pod(self):
        # A_GPU is the column her benchmark reports alongside the reserved ratio, and
        # the two only differ while something is mid-migration. Under
        # make-before-break the ledger counts the paused pod twice, so the discount
        # must be two copies of it, not one.
        moved = self.env(migration="one_move").run("first_fit")
        still = self.env().run("first_fit")
        self.assertEqual(still["reserved_gpu_ratio"], still["effective_gpu_ratio"])
        self.assertLess(moved["effective_gpu_ratio"], moved["reserved_gpu_ratio"])
        self.assertEqual(1, moved["migration"]["count"])
        gap = (moved["reserved_gpu_ratio"]-moved["effective_gpu_ratio"]) \
            * moved["gpu_capacity"] * moved["measurement_denominator_sec"]
        self.assertAlmostEqual(gap, 2.*4.*self.PAUSE, places=6)






class SpotGpuKnobTests(unittest.TestCase):
    """The six SpotGPU knobs each change one thing, and the defaults change nothing.

    Staged directly on the ledger rather than through a full run: the prefix rule and
    the ranking terms are decisions made at one instant, and a hand-built state shows
    them without a scenario accidentally re-timing them.
    """
    def staged(self, **knobs):
        node = NodeSpec("n1", "cluster", "H20", 8., 8.)
        tasks = [TaskSpec("cheap_gpu", 6., 0.5, ("H20",), 0., 100., "LP"),
                 TaskSpec("cheap_cpu", 2., 7., ("H20",), 0., 100., "LP"),
                 TaskSpec("hp", 6., 3., ("H20",), 1., 1., "HP")]
        env = GpuSchedEnv([node], tasks, clock="fast", measurement_window=(0., 10.),
                          spotgpu=knobs or None)
        env.reset()
        env.current_time = 4.
        env.free_gpu, env.free_cpu = [0.], [0.5]
        env.node_gpu_used, env.node_cpu_used, env.node_running_count = [8.], [7.5], [2]
        env.running = {pid: 0 for pid in ("cheap_gpu", "cheap_cpu")}
        # cheap_gpu has run 0.1 s so its cost is 0.6; cheap_cpu has run 4 s so its cost
        # is 8.0. The cost order therefore puts the GPU-heavy, CPU-light victim first,
        # which is the only arrangement where the two prefix rules can disagree.
        env.running_start = {"cheap_gpu": 3.9, "cheap_cpu": 0.}
        env.node_spot_pids = [{"cheap_gpu", "cheap_cpu"}]
        return env

    def test_joint_prefix_keeps_evacuating_until_cpu_is_covered(self):
        env = self.staged()
        node, victims, cost = env._spotgpu_best_preemption(env.by_id["hp"])
        self.assertEqual(["cheap_gpu", "cheap_cpu"], victims)
        self.assertAlmostEqual(0.6 + 8.0, cost, places=9)

    def test_gpu_prefix_stops_at_the_gpu_covering_prefix_and_then_refuses(self):
        # OSDI'26 stops the prefix at sum(G_t) >= G_hp. Here that is one victim, whose
        # CPU does not add up, so the node is refused rather than over-subscribed.
        env = self.staged(victim_prefix="gpu")
        self.assertIsNone(env._spotgpu_best_preemption(env.by_id["hp"]))

    def test_prefix_rule_never_allows_a_cpu_oversubscribed_placement(self):
        for prefix in ("joint", "gpu"):
            env = self.staged(victim_prefix=prefix)
            candidate = env._spotgpu_best_preemption(env.by_id["hp"])
            if candidate:
                _, victims, _ = candidate
                freed = sum(env.by_id[pid].cpu for pid in victims)
                self.assertGreaterEqual(env.free_cpu[0] + freed, env.by_id["hp"].cpu - EPS)

    def stage_ranking(self, terms, free_gpu, lp_running, evictions, gpu=1., cpu=1.):
        tasks = [TaskSpec("t", gpu, cpu, ("H20",), 0., 5., "LP")]
        env = GpuSchedEnv([NodeSpec("n1", "cluster", "H20", 8., 8.),
                           NodeSpec("n2", "cluster", "H20", 8., 8.)], tasks, clock="fast",
                          measurement_window=(0., 10.), spotgpu={"placement_terms": terms})
        env.reset()
        env.free_gpu, env.free_cpu = list(free_gpu), list(free_gpu)
        env.node_lp_running, env.node_hp_running = list(lp_running), [0, 0]
        env.node_evictions = list(evictions)
        env.pending_id = "t"
        return env

    def test_cpu_aware_packing_breaks_a_gpu_tie_that_the_paper_rule_leaves(self):
        node = NodeSpec("n1", "cluster", "H20", 8., 8.)
        other = NodeSpec("n2", "cluster", "H20", 8., 8.)
        tasks = [TaskSpec("t", 4., 4., ("H20",), 0., 5., "Other")]
        # Both nodes leave exactly 4 GPU after placing, so the paper's packing term
        # ties and falls through to the node index. n2 has far less CPU headroom, and
        # only a CPU-aware term can see that.
        for mode, expected in (("none", 0), ("sum", 1), ("lexicographic", 1)):
            env = GpuSchedEnv([node, other], tasks, clock="fast", measurement_window=(0., 10.),
                              spotgpu={"placement_cpu_aware": mode, "placement_terms": ("packing",)})
            env.reset()
            env.free_gpu = [4., 4.]
            env.free_cpu = [8., 4.5]
            env.pending_id = "t"
            self.assertEqual(expected, env._spotgpu_choose_action(), mode)

    def test_placement_terms_are_a_prefix_of_the_paper_three(self):
        # Each criterion only ever breaks the tie the previous one left, exactly as
        # §4.2.1 words it, so dropping a trailing term can only make the choice less
        # specific - never different on a pool where the kept terms already decide.
        picked = self.stage_ranking(("packing",), [7., 8.], [0, 0], [0, 0])
        self.assertEqual(0, picked._spotgpu_choose_action())
        tied = self.stage_ranking(("packing",), [8., 8.], [0, 1], [0, 0])
        self.assertEqual(0, tied._spotgpu_choose_action())          # tie falls to index
        colocated = self.stage_ranking(("packing", "co_locate"), [8., 8.], [0, 1], [0, 0])
        self.assertEqual(1, colocated._spotgpu_choose_action())     # LP joins the LP node
        both = self.stage_ranking(("packing", "co_locate"), [8., 8.], [1, 1], [5, 0])
        self.assertEqual(0, both._spotgpu_choose_action())          # still tied, index wins
        historic = self.stage_ranking(("packing", "co_locate", "history"), [8., 8.], [1, 1], [5, 0])
        self.assertEqual(1, historic._spotgpu_choose_action())      # spot avoids evicted node

    def test_knob_validation_rejects_nonsense(self):
        node = NodeSpec("n1", "cluster", "H20", 8., 8.)
        tasks = [TaskSpec("t", 1., 1., ("H20",), 0., 5.)]
        for knobs in ({"victim_prefix": "either"}, {"victim_selection": "cheapest"},
                      {"placement_cpu_aware": "both"}, {"placement_terms": ("history",)},
                      {"unknown": 1}):
            with self.assertRaises(ValueError, msg=str(knobs)):
                GpuSchedEnv([node], tasks, clock="fast", spotgpu=knobs)


class MetricDefinitionTests(unittest.TestCase):
    """The repo has two p95 definitions on purpose; this keeps them from drifting.

    The engine and the archived preemption suite report lower nearest-rank percentiles.
    The defragmentation benchmark reports numpy's interpolated one, and matching its
    archived numbers requires matching its definition, not just its algorithm. Having
    both in one repository is a trap for anyone assembling a table from two sources, so
    the difference is a named function pair with a test rather than an accident.
    """
    def test_the_two_percentile_rules_differ_where_they_should(self):
        self.assertEqual(3, percentile_nearest_rank([1., 2., 3.], .95))
        self.assertAlmostEqual(2.9, percentile_linear([1., 2., 3.], .95), places=12)
        self.assertEqual(0., percentile_nearest_rank([], .95))

    def test_the_experiment_harness_agrees_with_the_engine_rule(self):
        # harness.py keeps its own copy because its archived results record
        # harness_sha256; changing the file would invalidate that provenance. This test
        # is what stops the copy from silently becoming a third definition.
        sys.path.insert(0, str(ROOT / "experiments/preemption_20261107"))
        from harness import percentile
        for values in ([1., 2., 3.], [5.], list(range(1, 101)), [3., 1., 2.]):
            ordered = sorted(values)
            for fraction in (.5, .95, .99):
                self.assertEqual(percentile(ordered, fraction),
                                 percentile_nearest_rank(ordered, fraction),
                                 "%s at %s" % (values, fraction))

    def test_interval_overlap_matches_the_engine_integrator(self):
        self.assertEqual(0., interval_overlap(0., 10., 20., 30.))
        self.assertEqual(5., interval_overlap(0., 10., 5., 20.))
        self.assertEqual(10., interval_overlap(0., 100., 20., 30.))


if __name__ == "__main__":
    unittest.main()
