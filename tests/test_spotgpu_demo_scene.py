"""The SpotGPU preemption demo must keep demonstrating preemption.

Scenario A has no priority_class at all, so the preemption path never fires there;
scenario F does, but at 119,495 tasks it is not something a visitor clicks first.
runs/spotgpu_demo_20261121 is the small scene that shows the feature, and these
tests pin the contrast that makes it worth having: same data, no preemption
without the SpotGPU policy and real preemption with it.
"""
import sys
import unittest
from pathlib import Path

from vmagent.gpu.compare import diff_summaries
from vmagent.gpu.dashboard import BUILTIN_DATASETS
from vmagent.gpu.environment import GpuSchedEnv, ROOT, load_nodes, load_workload
from vmagent.gpu.run import make_environment

DEMO = ROOT / "runs/spotgpu_demo_20261121/inputs"
NODES = DEMO / "nodes.csv"
TASKS = DEMO / "jobs_spotgpu_demo.csv"
# The window and cap used by the reproduction commands in the scene README. These
# pods arrive around 87,000 s, so a window starting at 0 would drain the scene
# correctly and still report 0% load; the tests below pin the window instead.
OPTIONS = dict(clock="fast", queue_policy="backfill", measurement_window=[86400., 129600.],
               max_sim_time=400000., checkpoint_interval=3600., cpu_background_fraction=0., seed=42)


def run(policy, **overrides):
    options = dict(OPTIONS)
    options.update(overrides)
    env, _ = make_environment(NODES, TASKS, **options)
    env.run(policy)
    return env


def waits(env, priority):
    return [env.records[task.pod_id]["wait_sec"] for task in env.tasks if task.priority == priority]


class BuiltinDatasetCatalogTests(unittest.TestCase):
    def test_every_builtin_dataset_file_exists(self):
        # A renamed input directory would otherwise leave a dead button in the UI.
        missing = [str(path) for entry in BUILTIN_DATASETS.values()
                   for path in (entry["nodes"], entry["tasks"]) if not Path(path).is_file()]
        self.assertEqual([], missing)

    def test_dataset_options_are_accepted_by_the_dashboard(self):
        allowed = ("policy", "clock", "speed", "queue_policy", "max_sim_time",
                   "cpu_background_fraction", "checkpoint_interval", "seed",
                   "measurement_start_seconds", "measurement_seconds")
        for dataset_id, entry in BUILTIN_DATASETS.items():
            unknown = sorted(set(entry["run"]) - set(allowed))
            self.assertEqual([], unknown, "{} sets unknown options".format(dataset_id))
            self.assertIn(entry["run"]["policy"], ("first_fit", "best_fit", "worst_fit", "random", "spotgpu"))
            self.assertTrue(entry["label"] and entry["note"], "{} needs UI text".format(dataset_id))


class SpotGpuPreemptionDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spotgpu, cls.first_fit = run("spotgpu"), run("first_fit")

    def test_scene_is_a_mixed_priority_saturated_pool(self):
        self.assertEqual(29, len(self.first_fit.tasks))
        self.assertEqual(24, sum(1 for t in self.first_fit.tasks if t.priority == "LP"))
        self.assertEqual(5, sum(1 for t in self.first_fit.tasks if t.priority == "HP"))
        self.assertEqual(24., sum(t.gpu for t in self.first_fit.tasks if t.priority == "LP"))
        self.assertEqual(24., self.first_fit.total_gpu)

    def test_spotgpu_preempts_and_clears_high_priority_waits(self):
        summary = self.spotgpu.summary()
        self.assertEqual(5, summary["preemptions"])
        self.assertGreater(summary["wasted_gpu_sec"], 0.)
        self.assertEqual(0., max(waits(self.spotgpu, "HP")))
        self.assertEqual(0, summary["rejected"])
        self.assertEqual(29, summary["completed"])

    def test_the_measurement_window_covers_the_whole_scene(self):
        # A window that misses every pod is the failure mode this scene had: the run
        # drains, the preemption counters are right, and all four load ratios say 0%.
        start, end = OPTIONS["measurement_window"]
        for env in (self.first_fit, self.spotgpu):
            self.assertGreaterEqual(min(task.submit_time for task in env.tasks), start)
            self.assertLessEqual(max(record["finish_time_sec"] for record in env.records.values()), end)
            summary = env.summary()
            self.assertAlmostEqual(summary["measurement_denominator_sec"], end-start)
            for key in ("offered_gpu_ratio", "reserved_gpu_ratio", "offered_cpu_ratio", "reserved_cpu_ratio"):
                self.assertGreater(summary[key], 0., key)

    def test_the_same_data_does_not_preempt_under_first_fit(self):
        summary = self.first_fit.summary()
        self.assertEqual(0, summary["preemptions"])
        self.assertEqual(0., summary["wasted_gpu_sec"])
        self.assertGreater(max(waits(self.first_fit, "HP")), 8000.)
        self.assertEqual(0., max(waits(self.first_fit, "LP")))

    def test_preemption_trades_low_priority_wait_for_high_priority_slo(self):
        spot, plain = self.spotgpu.summary(), self.first_fit.summary()
        self.assertLess(spot["preempted_task_count"], spot["task_count"])
        self.assertGreater(max(waits(self.spotgpu, "LP")), 0.)
        self.assertLess(max(waits(self.spotgpu, "HP")), max(waits(self.first_fit, "HP")))

    def test_wasted_work_shrinks_with_the_checkpoint_interval(self):
        # Wasted time is the run since the last checkpoint, so it is bounded by the
        # checkpoint interval. This is the scene's one real modelling assumption:
        # the trace has no checkpoint data, so the number moves with this parameter.
        hourly = self.spotgpu.summary()
        frequent = run("spotgpu", checkpoint_interval=60.).summary()
        self.assertLess(frequent["wasted_gpu_sec"], hourly["wasted_gpu_sec"])
        self.assertGreater(frequent["wasted_gpu_sec"], 0.)
        self.assertEqual(hourly["completed"], frequent["completed"])
        self.assertEqual(0, frequent["rejected"])


class SpotGpuVariantEquivalenceTests(unittest.TestCase):
    """The knob bundle must produce the same run as the harness subclass it replaces.

    tools/verify_spotgpu_variants.py does this on a 40,000-task slice; this is the
    fast version on the 29-task demo, which does exercise the differences (no_preempt
    preempts zero times, paper_baseline places elsewhere). Same claim, cheaper check.
    """
    def setUp(self):
        sys.path.insert(0, str(ROOT / "experiments/preemption_20261107"))

    def results(self, env, experiment=False):
        # The harness entry point is run_experiment(), which is what sets
        # preempt_enabled per variant; calling the base run() on it would silently
        # re-enable preemption and make no_preempt look like a divergence.
        summary = env.run_experiment() if experiment else env.run("spotgpu")
        return summary, list(env.records.values()), list(env.events)

    def test_engine_knobs_match_the_experiment_harness(self):
        from vmagent.gpu.spotgpu import VARIANTS
        from harness import ExperimentEnv
        nodes = load_nodes(NODES)
        tasks = load_workload(TASKS)[0]
        settings = dict(clock="fast", queue_policy="backfill", seed=42,
                        measurement_window=[86400., 129600.], max_sim_time=400000.)
        for name, knobs in sorted(VARIANTS.items()):
            reference = self.results(ExperimentEnv(nodes, list(tasks), variant=name, **settings),
                                     experiment=True)
            candidate = self.results(GpuSchedEnv(nodes, list(tasks), spotgpu=knobs, **settings))
            self.assertEqual(reference[1:], candidate[1:], "%s records/events diverged" % name)
            self.assertEqual(reference[0]["preemptions"], candidate[0]["preemptions"], name)
            self.assertEqual([], diff_summaries(reference[0], candidate[0]), name)

    def test_the_six_variants_are_not_all_the_same_policy(self):
        # If every knob bundle collapsed to one behaviour, the decomposition would be
        # decoration. The demo scene separates at least three of them.
        from vmagent.gpu.spotgpu import VARIANTS
        nodes = load_nodes(NODES)
        outcomes = {}
        for name in sorted(VARIANTS):
            env = GpuSchedEnv(nodes, list(load_workload(TASKS)[0]), clock="fast",
                              queue_policy="backfill", seed=42,
                              measurement_window=[86400., 129600.], max_sim_time=400000.,
                              spotgpu=VARIANTS[name])
            summary = env.run("spotgpu")
            outcomes[name] = (summary["preemptions"], round(summary["reserved_gpu_ratio"], 9),
                              round(summary["mean_wait_sec_started"], 3))
        self.assertEqual(0, outcomes["no_preempt"][0])
        self.assertEqual(5, outcomes["legacy_spotgpu"][0])
        self.assertGreater(len(set(outcomes.values())), 2, str(outcomes))


if __name__ == "__main__":
    unittest.main()
