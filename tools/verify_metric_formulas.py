"""Check the documented metric formulas against what the engine actually does.

This is the verification half of `01_现状梳理/调度策略全集与评价指标口径.md`: that file
states every metric as a formula, and a formula nobody evaluated is a hypothesis. Each
assertion recomputes a quantity from raw engine state by a route independent of
`metrics.py` and compares.

The scenarios cover paths the unit suite does not: a non-zero CPU background fraction
(the `+background` term in `reserved_cpu_ratio`), all three permanent-rejection reasons,
a task evicted twice (`wait_sec` subtracting work already executed), and a migration that
actually lands (the double-booking arithmetic inside `effective_gpu_ratio`).

One assertion deliberately pins behaviour that is arguably wrong rather than ideal -
`same_instant_place_then_evict` - because it is a real inflation of `preemptions` and
silence about it would be worse. See the note at that case.

    C:/Users/trw/anaconda3/python.exe tools/verify_metric_formulas.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vmagent.gpu.environment import GpuSchedEnv, NodeSpec, TaskSpec
from vmagent.gpu.metrics import integrate_events, run_segments, interval_overlap

checks = []


def check(name, documented, actual, tol=1e-9):
    ok = abs(documented-actual) <= tol*max(1.0, abs(documented), abs(actual))
    checks.append((name, documented, actual, ok))
    print("%-60s %s  documented=%.10g actual=%.10g" % (name, "OK  " if ok else "FAIL",
                                                       documented, actual))


def expect(name, condition, detail):
    checks.append((name, True, bool(condition), bool(condition)))
    print("%-60s %s  %s" % (name, "OK  " if condition else "FAIL", detail))


def spec(pid, gpu, cpu, submit=0., service=10., priority="Other", models=("H20",), **extra):
    return TaskSpec(pid, gpu, cpu, models, submit, service, priority, dict(extra))


def background_case():
    """`reserved_cpu_ratio` is documented as foreground_area/(total*denominator) + bg.

    Note what the event column does instead: `cpu_reserved_cores` already contains the
    background offset, and the integral stops at the last event rather than running to
    the window end. Mixing the two bases is how this check was wrong the first time it
    was written, so both quantities are stated separately here.
    """
    nodes = [NodeSpec("n1", "c", "H20", 8., 100.), NodeSpec("n2", "c", "H20", 8., 100.)]
    tasks = [spec("a", 2., 30.), spec("b", 2., 30., submit=1.)]
    env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="fcfs",
                      cpu_background_fraction=0.25, measurement_window=(0., 40.),
                      max_sim_time=1000.)
    summary = env.run("first_fit")
    horizon = env.events[-1]["time_sec"]           # 11.0: b finishes, nothing is left
    check("bg: foreground core-seconds actually reserved", 600., env.cpu_area)
    check("bg: event column integrates foreground+background to the LAST EVENT, not the window end",
          600. + 200.*0.25*horizon, integrate_events(env.events, "cpu_reserved_cores"))
    check("bg: windowed event integral equals the ledger's cpu_area once bg is removed",
          env.cpu_area,
          integrate_events(env.events, "cpu_reserved_cores", (0., 40.))
          - 200.*0.25*interval_overlap(0., 40., 0., horizon))
    check("bg: summary reserved_cpu_ratio = 600/(200*40) + 0.25", 0.325,
          summary["reserved_cpu_ratio"])
    check("bg: gpu area is 4 cards busy for 10 s each", 40., env.gpu_area)
    check("bg: offered is demand over the window, independent of placement",
          (2.*10. + 2.*10.)/(16.*40.), summary["offered_gpu_ratio"])


def rejection_case():
    nodes = [NodeSpec("n1", "c", "H20", 8., 100.), NodeSpec("n2", "c", "A100", 8., 100.)]
    tasks = [spec("too_big", 9., 1.), spec("no_model", 1., 1., models=("V100",)),
             spec("cpu_hog", 1., 500.), spec("ok", 1., 1.)]
    env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="fcfs",
                      measurement_window=(0., 40.), max_sim_time=1000.)
    env.run("first_fit")
    reasons = {pid: r["rejection_reason"] for pid, r in env.records.items()
               if r["status"] == "rejected"}
    for pid, expected in [("too_big", "gpu_request_exceeds_single_node"),
                          ("no_model", "gpu_model_unavailable"),
                          ("cpu_hog", "cpu_request_exceeds_single_node_after_background")]:
        expect("rejection taxonomy: %s" % pid, reasons.get(pid) == expected,
               "%s (got %s)" % (expected, reasons.get(pid)))
    # Rejected tasks must never touch the ledger. Asserting on gpu_request would be
    # meaningless - the request is nonzero by definition - so assert on what they
    # occupied instead: no start event, and no start time on the record.
    expect("a permanently impossible task never gets a start event",
           not any(e["pod_id"] in reasons and e["event"] == "start" for e in env.events),
           "%d rejected, none started" % len(reasons))
    expect("and its record keeps start_time_sec null",
           all(env.records[p]["start_time_sec"] is None for p in reasons),
           str({p: env.records[p]["start_time_sec"] for p in reasons}))


def preemption_case():
    """Two HP arrivals evict one spot task, and the second eviction costs nothing.

    The timeline is: spot runs [0,1), evicted for hp1; hp1 completes at t=2 releasing the
    node; the normal path re-admits spot (it is earlier in the FCFS queue) at t=2; the
    preemption path then evicts it in the same instant for hp2. `wasted_sec` is therefore
    1.0 and not 2.0, because the second eviction happened 0 s after that placement.

    Pinned deliberately: `preemptions` = 2 while only 1 s of work was lost. A count that
    includes same-instant churn overstates disruption relative to the thing it appears to
    measure, and scenario F's numbers carry it. Changing the engine would move an
    already-reported figure, so this is recorded as behaviour, not as an assertion of
    correctness.
    """
    nodes = [NodeSpec("n1", "c", "H20", 8., 8.)]
    tasks = [spec("spot", 8., 8., service=30., priority="LP"),
             spec("hp1", 8., 8., submit=1., service=1., priority="HP"),
             spec("hp2", 8., 8., submit=2., service=1., priority="HP")]
    env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="fcfs",
                      checkpoint_interval=3600., measurement_window=(0., 60.),
                      max_sim_time=1000.)
    summary = env.run("spotgpu")
    record = env.records["spot"]
    segments = run_segments(env.events)
    check("two evictions are counted", 2., float(summary["preemptions"]))
    check("the victim occupied three separate intervals", 3., float(len(segments["spot"])))
    check("but only 1 s of work was lost, because the second eviction was same-instant",
          1., record["wasted_sec"])
    check("summary wasted_gpu_sec weights the lost seconds by the GPU count", 8.,
          summary["wasted_gpu_sec"])
    check("wait_sec at the final placement excludes the 1 s already executed", 2.,
          record["wait_sec"])
    check("total occupancy is service plus the one lost second", 31.,
          sum(end-start for start, end, _ in segments["spot"]))
    # The published `running_total_sec` must be the same 31 s. It used to be 30 s minus
    # the first segment, because the completion branch read `start_time_sec` through an
    # `or` and a segment starting at t=0 is falsy.
    check("running_total_sec on the record equals the occupied time from events", 31.,
          env.records["spot"]["running_total_sec"])
    check("records are positionally aligned with the sorted task list - the basis of the"
          " wasted_gpu_sec zip", 1.,
          float(all(r["pod_id"] == t.pod_id for r, t in zip(env.records.values(), env.tasks))))
    same = [e["time_sec"] for e in env.events if e["event"] == "start" and e["pod_id"] == "spot"]
    evicted_at = [e["time_sec"] for e in env.events if e["event"] == "preempt" and e["pod_id"] == "spot"]
    expect("and the churn is visible as a start and a preempt at one instant",
           bool(set(same) & set(evicted_at)), "shared instants %s" % sorted(set(same) & set(evicted_at)))


def migration_case():
    """Six GPUs wanted, four free on each of two nodes: that is fragmentation, by definition.

    Layout is imposed through `initial_placement` rather than left to first_fit, because
    the point of the case is a starting state no single node can serve while the pool
    aggregate can - exactly the state a policy-driven placement would not produce.
    """
    nodes = [NodeSpec("n1", "c", "H20", 8., 40.), NodeSpec("n2", "c", "H20", 8., 40.),
             NodeSpec("n3", "c", "H20", 8., 40.)]
    tasks = [spec("a", 4., 4.), spec("b", 4., 4.), spec("d", 4., 4.), spec("e", 4., 4.),
             spec("wanted", 6., 1., submit=1., service=20.)]
    layout = {"a": "n1", "b": "n1", "d": "n2", "e": "n3"}
    options = dict(clock="fast", queue_policy="backfill", measurement_window=(0., 200.),
                   max_sim_time=5000., initial_placement=layout,
                   migration="cheap_multi", migration_max_plan_moves=2,
                   migration_max_total_moves=4, migration_base_sec=10.,
                   migration_per_gpu_sec=1., migration_per_cpu_core_sec=0.)
    env = GpuSchedEnv(nodes, tasks, **options)
    summary = env.run("first_fit")
    move = summary["migration"]
    # `_classify_block` describes the instant it is called, so asking it after the run is
    # meaningless - the node is empty again and it correctly answers "direct". What the
    # migration trigger actually used is the first observed class, recorded at the time.
    expect("the blocked request was classified as fragmentation when it was blocked",
           env.first_block_class.get("wanted") == "fragmentation",
           env.first_block_class.get("wanted"))
    expect("a plan was found and executed", move["count"] > 0,
           "count=%d completed=%d plans=%d" % (move["count"], move["completed_count"],
                                               move["plans_started"]))
    expect("the pod really moved to another node", env.records["d"]["server_id"] == "n3",
           "d now on %s" % env.records["d"]["server_id"])
    expect("the fragmented request got served", env.records["wanted"]["status"] == "completed",
           env.records["wanted"]["status"])
    check("pause booked at start matches base + per_gpu*G", 14., move["gpu_pause_sec"]/4.)
    if move["completed_count"] and not move["inflight_at_horizon"]:
        check("with nothing in flight, planned and accrued pause agree",
              move["gpu_pause_sec"], move["gpu_pause_accrued_sec"], tol=1e-6)
        # NOT "A_GPU converges back onto reserved". The integral over the pause is
        # history: at the horizon nothing is in flight, but the run still contains the
        # seconds during which the pod was charged twice. The real invariant is that the
        # whole gap between the two areas equals exactly that double charge, which is
        # what the next check asserts - and it lands on 112.0 GPU-seconds, i.e. 4 GPUs
        # x 14 s of pause x 2 bookings, to the last bit.
        check("reserved_gpu_area - effective_gpu_area is exactly pause x GPUs x 2 bookings",
              4.*14.*2., env.gpu_area - env.effective_gpu_area, tol=1e-6)
    return env, summary


def reservation_semantics_case():
    """The four switches are documented as changing who is charged during a pause.

    Under make-before-break a moving pod is on two nodes at once, so `reserved_gpu_ratio`
    includes it twice while `effective_gpu_ratio` removes both copies. Under
    break-before-make it occupies neither, so reserved is *low* during the pause and
    effective equals it. Any implementation that discounts one copy instead of two shows
    up here rather than in a paper.
    """
    results = {}
    # All four combinations, not just the two extremes: the documented claim is that the
    # pause charge depends on `counted` alone, and the only way to assert that is to
    # include the two `counted == 1` cases and require them to agree with each other.
    # This is exactly the 2x2 that the S1/first_fit/cheap_multi table reports as three
    # distinct reservation ratios out of four combinations.
    for label, reserve, release in (("make-before-break", True, "end"),
                                    ("source only (counted=1)", False, "end"),
                                    ("destination only (counted=1)", True, "start"),
                                    ("break-before-make", False, "start")):
        nodes = [NodeSpec("n1", "c", "H20", 8., 40.), NodeSpec("n2", "c", "H20", 8., 40.),
                 NodeSpec("n3", "c", "H20", 8., 40.)]
        tasks = [spec("a", 4., 4.), spec("b", 4., 4.), spec("d", 4., 4.), spec("e", 4., 4.),
                 spec("wanted", 6., 1., submit=1., service=20.)]
        env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="backfill",
                          measurement_window=(0., 200.), max_sim_time=5000.,
                          initial_placement={"a": "n1", "b": "n1", "d": "n2", "e": "n3"},
                          migration="cheap_multi", migration_max_plan_moves=2,
                          migration_max_total_moves=4, migration_base_sec=40.,
                          migration_per_gpu_sec=1., migration_per_cpu_core_sec=0.,
                          reserve_destination=reserve, release_source_when=release)
        summary = env.run("first_fit")
        results[label] = (summary, env)
        counted = int(reserve) + int(release == "end")
        # The documented identity, not a hand-wave about "lower": the entire difference
        # between the two areas has to be the pause, times the pod's GPUs, times however
        # many times the ledger was counting that pod.
        check("%s: reserved - effective area == pause x GPUs x %d" % (label, counted),
              summary["migration"]["gpu_pause_sec"]*counted,
              env.gpu_area - env.effective_gpu_area, tol=1e-6)
    single = [results["source only (counted=1)"][0]["reserved_gpu_ratio"],
              results["destination only (counted=1)"][0]["reserved_gpu_ratio"]]
    check("the two counted==1 switch combinations give the SAME reservation ratio",
          single[0], single[1], tol=0.)
    print("      counted=1 both ways: %.6f and %.6f"
          "  <- this is why the documented 2x2 has three values, not four"
          % (single[0], single[1]))
    print("      reserved  make=%.6f break=%.6f | effective make=%.6f break=%.6f"
          % (results["make-before-break"][0]["reserved_gpu_ratio"],
             results["break-before-make"][0]["reserved_gpu_ratio"],
             results["make-before-break"][0]["effective_gpu_ratio"],
             results["break-before-make"][0]["effective_gpu_ratio"]))
    print("      (break-before-make books nothing during the pause, so its reserved ratio "
          "is the LOWER one and equals its own A_GPU - the intuitive reading is the wrong "
          "one, which is why this is asserted as an identity rather than as an ordering)")


def no_migration_identity_case():
    """The one place where "A_GPU equals reserved" is actually a true statement.

    The doc's shorthand "when nothing is migrating the two are equal" is false as written
    for any run that *did* migrate, because the reported fields are integrals - see
    `migration_case`, where the horizon is clean and the two areas still differ by 112
    GPU-seconds. What is true, and worth pinning, is that with `migration="none"` the
    integrand is never discounted at all, so the two areas and both ratios are identical
    to the last bit for every placement and queue rule.
    """
    nodes = [NodeSpec("n1", "c", "H20", 8., 40.), NodeSpec("n2", "c", "H20", 8., 40.)]
    for queue in ("fcfs", "backfill"):
        for placement in ("first_fit", "best_fit", "spotgpu"):
            env = GpuSchedEnv(nodes, [spec("a", 4., 4.), spec("b", 4., 4.),
                                      spec("c", 6., 4., submit=1., service=20.)],
                              clock="fast", queue_policy=queue,
                              measurement_window=(0., 200.), max_sim_time=5000.)
            summary = env.run(placement)
            check("%s/%s: no migration means gpu_area == effective_gpu_area exactly"
                  % (placement, queue), env.gpu_area, env.effective_gpu_area, tol=0.)
            expect("%s/%s: and so are both ratios, bit for bit" % (placement, queue),
                   summary["reserved_gpu_ratio"] == summary["effective_gpu_ratio"]
                   and summary["reserved_gpu_ratio"] != 0.,
                   "reserved=%.12g effective=%.12g" % (summary["reserved_gpu_ratio"],
                                                       summary["effective_gpu_ratio"]))


def block_coverage_case():
    """`tasks_never_reaching_a_decision` mixes two different reasons; separate them.

    A pod that arrived with the input layout was placed by the *scenario*, not by the
    scheduler, so it never had a decision to reach. A pod still queued at the horizon was
    denied one. Reporting only the sum makes the first look like the second.
    """
    nodes = [NodeSpec("n1", "c", "H20", 8., 40.), NodeSpec("n2", "c", "H20", 8., 40.)]
    tasks = [spec("laid", 4., 4.), spec("late", 4., 4., submit=1e6, service=10.)]
    env = GpuSchedEnv(nodes, tasks, clock="fast", queue_policy="fcfs",
                      measurement_window=(0., 200.), max_sim_time=500.,
                      initial_placement={"laid": "n1"})
    summary = env.run("first_fit")
    block = summary["block_classification"]
    check("the laid-out pod is counted as placed by the input", 1,
          block["tasks_placed_by_input"])
    expect("the laid-out pod is in neither class of *decision* coverage",
           "laid" not in env.first_block_class, sorted(env.first_block_class))
    check("never-reaching = task_count - classified, exactly as documented",
          len(tasks)-block["tasks_classified"], block["tasks_never_reaching_a_decision"])
    print("      never_reaching=%d of which placed_by_input=%d, so %d were genuinely "
          "denied a decision" % (block["tasks_never_reaching_a_decision"],
                                 block["tasks_placed_by_input"],
                                 block["tasks_never_reaching_a_decision"]
                                 - block["tasks_placed_by_input"]))


def main():
    for case in (background_case, rejection_case, preemption_case, migration_case,
                 reservation_semantics_case, no_migration_identity_case,
                 block_coverage_case):
        print("\n--- %s" % case.__name__)
        case()
    failed = [name for name, _, _, ok in checks if not ok]
    print("\n%d checks, %d failed" % (len(checks), len(failed)))
    for name in failed:
        print("  FAILED: " + name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
