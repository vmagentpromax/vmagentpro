"""The simulation clock: event heap, time advance, and the two clock modes.

`fast` advances straight to the next event, so results are exact and reproducible.
`realtime` advances by wall clock scaled by `speed`, which is what makes the
dashboard watchable and also why its `sim_time_sec` drifts with machine load:
thread wake-up and the cost of the scheduling computation both consume simulated
time. Nothing may assume the two modes agree to the last digit.

Tie-breaking is a semantic choice, not an accident: completion events are pushed
with priority 0 and arrivals with priority 1, so at the same simulated instant a
task releases its resources before another is admitted. `_process_due` pops in that
order and the heap tuple is (time, priority, counter, kind, pod_id).
"""
import heapq
import time

from .contract import EPS


class ClockMixin:
    def _push(self, when, priority, kind, pod_id):
        self.counter += 1
        heapq.heappush(self.heap, (when, priority, self.counter, kind, pod_id))
        return self.counter

    def _move_to(self, target):
        target = min(target, self.max_sim_time)
        if self.clock == "realtime":
            while not self.stop_requested.is_set():
                delay = (target-(time.monotonic()-self.wall_start)*self.speed)/self.speed
                if delay <= 0:
                    break
                self.stop_requested.wait(min(delay, .1))
            elapsed = (time.monotonic()-self.wall_start)*self.speed
            actual = min(self.max_sim_time, max(self.current_time if self.stop_requested.is_set() else target, elapsed))
        else:
            actual = self.current_time if self.stop_requested.is_set() else target
        with self.lock:
            self._integrate_to(max(self.current_time, actual))

    def _advance_until_ready(self):
        while True:
            with self.lock:
                if self.stop_requested.is_set():
                    self._finish("cancelled")
                if self.done:
                    return
                self._process_due()
                if self.current_time >= self.max_sim_time:
                    self._finish("drained" if not self.heap and not self.waiting and not self.running else "max_sim_time")
                    return
                self.pending_id = None if self.blocked_by_migration() else self._next_ready()
                if self.pending_id is not None:
                    # Reaching a decision point means the task fits now. Record
                    # "direct" only if it was never seen blocked, so a task that
                    # waited and then started keeps the reason it waited for.
                    self.first_block_class.setdefault(self.pending_id, "direct")
                    return
                # `blocked_by_migration` is the documented "no admissions while a plan is
                # executing" freeze, and a preemption-driven placement is an admission:
                # this path used to bypass it entirely, so the engine could evict into
                # capacity that the plan had already reserved at the destination. That is
                # not a new migration/preemption semantics - the freeze is off by design
                # when `block_admission_during_plan=False` - it is the guard respecting
                # the one switch it was given.
                if (self.preempt_enabled and not self.blocked_by_migration()
                        and self._try_preempt()):
                    continue
                if self.try_defragment():
                    continue
                if not self.heap:
                    self._finish("drained" if not self.waiting and not self.running else "blocked")
                    return
                target = self.heap[0][0]
            self._move_to(target)

    def _finish(self, reason):
        if not self.done:
            self.done, self.end_reason, self.pending_id = True, reason, None
            self.wall_end = time.monotonic()
            self._record(reason)
