import argparse
from typing import Any, Iterable

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


def _sum_task_done_time(done_list: Iterable[Any]) -> float:
    if not done_list:
        return 0.0
    total = 0.0
    try:
        # Fast path: list of numbers
        return float(sum(done_list))  # type: ignore[arg-type]
    except TypeError:
        pass
    for item in done_list:
        if isinstance(item, (int, float)):
            total += float(item)
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                a, b = item
                a_val = None
                b_val = None
                try:
                    a_val = float(a)
                except Exception:
                    pass
                try:
                    b_val = float(b)
                except Exception:
                    pass
                if a_val is not None and b_val is not None:
                    if b_val >= a_val:
                        total += (b_val - a_val)
                    elif b_val > 0:
                        total += b_val
            elif len(item) > 2:
                try:
                    d = float(item[-1])
                    if d > 0:
                        total += d
                except Exception:
                    pass
        else:
            d = getattr(item, "duration", None)
            try:
                if d is not None:
                    d = float(d)
                    if d > 0:
                        total += d
            except Exception:
                pass
    return total


class Solution(Strategy):
    NAME = "deadline_guard_spot_first"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal flags; avoid overriding __init__ to maintain compatibility
        if not hasattr(self, "_committed_od"):
            self._committed_od = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize internal state if not set
        if not hasattr(self, "_committed_od"):
            self._committed_od = False

        # Compute remaining work and time
        try:
            gap = float(getattr(self.env, "gap_seconds", 60.0))
        except Exception:
            gap = 60.0
        try:
            elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        except Exception:
            elapsed = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed
        time_left = max(deadline - elapsed, 0.0)

        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0
        done_seconds = _sum_task_done_time(getattr(self, "task_done_time", []))
        remaining_work = max(task_duration - done_seconds, 0.0)

        # If already completed, do nothing
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # If we've already committed to on-demand, continue with it
        if self._committed_od:
            return ClusterType.ON_DEMAND

        # Determine safety buffer and overhead
        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        # Fudge factor: cover decision granularity and potential instantaneous preemption
        # Use at least 2 steps plus small constant (5 min)
        fudge = max(2.0 * gap, 300.0)

        # Commit condition: ensure we can finish on OD including one restart overhead and buffer
        if time_left <= remaining_work + overhead + fudge:
            self._committed_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, prefer spot when available; wait if not
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)