import math
from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_threshold_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self.locked_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work(self) -> float:
        if hasattr(self, "task_done_time") and self.task_done_time:
            done = float(sum(self.task_done_time))
        else:
            done = 0.0
        remaining = float(self.task_duration) - done
        return max(0.0, remaining)

    def _should_lock_to_od(self, remaining_work: float, time_left: float) -> bool:
        # Safety margin to cover discretization and overhead inaccuracies
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        overhead = float(self.restart_overhead)
        # Margin: one gap + half overhead + small buffer
        safety_margin = gap + 0.5 * overhead + 120.0

        # If already on OD, no new overhead to continue
        od_overhead_now = 0.0 if self.env.cluster_type == ClusterType.ON_DEMAND else overhead

        # If continuing on OD from now, can we make the deadline?
        min_time_to_finish_on_od_now = remaining_work + od_overhead_now

        return (min_time_to_finish_on_od_now + safety_margin) >= time_left

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already committed to on-demand, never switch back
        if self.locked_to_od:
            return ClusterType.ON_DEMAND

        remaining = self._remaining_work()
        if remaining <= 0.0:
            return ClusterType.NONE

        time_left = float(self.deadline) - float(self.env.elapsed_seconds)
        if time_left <= 0.0:
            # Past deadline; stick to OD to minimize further penalty risk
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # If we're already on OD, keep it (avoid extra overhead)
        if self.env.cluster_type == ClusterType.ON_DEMAND:
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # Check if we must lock to OD now to safely finish
        if self._should_lock_to_od(remaining, time_left):
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # Prefer SPOT when available
        if has_spot:
            return ClusterType.SPOT

        # Spot not available: decide to wait or switch to OD
        gap = float(self.env.gap_seconds)
        overhead = float(self.restart_overhead)
        safety_margin = gap + 0.5 * overhead + 120.0

        time_left_after_wait = time_left - gap
        if time_left_after_wait <= 0.0:
            # No time to wait; must switch to OD now
            self.locked_to_od = True
            return ClusterType.ON_DEMAND

        # If we wait one more step, can we still finish on OD?
        if (remaining + overhead + safety_margin) <= time_left_after_wait:
            return ClusterType.NONE

        # Otherwise, switch to OD now
        self.locked_to_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)