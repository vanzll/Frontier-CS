from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_sched_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self.lock_to_od = False
        self.spot_availability_ema = 0.6
        self._ema_alpha = 0.05

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _work_remaining(self) -> float:
        try:
            done = sum(self.task_done_time) if self.task_done_time else 0.0
        except Exception:
            done = 0.0
        remaining = self.task_duration - done
        return remaining if remaining > 0.0 else 0.0

    def _time_remaining(self) -> float:
        tr = self.deadline - self.env.elapsed_seconds
        return tr if tr > 0.0 else 0.0

    def _commit_to_od(self, last_cluster_type: ClusterType) -> bool:
        if self.lock_to_od:
            return True

        work_remaining = self._work_remaining()
        if work_remaining <= 0.0:
            return False

        time_remaining = self._time_remaining()
        gap = getattr(self.env, "gap_seconds", 60.0)
        overhead_if_switch = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead

        # Slack represents how much extra wall-time we have if we switched to OD right now.
        slack = time_remaining - (work_remaining + overhead_if_switch)

        # Commit threshold: ensure we can tolerate one time-step granularity plus one restart overhead.
        # This guarantees feasibility after committing to OD, even with immediate preemption.
        commit_threshold = self.restart_overhead + gap

        return slack <= commit_threshold

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update spot availability estimate (not strictly required for the core policy)
        self.spot_availability_ema = (1 - self._ema_alpha) * self.spot_availability_ema + self._ema_alpha * (1.0 if has_spot else 0.0)

        work_remaining = self._work_remaining()
        if work_remaining <= 0.0:
            return ClusterType.NONE

        if self._commit_to_od(last_cluster_type):
            self.lock_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and we still have enough slack; wait to save cost.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)