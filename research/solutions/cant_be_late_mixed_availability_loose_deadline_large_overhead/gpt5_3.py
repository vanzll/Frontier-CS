from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_threshold_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._commit_to_od = False
        self._epsilon = 1e-6
        self._fudge_multiplier = getattr(args, "fudge_multiplier", 1.0)
        self._extra_buffer_seconds = getattr(args, "extra_buffer_seconds", 0.0)

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work(self) -> float:
        done = 0.0
        if hasattr(self, "task_done_time") and self.task_done_time:
            try:
                done = float(sum(self.task_done_time))
            except Exception:
                done = 0.0
        total = getattr(self, "task_duration", 0.0) or 0.0
        remaining = max(0.0, total - done)
        return remaining

    def _should_commit_to_od(self) -> bool:
        # Compute if we must begin (or continue) on-demand to guarantee completion by deadline
        remaining = self._remaining_work()
        if remaining <= self._epsilon:
            return False

        elapsed = getattr(self.env, "elapsed_seconds", 0.0) or 0.0
        deadline = getattr(self, "deadline", 0.0) or 0.0
        slack = max(0.0, deadline - elapsed)

        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        restart_overhead = getattr(self, "restart_overhead", 0.0) or 0.0

        # Safety margin: at least gap to handle discretization; allow tuning via multiplier and extra buffer
        fudge = max(gap, 0.0) * max(self._fudge_multiplier, 0.0) + max(self._extra_buffer_seconds, 0.0)

        # Worst-case: if we need to start OD from current state, we should budget exactly one restart overhead.
        # This accounts for potential preemption just before switching (bounded by gap via fudge).
        needed_time_if_od = remaining + restart_overhead + fudge

        return slack <= needed_time_if_od

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        remaining = self._remaining_work()
        if remaining <= self._epsilon:
            return ClusterType.NONE

        # Once committed to OD, stay on OD to avoid further restart overheads and risk.
        if not self._commit_to_od:
            if self._should_commit_to_od():
                self._commit_to_od = True

        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        # Not committed to OD yet:
        # Prefer SPOT when available; otherwise, wait (NONE) until we must commit to OD.
        if has_spot:
            return ClusterType.SPOT

        # Spot not available; if we still have sufficient slack, wait to minimize cost.
        # Re-evaluate commitment next step.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        # Optional tunable parameters
        parser.add_argument("--fudge_multiplier", type=float, default=1.0)
        parser.add_argument("--extra_buffer_seconds", type=float, default=0.0)
        args, _ = parser.parse_known_args()
        return cls(args)