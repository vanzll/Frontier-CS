from typing import Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "deadline_fallback_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self.args = args
        self._commit_on_demand: bool = False
        self._cached_done_sum: float = 0.0
        self._cached_done_len: int = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _sum_done_progress(self) -> float:
        lst = self.task_done_time
        if not lst:
            self._cached_done_sum = 0.0
            self._cached_done_len = 0
            return 0.0
        # Incrementally update if list only grows by appending.
        if len(lst) == self._cached_done_len:
            return self._cached_done_sum
        if len(lst) > self._cached_done_len:
            added = sum(lst[self._cached_done_len :])
            self._cached_done_sum += added
            self._cached_done_len = len(lst)
            return self._cached_done_sum
        # Fallback: list replaced or shrank; recompute
        s = sum(lst)
        self._cached_done_sum = s
        self._cached_done_len = len(lst)
        return s

    def _should_commit_on_demand(self, rem_wall: float, rem_work: float) -> bool:
        # Safety margin: account for one potential restart overhead when switching after a spot preemption
        # plus discretization cushion of two time steps.
        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        margin = float(self.restart_overhead) + 2.0 * gap
        return rem_wall <= rem_work + margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Compute remaining work and remaining wall time
        done = self._sum_done_progress()
        remaining_work = max(self.task_duration - done, 0.0)
        if remaining_work <= 0:
            self._commit_on_demand = False
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        rem_wall = float(self.deadline) - elapsed

        # Commit to on-demand if we are approaching the latest safe fallback time
        if not self._commit_on_demand and self._should_commit_on_demand(rem_wall, remaining_work):
            self._commit_on_demand = True

        if self._commit_on_demand:
            return ClusterType.ON_DEMAND

        # Prefer spot if available; otherwise wait (NONE) until fallback threshold triggers
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)