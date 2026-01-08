from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cbl_safe_threshold_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._committed_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already committed to on-demand, stay on it.
        if self._committed_od:
            return ClusterType.ON_DEMAND

        # Calculate remaining work and time.
        done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = max(self.task_duration - done, 0.0)

        # If work is done (or effectively done), no need to run further.
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        time_left = max(self.deadline - self.env.elapsed_seconds, 0.0)

        # Safety margin to account for restart overhead and discretization.
        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        safety_margin = self.restart_overhead + 2.0 * gap

        # Commit to on-demand if it's the latest safe time to ensure completion.
        if time_left <= remaining_work + safety_margin:
            self._committed_od = True
            return ClusterType.ON_DEMAND

        # Prefer spot when available before commitment; otherwise, wait.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)