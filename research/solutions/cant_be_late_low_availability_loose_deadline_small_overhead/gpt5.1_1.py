from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_threshold_strategy"

    def __init__(self, args=None):
        super().__init__(args)
        self._commit_to_on_demand = False
        self._prev_elapsed = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        now = self.env.elapsed_seconds
        gap = self.env.gap_seconds

        # Reset internal state if this is a new episode (elapsed time went backwards or first call)
        if self._prev_elapsed is None or now < self._prev_elapsed:
            self._commit_to_on_demand = False
        self._prev_elapsed = now

        # Decide whether to commit to on-demand for the rest of the task
        if not self._commit_to_on_demand:
            # Continuous-time latest safe start for pure on-demand,
            # assuming worst-case: no progress so far and one restart overhead.
            safe_start = self.deadline - (self.task_duration + self.restart_overhead)

            if safe_start <= 0:
                # No slack: must always run on-demand to have any chance to finish.
                self._commit_to_on_demand = True
            else:
                # Account for discrete step size: commit up to one step earlier
                # to ensure we start on-demand no later than safe_start.
                commit_time = max(0.0, safe_start - gap)
                if now >= commit_time:
                    self._commit_to_on_demand = True

        if self._commit_to_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit phase: use spot when available, otherwise pause.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)