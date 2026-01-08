from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args=None):
        super().__init__(args)
        self.use_only_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _estimate_remaining_work(self) -> float:
        """Estimate remaining work time in seconds."""
        gap = getattr(self.env, "gap_seconds", 0.0)
        task_duration = getattr(self, "task_duration", 0.0)
        done_list = getattr(self, "task_done_time", None)
        if gap <= 0 or done_list is None:
            # Fallback: assume no progress
            return float(task_duration)
        completed_segments = len(done_list)
        completed_work = completed_segments * gap
        remaining = task_duration - completed_work
        if remaining < 0:
            remaining = 0.0
        return remaining

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, keep using it.
        if self.use_only_on_demand:
            return ClusterType.ON_DEMAND

        remaining_work = self._estimate_remaining_work()

        # If job is effectively done, stop using any instances.
        if remaining_work <= 0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        gap = self.env.gap_seconds
        restart_overhead = self.restart_overhead

        # If time is up but work remains, best effort: use on-demand.
        if time_left <= 0:
            self.use_only_on_demand = True
            return ClusterType.ON_DEMAND

        # Decide whether it's still safe to continue exploring (SPOT or NONE)
        # for one more step, assuming worst case: no progress this step and a
        # full restart_overhead still to pay when we finally commit to OD.
        #
        # Safe explore condition:
        #   time_left - gap >= remaining_work + restart_overhead
        # <=> time_left >= remaining_work + restart_overhead + gap
        if time_left < remaining_work + restart_overhead + gap:
            # Not enough slack to risk another explore step; lock into OD.
            self.use_only_on_demand = True
            return ClusterType.ON_DEMAND

        # We have enough slack to explore cheaper options.
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot unavailable; it's safe to idle this step to save cost.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)