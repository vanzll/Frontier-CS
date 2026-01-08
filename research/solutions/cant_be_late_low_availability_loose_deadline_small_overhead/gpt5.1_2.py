from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_hedged_v1"

    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def solve(self, spec_path: str):
        # Optional: could read spec_path for configuration.
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If task is already complete, avoid incurring any more cost.
        task_done = sum(self.task_done_time) if self.task_done_time else 0.0
        if task_done >= self.task_duration:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        restart_overhead = self.restart_overhead

        time_left = max(0.0, deadline - elapsed)
        remaining = max(0.0, self.task_duration - task_done)
        slack = time_left - remaining

        # Safety margin: enough slack to pay at least one restart overhead
        # plus extra to account for discrete time steps.
        panic_slack = restart_overhead + 2.0 * gap

        # In panic region, commit to on-demand and never return to spot.
        if slack <= panic_slack:
            return ClusterType.ON_DEMAND

        # Outside panic region: use spot when available, otherwise wait.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)