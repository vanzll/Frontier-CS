import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self.CRITICAL_CUSHION_MULTIPLIER = 2.0
        self.COMFORTABLE_CUSHION_MULTIPLIER = 5.0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_now = self.env.elapsed_seconds
        time_left_until_deadline = self.deadline - time_now

        slack = time_left_until_deadline - work_remaining

        cushion_critical = self.CRITICAL_CUSHION_MULTIPLIER * self.restart_overhead
        cushion_comfortable = self.COMFORTABLE_CUSHION_MULTIPLIER * self.restart_overhead

        if slack <= cushion_critical:
            return ClusterType.ON_DEMAND
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                if slack > cushion_comfortable:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)