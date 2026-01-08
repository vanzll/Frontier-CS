import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self.prior_success: float = 6.0
        self.prior_total: float = 10.0
        self.total_steps: int = 0
        self.spot_available_steps: int = 0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.total_steps += 1
        if has_spot:
            self.spot_available_steps += 1

        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_time = self.env.elapsed_seconds
        time_left_to_deadline = self.deadline - elapsed_time

        safety_buffer = self.restart_overhead
        if time_left_to_deadline <= safety_buffer or work_remaining + safety_buffer >= time_left_to_deadline:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        p_hat = (self.spot_available_steps + self.prior_success) / (self.total_steps + self.prior_total)

        if time_left_to_deadline <= 1e-9:
            return ClusterType.ON_DEMAND
            
        required_rate = work_remaining / time_left_to_deadline
        
        if required_rate > p_hat:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)