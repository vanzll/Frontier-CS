import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.in_overhead = False
        self.overhead_end = 0.0
        self.spot_available_history = []

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        step_size = self.env.gap_seconds
        
        # Track spot availability for future prediction
        self.spot_available_history.append(has_spot)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)

        # Update overhead state
        if self.in_overhead and current_time >= self.overhead_end:
            self.in_overhead = False

        # Check if we just got preempted
        if (last_cluster_type == ClusterType.SPOT and not has_spot and 
            not self.in_overhead):
            self.in_overhead = True
            self.overhead_end = current_time + self.restart_overhead

        # If in overhead, wait
        if self.in_overhead:
            return ClusterType.NONE

        # Calculate progress and deadlines
        total_work_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = self.task_duration - total_work_done
        time_left = self.deadline - current_time
        
        # If work is done, stop
        if remaining_work <= 0:
            return ClusterType.NONE

        # Calculate conservative estimates
        time_needed_od = remaining_work
        time_needed_spot = remaining_work + self.restart_overhead * 2
        
        # Emergency: must use on-demand
        if time_left < time_needed_od:
            return ClusterType.ON_DEMAND
        
        # Calculate spot reliability
        spot_reliability = sum(self.spot_available_history) / max(len(self.spot_available_history), 1)
        
        # Determine risk level
        slack_ratio = (time_left - time_needed_od) / self.restart_overhead
        
        # Strategy selection
        if has_spot:
            if slack_ratio > 3.0 and spot_reliability > 0.6:
                return ClusterType.SPOT
            elif slack_ratio > 1.5 and spot_reliability > 0.5:
                return ClusterType.SPOT
            elif slack_ratio > 0.5 and time_left > time_needed_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            if slack_ratio > 2.0:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)