from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "Solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate the total work completed so far
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        # If the task is already completed, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE

        # Calculate time remaining until the hard deadline
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate the absolute minimum time required to finish if we switch to On-Demand now.
        # This includes the remaining work duration plus the restart overhead 
        # if we are not already running on an On-Demand instance.
        time_needed_on_od = remaining_work
        if last_cluster_type != ClusterType.ON_DEMAND:
            time_needed_on_od += self.restart_overhead
            
        # Define a safety buffer to account for simulation step granularity and floating point variations.
        # Using 2.0 steps ensures we switch slightly before the mathematical deadline.
        safety_buffer = self.env.gap_seconds * 2.0
        
        # Critical Path Check:
        # If the remaining time is close to the time needed for OD completion, 
        # we must switch to OD immediately to guarantee meeting the deadline.
        if time_remaining <= (time_needed_on_od + safety_buffer):
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Strategy:
        # If we have slack (time_remaining > time_needed), we prioritize cost.
        
        # If Spot instances are available, use them as they are cheaper.
        if has_spot:
            return ClusterType.SPOT
            
        # If Spot is unavailable but we still have slack, 
        # pause (NONE) to avoid the high cost of On-Demand, waiting for Spot to return.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)