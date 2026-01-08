from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get current environment state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds if self.env.gap_seconds is not None else 60.0
        
        # Calculate remaining work and time
        completed_work = sum(self.task_done_time)
        remaining_work = self.task_duration - completed_work
        
        # If work is done, stop
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        remaining_time = self.deadline - elapsed
        
        # Calculate slack: the amount of time we can afford to waste (waiting or overheads)
        # Slack = Time Left - Work Left
        slack = remaining_time - remaining_work
        
        # Define Safety Threshold
        # We must use On-Demand if the slack drops below a critical level.
        # Components of the threshold:
        # 1. self.restart_overhead: If we switch to OD, we incur this penalty. We must have enough slack to absorb it.
        # 2. gap: In the worst case (waiting or preemption), we lose 'gap' seconds of slack in this step.
        # 3. Buffer (600s): A safety margin (10 mins) to account for floating point issues, trace granularity, or small delays.
        # Total Threshold = Overhead + Gap + Buffer
        panic_threshold = self.restart_overhead + gap + 600.0
        
        # Decision Logic
        
        # 1. Panic Mode: If slack is critically low, we must use On-Demand to guarantee meeting the deadline.
        # Even if Spot is available, the risk of preemption (and another overhead) is too high.
        if slack < panic_threshold:
            return ClusterType.ON_DEMAND
        
        # 2. Cost Saving Mode: If we have slack and Spot is available, use it.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Wait Mode: If Spot is unavailable but we have plenty of slack, wait (NONE) to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)