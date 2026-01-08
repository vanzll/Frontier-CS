from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "OptimalSlackStrategy"

    def __init__(self, args):
        self.args = args

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Retrieve environment state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        
        # Calculate remaining work
        # task_done_time is a list of completed work segments
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_remaining = max(0.0, self.task_duration - work_done)
        
        # If task is completed (with small epsilon for float precision), stop.
        if work_remaining < 1e-4:
            return ClusterType.NONE
            
        time_remaining = max(0.0, deadline - elapsed)
        
        # Calculate time required to finish if we use On-Demand (safe baseline)
        # If we are already on OD, we assume continuity (no new overhead).
        # If we are on Spot or None, we must pay overhead to switch/start OD.
        if last_cluster_type == ClusterType.ON_DEMAND:
            time_needed_od = work_remaining
        else:
            time_needed_od = work_remaining + overhead
            
        # Calculate Slack: Time we can afford to waste (wait or overhead) 
        # before we are forced to use OD to meet the deadline.
        slack = time_remaining - time_needed_od
        
        # Define Safety Buffer
        # If we choose NONE this step, slack decreases by 'gap'.
        # We must switch to OD before slack drops below zero.
        # We use 1.5 * gap + small constant to handle step boundaries safely.
        safety_buffer = 1.5 * gap + 5.0
        
        # Decision Logic:
        # 1. If slack is critically low, force OD to guarantee deadline.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. If we have slack, prefer Spot (Cheapest option).
        if has_spot:
            return ClusterType.SPOT
            
        # 3. If no Spot and we have slack, wait (Free).
        # This burns slack but saves money compared to premature OD.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)