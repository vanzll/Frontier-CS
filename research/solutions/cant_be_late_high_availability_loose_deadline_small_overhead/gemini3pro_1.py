from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        # Calculate remaining work based on completed segments
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 1e-6:
            return ClusterType.NONE
            
        time_remaining = self.deadline - elapsed
        
        # Safety Calculation
        # Determine the latest safe time to switch to On-Demand to guarantee completion.
        # We assume the worst case where we need to pay restart overhead.
        # We add a buffer of 2 * gap_seconds to handle step discretization and ensure
        # we trigger the switch before it is too late.
        safety_buffer = self.restart_overhead + (2.0 * gap)
        required_time = work_remaining + safety_buffer
        
        # Panic Mode: If strictly required to meet deadline, force On-Demand
        if time_remaining <= required_time:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Mode
        # If we have slack, we prioritize cost.
        # Spot is cheapest ($0.97). NONE is free ($0). OD is expensive ($3.06).
        # If Spot is available, use it.
        # If Spot is unavailable, wait (NONE) rather than using expensive OD,
        # because we have plenty of slack (22 hours) and OD is 3x the price of Spot.
        
        if has_spot:
            return ClusterType.SPOT
            
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)