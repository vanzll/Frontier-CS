from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        progress = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = self.task_duration - progress
        
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        # Calculate slack based on the conservative assumption that we might need 
        # to pay a restart overhead to finish (e.g. switching to OD).
        # We must ensure: remaining_time >= remaining_work + restart_overhead
        time_needed = remaining_work + self.restart_overhead
        slack = remaining_time - time_needed
        
        # Safety buffer definition:
        # 1. Overhead margin: 1.5x restart_overhead ensures we absorb potential preemption 
        #    delays or switch costs without dipping into negative slack.
        # 2. Gap margin: 2x gap_seconds ensures we don't overshoot the decision boundary 
        #    due to time step granularity.
        buffer = 1.5 * self.restart_overhead + 2.0 * self.env.gap_seconds
        
        # Panic condition: If slack is low, force On-Demand to guarantee completion.
        if slack < buffer:
            return ClusterType.ON_DEMAND
            
        # Cost optimization: If we have enough slack...
        if has_spot:
            # Use Spot if available (cheapest option).
            return ClusterType.SPOT
        else:
            # If Spot is unavailable but we have slack, wait (NONE).
            # Waiting saves money compared to running OD unnecessarily.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)