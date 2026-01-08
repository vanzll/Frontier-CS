from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - elapsed
        
        # Calculate slack: time available minus time required for work (excluding overheads)
        slack = time_remaining - work_remaining
        
        overhead = self.restart_overhead
        
        # Buffer to handle step discretization and small simulation variances.
        # We need to switch before we hit the exact limit to be safe.
        padding = max(2.0 * gap, 60.0)
        
        # 1. Deadline Safety (Panic Mode)
        # Calculate cost to switch to OD now. 
        # If we are already on OD, cost is 0. If not, we pay restart overhead.
        switch_cost = overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0
        
        # If remaining slack is close to the cost required to stabilize on OD,
        # we must switch immediately to guarantee meeting the deadline.
        if slack <= switch_cost + padding:
            return ClusterType.ON_DEMAND
            
        # 2. Hysteresis (Sticky OD)
        # If we are already on OD (likely due to previous low slack), 
        # we should stay on OD unless we have regained significant slack.
        # This prevents costly oscillation between Spot and OD.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # To leave OD, we need enough slack to:
            # - Pay overhead to start Spot (overhead)
            # - Have enough reserve to switch back to OD later if Spot fails (overhead + padding)
            min_slack_to_leave = 2.0 * overhead + padding
            
            if has_spot and slack > min_slack_to_leave:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
                
        # 3. Standard Strategy
        # If we have slack and aren't forced to OD:
        # Use Spot if available.
        # If Spot is unavailable, return NONE (wait) to save money, since we have slack.
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)