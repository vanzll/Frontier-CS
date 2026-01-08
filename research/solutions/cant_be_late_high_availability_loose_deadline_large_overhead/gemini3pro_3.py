from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "OptimalSlackStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Retrieve current simulation state
        elapsed = self.env.elapsed_seconds
        progress = sum(self.task_done_time)
        remaining_work = self.task_duration - progress
        time_until_deadline = self.deadline - elapsed
        
        # Calculate Slack:
        # Slack represents the time budget we can afford to lose (via waiting or overheads)
        # while still strictly meeting the deadline if we were to switch to perfect execution immediately.
        slack = time_until_deadline - remaining_work
        
        # Determine Safety Buffer:
        # We must reserve time for:
        # 1. Restart overhead (self.restart_overhead) if we switch to On-Demand or recover from preemption.
        # 2. The simulation time step (gap_seconds) to ensure we don't detect the threshold too late.
        # 3. General safety margin.
        # We use a conservative buffer of 3x overhead + 2x time step.
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        safe_buffer = (3.0 * overhead) + (2.0 * gap)
        
        # Strategy Implementation:
        
        # 1. Critical Phase: If slack falls below the safe buffer, we are at risk of missing the deadline.
        # We must use On-Demand instances which are guaranteed to complete the work (never interrupted).
        if slack < safe_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Economic Phase: We have sufficient slack to optimize for cost.
        # If Spot instances (cheaper) are available, use them.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Waiting Phase: Spot is unavailable, but we have plenty of slack.
        # Instead of burning money on On-Demand ($3.06/hr), we wait (NONE, $0/hr)
        # effectively converting our time slack into cost savings.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)