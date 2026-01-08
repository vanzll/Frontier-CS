from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate completed work and remaining work needed
        done = sum(self.task_done_time)
        needed = self.task_duration - done
        
        # If task is effectively complete, stop resources
        if needed <= 1e-6:
            return ClusterType.NONE

        # Calculate time remaining until the hard deadline
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Calculate 'buffer': the amount of slack time we have.
        # buffer = (Time Remaining) - (Time Needed if running on ideal reliable hardware)
        # If buffer is 0, we must run continuously on reliable hardware (OD) to finish exactly on time.
        buffer = time_left - needed
        
        # Determine safety threshold for switching to On-Demand.
        # We need to preserve enough buffer to handle:
        # 1. Restart overhead: If we switch to OD, we might pay initialization time.
        # 2. Simulation gap: The time step size, to avoid missing the decision window.
        # 3. Safety margin: To protect against floating point errors or trace anomalies.
        gap = self.env.gap_seconds
        
        # Threshold: 1.5x restart overhead + 2x gap size + 5 minutes padding
        # This ensures that even in worst-case transition scenarios, we finish before the deadline.
        safe_buffer = (self.restart_overhead * 1.5) + (gap * 2.0) + 300.0
        
        # Strategy Logic:
        # 1. Panic Mode: If slack is critically low, force OD usage to guarantee completion.
        if buffer < safe_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Economy Mode: If we have buffer, try to use Spot to save money.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Wait Mode: Spot is unavailable, but we have buffer.
        #    Wait (return NONE) to conserve money, consuming buffer time.
        #    We bet that Spot will return before buffer drops to safe_buffer.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)