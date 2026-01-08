from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CostOptimizedStrategy"

    def __init__(self, args):
        super().__init__(args)

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        Args:
            last_cluster_type: The cluster type used in the previous step
            has_spot: Whether spot instances are available this step

        Returns:
            ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Retrieve environment state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        restart_overhead = self.restart_overhead
        
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time needed to finish if we commit to On-Demand NOW
        # If we are already on OD, we don't incur restart overhead to continue/start OD
        # If we are on Spot or None, we incur overhead to switch/start OD
        penalty = 0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead
        time_needed_od = work_remaining + penalty
        
        # Safety buffer to account for simulation steps and small delays
        # Use at least 2 simulation steps or 15 minutes (900s), whichever is larger
        buffer = max(2 * gap, 900.0)
        
        # Panic Threshold: The latest elapsed time we can theoretically be at and still finish on OD
        panic_threshold = deadline - time_needed_od - buffer
        
        # 1. Panic Mode: If we are close to the point of no return, force On-Demand
        if elapsed >= panic_threshold:
            return ClusterType.ON_DEMAND
            
        # 2. Opportunity Mode: If we have slack, prefer Spot to save money
        if has_spot:
            # If we are currently on OD, only switch to Spot if we have significant slack.
            # Switching incurs overhead (wasted time) and risk.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Calculate slack relative to the panic threshold
                slack = panic_threshold - elapsed
                # Require slack > 3 * overhead to justify switching back to Spot
                # This prevents rapid flapping between OD and Spot near the deadline
                if slack > 3.0 * restart_overhead:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            # If currently Spot or None, always take Spot if available
            return ClusterType.SPOT
            
        # 3. Wait Mode: No Spot, but we have slack (not in panic).
        # Wait for Spot to become available to avoid high OD costs.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)