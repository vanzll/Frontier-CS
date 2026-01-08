from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

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
        """
        # Retrieve current state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        task_duration = self.task_duration
        work_done = sum(self.task_done_time)
        
        # Calculate remaining work
        work_rem = max(0.0, task_duration - work_done)
        
        # If work is effectively finished, do nothing
        if work_rem <= 1e-6:
            return ClusterType.NONE
            
        # Calculate remaining time until deadline
        time_rem = max(0.0, deadline - elapsed)
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Determine overhead penalty if we were to switch to ON_DEMAND now.
        # If we are already running ON_DEMAND, we continue without new overhead.
        # Otherwise (SPOT or NONE), we assume we must incur restart overhead to run OD.
        switch_penalty = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            switch_penalty = overhead
            
        # Time required to finish if we commit to ON_DEMAND immediately
        time_needed_od = work_rem + switch_penalty
        
        # Calculate slack (time buffer available)
        slack = time_rem - time_needed_od
        
        # Define safety buffer.
        # We must act before slack becomes zero.
        # The buffer should be larger than the simulation step (gap) to allow safely waiting (NONE),
        # plus margin for overheads.
        # 1200 seconds (20 mins) covers the 3 min overhead multiple times and handles small gaps.
        # 1.5 * gap handles large time steps (e.g. 1 hour).
        safety_buffer = max(1.5 * gap, 1200.0)
        
        # Critical Logic: If slack is running out, force ON_DEMAND usage to guarantee completion.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # Optimization Logic: If we have plenty of slack, try to minimize cost.
        if has_spot:
            # Spot is available and cheap (and we have slack to handle preemption risk)
            return ClusterType.SPOT
        else:
            # Spot unavailable, but we have slack. Wait for Spot to save money.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)