from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "DeadlineAwareCostMinimizer"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        # Calculate time budget
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Calculate overhead cost if we force On-Demand now
        # If already OD, 0 overhead. If Spot/None, we pay restart penalty.
        od_overhead = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            od_overhead = self.restart_overhead
            
        # Time required to finish using OD (Safe Path)
        time_needed_od = work_remaining + od_overhead
        
        # Safety margin: account for discrete time steps
        # Use 2.1x gap to handle boundary conditions strictly and small floating point errors
        safety_buffer = 2.1 * self.env.gap_seconds
        
        # Calculate Slack: Extra time available beyond the safe path requirements
        slack = time_left - time_needed_od
        
        # 1. Panic Mode: Approaching deadline with minimal margin
        # Must use On-Demand to guarantee completion
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Cost Optimization
        if has_spot:
            # If we are currently on OD, we should only switch to Spot if we have
            # enough slack to pay the restart penalty and still maintain safety.
            # This prevents oscillating OD->Spot->OD at the critical boundary.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > (self.restart_overhead + safety_buffer):
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            # Otherwise (currently Spot or None), use Spot
            return ClusterType.SPOT
            
        # 3. Waiting Strategy
        # Spot unavailable, but we have slack. Wait (NONE) to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)