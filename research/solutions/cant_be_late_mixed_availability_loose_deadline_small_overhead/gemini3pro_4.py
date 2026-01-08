from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate progress and time
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - elapsed
        
        # Determine safety threshold
        # We must finish work before deadline.
        # Worst case: we need to use On-Demand, which might incur restart overhead.
        # We need a buffer to account for the discrete time steps (gap_seconds).
        # Buffer = 15 minutes (900s) constant + 2 time steps to ensure we switch in time.
        gap = self.env.gap_seconds
        safety_buffer = 900.0 + (2.0 * gap)
        
        min_time_needed = remaining_work + self.restart_overhead + safety_buffer
        
        # Decision Logic
        # 1. Panic Mode: If we are close to the point of no return, force On-Demand.
        if time_left <= min_time_needed:
            return ClusterType.ON_DEMAND
            
        # 2. Economy Mode: If we have slack, prioritize Spot to save cost.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. Wait Mode: If no Spot and plenty of slack, wait (pause) to avoid high OD cost.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)