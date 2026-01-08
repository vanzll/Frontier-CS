from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "JustInTimeOD"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # State retrieval
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate progress
        work_done = sum(self.task_done_time)
        total_work = self.task_duration
        remaining_work = max(0.0, total_work - work_done)
        
        # If work is done, do nothing (though environment should handle termination)
        if remaining_work <= 0:
            return ClusterType.NONE

        # Time remaining until deadline
        time_left = deadline - elapsed

        # Calculate the panic threshold (Latest Start Time for OD)
        # We need enough time to:
        # 1. Start the instance (incur overhead)
        # 2. Run the remaining work
        # 3. Account for the current scheduling gap/step
        # 4. A small safety buffer for simulation noise
        
        # Why assume overhead? 
        # Even if we are currently running, we must ensure we can finish if we are forced 
        # to switch (e.g., Spot preemption). If we are on OD, we stick to it.
        worst_case_time_needed = remaining_work + overhead
        
        # Buffer: at least 2 steps or 5 minutes, whichever is reasonable to prevent
        # floating point issues or step misalignments.
        safety_buffer = max(gap * 2.0, 300.0)
        
        # Logic:
        # If we are approaching the point of no return, force On-Demand.
        # This ensures we satisfy the hard deadline.
        if time_left <= (worst_case_time_needed + safety_buffer):
            return ClusterType.ON_DEMAND
        
        # If we have slack, prefer Spot to minimize cost.
        if has_spot:
            return ClusterType.SPOT
        
        # If Spot is unavailable and we have slack, wait (pause) to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)