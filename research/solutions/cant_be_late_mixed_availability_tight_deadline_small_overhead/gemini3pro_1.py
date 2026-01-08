from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # If work is effectively done, do nothing
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        
        # Calculate time needed to finish using On-Demand (guaranteed capacity).
        # We must add restart overhead if we are not currently running On-Demand,
        # as switching or starting incurs this cost.
        switch_cost = 0
        if last_cluster_type != ClusterType.ON_DEMAND:
            switch_cost = overhead
            
        time_needed_od = remaining_work + switch_cost
        
        # Safety buffer to handle simulation granularity and ensure we don't miss the deadline.
        # 1800 seconds (30 minutes) is conservative given the 4-hour slack,
        # minimizing the risk of the -100,000 penalty while allowing Spot usage.
        safety_buffer = 1800.0
        
        # Panic condition: If remaining time is close to the minimum needed for OD, force OD.
        if (deadline - elapsed) < (time_needed_od + safety_buffer):
            return ClusterType.ON_DEMAND
            
        # If we have slack and Spot is available, use Spot to minimize cost.
        if has_spot:
            # Optimization: If currently on OD, switching to Spot incurs overhead.
            # If remaining work is very small, the overhead cost might exceed the Spot savings.
            # Heuristic: Spot is ~1/3 cost of OD. Stay on OD if work < overhead / 2.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if remaining_work < overhead / 2.0:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT
            
        # If no Spot available and we have plenty of slack, wait (return NONE) to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)