from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Current state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate remaining work
        done_work = sum(self.task_done_time)
        remaining_work = self.task_duration - done_work
        
        # If work is done, stop
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        # Calculate time remaining until deadline
        time_left = deadline - elapsed
        
        # Safety Check Logic:
        # Determine the absolute latest time we must be running On-Demand to finish.
        # We assume worst-case: switching to On-Demand incurs 'overhead' (start-up time).
        # We need to guarantee that at the *next* time step, we still have enough time.
        # If we choose NONE or SPOT (and it fails) this step, we lose 'gap' seconds.
        # Requirement at next step: (time_left - gap) >= (remaining_work + overhead)
        # Threshold: time_left < remaining_work + overhead + gap
        
        # We use a multiplier on gap (1.5x) for robust safety against discrete time steps.
        # We include 'overhead' in the threshold even if currently on OD to prevent
        # unsafe switching to Spot when margin is thin.
        panic_threshold = remaining_work + overhead + (1.5 * gap)
        
        if time_left <= panic_threshold:
            return ClusterType.ON_DEMAND
            
        # If we have sufficient slack, prefer Spot to minimize cost
        if has_spot:
            return ClusterType.SPOT
            
        # If Spot is unavailable but we are still safe, pause to save money
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)