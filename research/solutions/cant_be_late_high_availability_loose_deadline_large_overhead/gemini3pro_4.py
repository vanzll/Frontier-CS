from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed_seconds = self.env.elapsed_seconds
        gap_seconds = self.env.gap_seconds
        
        # Calculate remaining work based on completed segments
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        time_remaining = self.deadline - elapsed_seconds
        
        # Calculate the threshold for the "Danger Zone".
        # We must switch to On-Demand if the remaining time is close to the minimum required time.
        # Minimum required = Remaining Work + Restart Overhead (to spin up OD).
        # We add a buffer of 1.5 * gap_seconds to ensure we act before it's mathematically impossible.
        # This guarantees we finish before the deadline.
        
        safety_buffer = 1.5 * gap_seconds
        min_required_time = remaining_work + self.restart_overhead + safety_buffer
        
        if time_remaining <= min_required_time:
            # We are running out of slack -> Must use On-Demand to guarantee completion
            return ClusterType.ON_DEMAND
            
        # If we have slack:
        if has_spot:
            # Spot is available and cheaper -> Use Spot
            return ClusterType.SPOT
        else:
            # Spot unavailable but we have slack -> Wait (pause) to save money
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)