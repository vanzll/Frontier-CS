from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def __init__(self, args):
        try:
            super().__init__(args)
        except TypeError:
            super().__init__()
        self.args = args

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work
        # self.task_done_time is a list of completed segment durations
        completed_work = sum(self.task_done_time)
        remaining_work = self.task_duration - completed_work
        
        # If work is done, pause (though env should handle termination)
        if remaining_work <= 0:
            return ClusterType.NONE

        # Current context
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        time_left = deadline - elapsed
        gap = self.env.gap_seconds
        restart_overhead = self.restart_overhead

        # --- Deadline Safety Logic (Panic Mode) ---
        # We must ensure we have enough time to finish using On-Demand (reliable).
        # We assume worst-case: we might need to pay restart overhead to start/continue OD.
        # We also need a safety buffer to account for the decision granularity (gap) 
        # and floating point margins.
        # Condition: If we wait/use spot this step, will we still have enough time?
        # We need: time_left - gap >= remaining_work + restart_overhead
        
        # Buffer includes the current step (gap) + a safety margin (gap)
        safety_buffer = 2.0 * gap
        
        # The latest time we *must* be running On-Demand
        required_time = remaining_work + restart_overhead
        
        if time_left <= required_time + safety_buffer:
            return ClusterType.ON_DEMAND

        # --- Cost Optimization Logic ---
        # If we have slack (not in Panic Mode), prioritize cost.
        if has_spot:
            # Spot is cheapest and available
            return ClusterType.SPOT
        else:
            # Spot is unavailable, but we have slack.
            # Wait (NONE) to save money instead of burning expensive OD.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)