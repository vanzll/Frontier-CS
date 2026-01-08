from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Fetch environment state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        work_done = sum(self.task_done_time)
        total_work = self.task_duration
        restart_overhead = self.restart_overhead
        gap = self.env.gap_seconds

        work_remaining = max(0.0, total_work - work_done)
        time_remaining = max(0.0, deadline - elapsed)

        # Calculate time required to finish using On-Demand (our fallback)
        # If we are not currently running On-Demand, we must account for the restart overhead
        # to spin it up.
        overhead_if_switch_to_od = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            overhead_if_switch_to_od = restart_overhead

        # Define safety buffer
        # We need to ensure we switch to On-Demand before it's mathematically impossible to finish.
        # We add 'gap' to ensure we don't miss the window between steps.
        # We add a fixed margin (5 minutes) for safety against small variations.
        safety_margin = 300.0
        buffer = gap + safety_margin

        # Panic Logic:
        # If remaining time is critically low, force On-Demand usage to guarantee completion.
        # Threshold = Work + Overhead (if needed) + Buffer
        if time_remaining < (work_remaining + overhead_if_switch_to_od + buffer):
            return ClusterType.ON_DEMAND

        # Cost Optimization Logic (when slack is sufficient):
        if has_spot:
            # Spot is available.
            # If we are currently on On-Demand, we should generally switch to Spot to save money.
            # However, if the remaining work is very short, the overhead of switching might
            # cost more (in time and risk) or negate savings.
            # Simple heuristic: Only stick with OD if work is trivial (< 1/2 overhead).
            if last_cluster_type == ClusterType.ON_DEMAND:
                if work_remaining < (0.5 * restart_overhead):
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT
        
        # Spot is unavailable, but we have enough slack (not in panic mode).
        # We pause (NONE) to avoid paying high On-Demand costs, waiting for Spot to return.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)