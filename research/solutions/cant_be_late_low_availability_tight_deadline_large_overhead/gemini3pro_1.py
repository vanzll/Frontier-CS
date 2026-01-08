import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "DeadlineCostOptimizer"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        task_duration = self.task_duration
        work_done = sum(self.task_done_time)
        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        work_remaining = task_duration - work_done
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_remaining = deadline - elapsed

        # Calculate time needed to finish if we switch/continue with On-Demand
        if last_cluster_type == ClusterType.ON_DEMAND:
            # No overhead if continuing OD
            time_needed_od = work_remaining
        else:
            # Overhead incurred if switching from NONE or SPOT to OD
            time_needed_od = work_remaining + overhead

        # Slack represents how much time we can afford to waste (waiting or overheads)
        slack = time_remaining - time_needed_od
        
        # Safety margin: we must act before slack drops below the duration of a single timestep
        # Using 1.1x gap to handle potential floating point inaccuracies
        safety_threshold = gap * 1.1

        # CRITICAL: If slack is running out, force On-Demand to meet deadline
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND

        # OPTIMIZATION: If we have buffer, try to use Spot or wait
        if has_spot:
            # If we are currently on OD, avoid "thrashing" (switching to Spot and back)
            # unless we have enough buffer to pay the overheads.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # We risk paying overhead to start Spot, and if it fails, overhead to go back to OD.
                # Heuristic: Require buffer > 2 * overhead
                if slack > (2 * overhead + safety_threshold):
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # Currently NONE or SPOT, and Spot is available -> Use it
                return ClusterType.SPOT
        else:
            # Spot unavailable, but we have slack.
            # Wait (NONE) to save money rather than burning expensive OD hours.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)