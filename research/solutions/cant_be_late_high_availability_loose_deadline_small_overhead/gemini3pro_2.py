from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work (total duration - work completed so far)
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done

        # If the task is already completed (or effectively so), do nothing
        if work_rem <= 0:
            return ClusterType.NONE

        # Calculate remaining time until the deadline
        time_rem = self.deadline - self.env.elapsed_seconds

        # Calculate the panic threshold.
        # We must switch to On-Demand if the remaining time is close to the minimum time required to finish.
        # Minimum time = Remaining Work + Restart Overhead (to spin up OD) + Safety Buffer.
        # We use a safety buffer of 2.0 * gap_seconds to account for discrete time steps.
        safety_buffer = self.restart_overhead + (2.0 * self.env.gap_seconds)
        panic_threshold = work_rem + safety_buffer

        # Rule 1: Deadline Guarantee (Panic Mode)
        # If we are approaching the point of no return, force On-Demand usage.
        # On-Demand is guaranteed available and never interrupted.
        if time_rem <= panic_threshold:
            return ClusterType.ON_DEMAND

        # Rule 2: Cost Minimization
        # If we have slack (not in panic mode), prioritize Spot instances to save cost.
        if has_spot:
            return ClusterType.SPOT
        
        # Rule 3: Wait for Spot
        # If Spot is unavailable but we still have slack, pause execution (NONE).
        # This incurs no cost and preserves the budget for when Spot returns or we hit panic mode.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)