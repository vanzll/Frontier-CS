from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLate_GreedySafe"

    def __init__(self, args):
        super().__init__(args)

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the cluster type for the next time step.
        Prioritizes safety (meeting deadline) then cost (using Spot).
        """
        # Current state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time required to finish if we switch to (or stay on) On-Demand NOW.
        # If we are not currently running On-Demand, we incur the restart overhead.
        overhead_cost = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            overhead_cost = self.restart_overhead
            
        time_needed_on_od = work_remaining + overhead_cost
        time_remaining_until_deadline = deadline - elapsed
        
        # Slack represents how much extra time we have before we MUST run On-Demand to finish.
        slack = time_remaining_until_deadline - time_needed_on_od
        
        # Safety Buffer:
        # We need a buffer to cover:
        # 1. The time gap until the next decision step
        # 2. A safety margin for quantization or minor variations
        # Buffer = 2 * gap (ensure we survive current and next step wait) + 5 minutes absolute padding
        buffer_seconds = (2.0 * gap) + 300.0
        
        # Decision Logic:
        
        # 1. Critical Safety Check
        # If slack is running out, we must use On-Demand immediately to guarantee completion.
        if slack < buffer_seconds:
            return ClusterType.ON_DEMAND
            
        # 2. Cost Optimization
        # If we have sufficient slack, we prioritize cost.
        if has_spot:
            # Spot is available and cheaper. Use it.
            return ClusterType.SPOT
        else:
            # Spot is unavailable, but we have plenty of slack.
            # We choose NONE (wait) instead of On-Demand to save money.
            # We only burn On-Demand if we are forced by the deadline (checked above).
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)