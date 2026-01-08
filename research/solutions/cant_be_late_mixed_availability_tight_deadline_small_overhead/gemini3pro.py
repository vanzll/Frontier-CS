from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CostOptimizedSolution"

    def __init__(self, args):
        super().__init__(args)

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Get current state from environment
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        # Calculate progress
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # Check if task is already complete
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        # Calculate time metrics
        time_remaining = self.deadline - elapsed
        slack = time_remaining - work_remaining

        # Determine Safety Threshold
        # We must preserve enough slack to successfully finish on On-Demand if Spot fails.
        # 1. If we choose NONE this step (waiting for Spot), we consume 'gap' seconds of slack.
        # 2. If we switch to ON_DEMAND at the next step, we consume 'overhead' seconds of slack 
        #    (time spent starting up without doing work).
        # Therefore, we need slack >= gap + overhead at all times.
        # We apply safety multipliers to handle potential floating point inaccuracies or slight delays.
        safety_threshold = (1.1 * overhead) + (1.5 * gap)

        # Critical Path: If slack is running out, force On-Demand to guarantee deadline.
        if slack < safety_threshold:
            return ClusterType.ON_DEMAND

        # Cost Optimization: Use Spot if available.
        if has_spot:
            return ClusterType.SPOT

        # If Spot is unavailable but we have plenty of slack, wait (NONE) to save money.
        # This consumes slack but avoids the high cost of On-Demand.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)