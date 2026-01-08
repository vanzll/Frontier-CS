from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "optimal_slack_hysteresis"

    def __init__(self, args):
        super().__init__(args)
        self.last_decision = ClusterType.NONE

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current progress and slack
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If work is done, stop incurring cost
        if work_remaining <= 0:
            return ClusterType.NONE

        time_remaining = self.deadline - elapsed
        slack = time_remaining - work_remaining

        # Constants
        gap = self.env.gap_seconds
        overhead = self.restart_overhead

        # Determine Safety Threshold
        # We need enough slack to cover the overhead of switching to/starting OD
        # plus a safety buffer for step granularity.
        # Base threshold = Overhead (to start OD) + Buffer (2 steps)
        base_threshold = overhead + (2 * gap)

        # Hysteresis Logic:
        # If we are currently committed to ON_DEMAND, we should only switch back to SPOT
        # if we have significantly more slack. Specifically, we need enough slack to:
        # 1. Pay the overhead of switching to SPOT.
        # 2. Still satisfy the base_threshold in the new state (so we don't immediately panic back).
        # Thus, we add 'overhead' to the threshold if we are currently on OD.
        if self.last_decision == ClusterType.ON_DEMAND:
            threshold = base_threshold + overhead
        else:
            threshold = base_threshold

        # Decision
        decision = ClusterType.NONE

        if slack < threshold:
            # Critical zone: Must ensure progress via On-Demand
            decision = ClusterType.ON_DEMAND
        elif has_spot:
            # Safe zone: Use Spot to save money
            decision = ClusterType.SPOT
        else:
            # Safe zone but Spot unavailable: Wait (Pause) to save money
            # We burn slack instead of burning cash on OD
            decision = ClusterType.NONE

        self.last_decision = decision
        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)