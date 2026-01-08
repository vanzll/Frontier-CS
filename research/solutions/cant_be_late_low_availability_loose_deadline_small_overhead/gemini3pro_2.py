from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLate_Robust_Slack"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialization logic. Returns self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the cluster type for the next time step.
        Implements a Least Slack Time (LST) strategy with a safety buffer.
        Prioritizes meeting the deadline over cost when slack is low.
        """
        # Retrieve current environment state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        # Calculate remaining work
        # task_done_time is a list of completed work segment durations
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done
        
        # If task is effectively complete, stop
        if work_rem <= 1e-6:
            return ClusterType.NONE

        time_rem = self.deadline - elapsed

        # Calculate Safety Threshold
        # We must ensure that at any point, we have enough time to:
        # 1. Restart on On-Demand (incurring restart_overhead)
        # 2. Complete the remaining work
        # 3. Absorb the time loss of the current step (gap) if we choose not to run OD or if Spot fails
        #
        # We also add a fixed padding (15 minutes) to account for simulation noise,
        # discretization errors, or slight variations in overhead handling.
        #
        # Note: We include restart_overhead in the threshold calculation even if we are currently
        # on OD. This prevents "thrashing" (switching OD->NONE->OD) by creating a hysteresis 
        # effect: once we are forced to OD due to low slack, we stay there unless we gain 
        # significant slack (e.g. if the task duration was reduced or deadline extended, 
        # which doesn't happen here, effectively locking us in OD for the final stretch).
        
        padding = 900.0  # 15 minutes safety buffer
        min_required_time = work_rem + self.restart_overhead + gap + padding
        
        # Panic Mode: If remaining time is critically low, force On-Demand usage.
        # This overrides spot availability to guarantee deadline completion.
        if time_rem <= min_required_time:
            return ClusterType.ON_DEMAND

        # Cost Optimization Mode:
        # If we have sufficient slack, we prioritize cost.
        if has_spot:
            # Spot is available and cheap -> Use it
            return ClusterType.SPOT
        else:
            # Spot is unavailable, but we have slack -> Wait (NONE)
            # This is cheaper than running On-Demand unnecessarily.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)