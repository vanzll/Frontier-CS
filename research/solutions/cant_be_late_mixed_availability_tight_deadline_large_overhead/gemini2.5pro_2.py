import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "slack_based_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        Args:
            last_cluster_type: The cluster type used in the previous step
            has_spot: Whether spot instances are available this step

        Returns:
            ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # 1. Calculate progress
        work_done = 0
        if self.task_done_time:
            work_done = sum(end - start for start, end in self.task_done_time)
        
        remaining_work = self.task_duration - work_done

        # If the job is done, do nothing to save cost.
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        # 2. Calculate current state
        remaining_time_to_deadline = self.deadline - self.env.elapsed_seconds
        current_slack = remaining_time_to_deadline - remaining_work

        # 3. Define slack-based thresholds
        # Point-of-No-Return (PONR) threshold: If slack is less than one
        # restart_overhead, we must use On-Demand to guarantee completion.
        ponr_threshold = self.restart_overhead

        # Proactive On-Demand threshold: If slack is below this and spot is
        # unavailable, we use On-Demand instead of waiting to preserve slack.
        # 5 * restart_overhead corresponds to a 1-hour buffer (5 * 720s).
        proactive_od_threshold = 5.0 * self.restart_overhead

        # 4. Decision Logic
        # Rule 1 (Criticality): If slack is below the PONR threshold, we must
        # use On-Demand, regardless of Spot availability.
        if current_slack <= ponr_threshold:
            return ClusterType.ON_DEMAND

        # Rule 2 (Cost-saving): If not critical and Spot is available, use it.
        if has_spot:
            return ClusterType.SPOT

        # Rule 3 (Risk Management): Spot is unavailable. Decide whether to
        # wait (NONE) or use On-Demand proactively based on remaining slack.
        if current_slack <= proactive_od_threshold:
            return ClusterType.ON_DEMAND
        else:
            # We have enough slack to wait for Spot to become available again.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)