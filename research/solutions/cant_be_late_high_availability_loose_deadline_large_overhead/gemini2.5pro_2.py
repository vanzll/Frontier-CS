import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # This strategy uses a "slack ratio" to make decisions.
        # Slack Ratio = (Time To Deadline - Work Remaining) / Work Remaining
        # It represents the buffer time relative to the work left.

        # Threshold to switch from cost-saving mode (SPOT/NONE) to progress-focused
        # mode (SPOT/ON_DEMAND). Initial slack ratio is (70-48)/48 ~= 0.46.
        self.high_slack_ratio = 0.35

        # Threshold to switch to safety mode (always ON_DEMAND).
        # A value of 0.25 provides a 25% time buffer to absorb potential
        # spot preemptions and unavailability, especially near the end of the job.
        self.low_slack_ratio = 0.25
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
        # 1. Calculate current state
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is finished, do nothing to save cost.
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # If the time remaining is less than the work remaining, we are in a critical
        # state. We must use On-Demand to finish as fast as possible. This is
        # equivalent to having zero or negative slack.
        if time_to_deadline <= work_remaining:
            return ClusterType.ON_DEMAND

        # 2. Calculate the core metric: the slack ratio
        slack = time_to_deadline - work_remaining
        slack_ratio = slack / work_remaining

        # 3. Apply three-zone decision logic based on the slack ratio

        # Zone 1: Urgent (low slack).
        # The risk of missing the deadline is high. Use On-Demand exclusively
        # to guarantee progress and avoid any further risk.
        if slack_ratio <= self.low_slack_ratio:
            return ClusterType.ON_DEMAND

        # Zone 2: Cautious (medium slack).
        # The time buffer is shrinking. Prioritize making progress.
        # Use Spot if available, but fall back to On-Demand instead of waiting.
        elif self.low_slack_ratio < slack_ratio <= self.high_slack_ratio:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # Zone 3: Safe (high slack).
        # We have plenty of time. Prioritize cost savings.
        # Use Spot if available, and wait (NONE) if it is not.
        else:  # slack_ratio > self.high_slack_ratio
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)