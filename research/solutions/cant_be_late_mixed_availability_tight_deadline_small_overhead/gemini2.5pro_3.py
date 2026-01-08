import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize strategy parameters. Called once before evaluation.
        """
        # Multiplier for restart_overhead to determine the critical slack threshold.
        # If slack is below this, we cannot risk a preemption.
        self.CRITICAL_SLACK_MULTIPLIER = 1.2

        # Ratio of the initial slack to use for the opportunistic threshold.
        # This controls how aggressively we wait for spot instances.
        self.OPPORTUNISTIC_SLACK_RATIO = 0.7

        # The initial slack is the total buffer time we have at the start.
        self.initial_slack = self.deadline - self.task_duration
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decision-making logic at each time step.
        This strategy is based on managing "slack", the amount of extra time
        available before the deadline. The decision of which cluster to use
        depends on how much slack we have.
        """
        # Calculate current progress and remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is completed, do nothing to save costs.
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate time remaining until the deadline
        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # If we are already past the deadline but not finished, we must
        # use the fastest guaranteed option to minimize penalty.
        if time_to_deadline <= 0:
            return ClusterType.ON_DEMAND

        # Slack is the time we have left minus the work we still need to do.
        # It's our buffer against delays and preemptions.
        slack = time_to_deadline - work_remaining

        # Define dynamic thresholds for our three-zone strategy.
        
        # 1. Critical Threshold (s_critical):
        # If slack falls below this, we are in danger of missing the deadline
        # if even one preemption occurs. It's safest to switch to on-demand.
        s_critical = self.restart_overhead * self.CRITICAL_SLACK_MULTIPLIER

        # 2. Opportunistic Threshold (s_opportunistic):
        # This threshold determines when we can afford to wait (NONE) for
        # a spot instance. It decays as the deadline approaches, making the
        # strategy more conservative over time.
        s_opportunistic = self.initial_slack * self.OPPORTUNISTIC_SLACK_RATIO * \
                          (time_to_deadline / self.deadline)
        
        # Ensure the opportunistic threshold is always at least as large as
        # the critical one to maintain three distinct operational zones.
        s_opportunistic = max(s_opportunistic, s_critical)

        # Apply the strategy based on which zone the current slack falls into.
        if slack <= s_critical:
            # CRITICAL ZONE: Slack is dangerously low. We must use the most
            # reliable option (On-Demand) to guarantee progress.
            return ClusterType.ON_DEMAND
        elif slack <= s_opportunistic:
            # CAUTIOUS ZONE: Slack is shrinking. We should make progress, but can
            # still prefer the cheaper Spot if it's available. We fall back to
            # On-Demand otherwise, as we can't afford to wait.
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:  # slack > s_opportunistic
            # OPPORTUNISTIC ZONE: We have a comfortable amount of slack.
            # We can prioritize cost savings by waiting (NONE) for a cheap
            # Spot instance to become available.
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)