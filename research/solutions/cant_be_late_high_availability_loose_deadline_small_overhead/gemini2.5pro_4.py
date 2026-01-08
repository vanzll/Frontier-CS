import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, _args: argparse.Namespace = None):
        super().__init__()
        # A safety multiplier for the critical slack threshold. A value > 1.0
        # creates a buffer to avoid edge cases.
        self.critical_slack_multiplier = 1.1

        # A factor determining the wait threshold. We will wait for spot
        # instances only if our current slack is greater than this fraction
        # of the total initial slack.
        self.wait_slack_factor = 0.5

        # Internal state to hold calculated thresholds.
        # Initialized to a sentinel value (-1) before calculation.
        self.wait_slack_threshold = -1
        self.critical_slack_threshold = -1

    def solve(self, spec_path: str) -> "Solution":
        """
        Called once before evaluation.
        Pre-calculates static thresholds based on the problem specification.
        """
        # The total slack time available at the start of the task.
        initial_slack = self.deadline - self.task_duration

        # Threshold for deciding whether to wait for spot instances.
        self.wait_slack_threshold = initial_slack * self.wait_slack_factor
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step to decide which cluster type to use.
        The core logic is based on managing "slack" time.
        """
        # --- 1. State Calculation ---
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is completed, we no longer need compute resources.
        if work_remaining <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # Slack is the key metric: the amount of time we can afford to lose
        # before the deadline is at risk.
        current_slack = time_to_deadline - work_remaining

        # --- 2. One-time Initialization ---
        # The critical slack threshold depends on gap_seconds, which is only
        # available after the environment is fully initialized at the first step.
        if self.critical_slack_threshold == -1:
            # The time penalty for a single spot preemption.
            preemption_penalty = self.restart_overhead + self.env.gap_seconds
            self.critical_slack_threshold = preemption_penalty * self.critical_slack_multiplier

        # --- 3. Decision Logic ---
        
        # A) PANIC MODE: If slack is critically low.
        # If our current slack is less than the time it would take to recover
        # from a single spot preemption, we must use on-demand to guarantee
        # progress and ensure we meet the deadline.
        if current_slack <= self.critical_slack_threshold:
            return ClusterType.ON_DEMAND

        # B) NORMAL MODE: We have a safe amount of slack.
        
        # If spot instances are available, they are the most cost-effective
        # choice, and we have determined it's safe to use them.
        if has_spot:
            return ClusterType.SPOT
        
        # If spot is not available, we choose between waiting (NONE) or making
        # progress with on-demand.
        else:
            # If we have a large amount of slack, we can afford to wait for
            # spot instances to become available again, saving costs.
            if current_slack > self.wait_slack_threshold:
                return ClusterType.NONE
            # If our slack is in a medium range, it's more valuable to preserve
            # it by making progress with on-demand, rather than spending it
            # by waiting.
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)