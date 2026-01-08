import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes state variables. The main initialization of strategy
        parameters is deferred to the first call of _step, as task-specific
        attributes like `self.deadline` are not guaranteed to be available here.
        """
        self._initialized = False
        self._total_work_done = 0.0
        self._last_done_time_len = 0
        return self

    def _initialize(self):
        """
        Lazy initialization of strategy parameters. This is called on the first
        _step, ensuring all environment attributes are populated.
        """
        if self.task_duration > 0:
            initial_slack = self.deadline - self.task_duration
            # Ensure initial_slack is not negative for impossible tasks
            initial_slack = max(0, initial_slack)
            self.initial_slack_ratio = initial_slack / self.task_duration
        else:
            self.initial_slack_ratio = 0.0

        # These factors control the risk-aversion of the strategy. They are
        # multiplied by the initial slack-to-work ratio to determine dynamic
        # thresholds for our decision-making.

        # K_CUSHION_FACTOR: Determines the transition from the "Safe Zone"
        # (willing to wait for Spot) to the "Caution Zone" (use On-Demand if
        # Spot is unavailable). A higher value makes the strategy less patient.
        # Given the low-availability environment, being less patient is safer.
        k_cushion_factor = 0.8

        # K_SAFETY_FACTOR: Determines the transition to the "Danger Zone"
        # (use On-Demand exclusively). A higher value makes the strategy more
        # risk-averse.
        k_safety_factor = 0.2

        self.k_cushion = k_cushion_factor * self.initial_slack_ratio
        self.k_safety = k_safety_factor * self.initial_slack_ratio

        # A constant minimum buffer to account for at least one restart
        # overhead. This provides a baseline level of safety and prevents
        # issues when `work_remaining` is very small.
        self.min_safety_slack_s = self.restart_overhead * 1.5

        self._initialized = True

    def _update_progress(self):
        """
        Efficiently updates the total work done by only summing new segments
        from `self.task_done_time`, avoiding redundant calculations.
        """
        if len(self.task_done_time) > self._last_done_time_len:
            new_segments = self.task_done_time[self._last_done_time_len:]
            self._total_work_done += sum(end - start for start, end in new_segments)
            self._last_done_time_len = len(self.task_done_time)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Implements a two-threshold policy based on the current "slack".
        Slack = (Time to Deadline) - (Work Remaining on On-Demand)
        The policy decides whether to be in a "Safe", "Caution", or "Danger"
        zone, dictating whether to use SPOT, ON_DEMAND, or NONE.
        """
        if not self._initialized:
            self._initialize()

        self._update_progress()

        work_remaining = self.task_duration - self._total_work_done

        # If the task is finished, do nothing. Use a small epsilon for FP comparison.
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # If there's not enough time to finish even with uninterrupted On-Demand,
        # we must use On-Demand. This is the absolute last resort.
        if time_to_deadline < work_remaining:
            return ClusterType.ON_DEMAND

        current_slack = time_to_deadline - work_remaining

        # Calculate dynamic thresholds based on remaining work. These margins
        # represent the minimum slack we want to maintain for each zone.
        safety_margin = work_remaining * self.k_safety + self.min_safety_slack_s
        cushion_margin = work_remaining * self.k_cushion + self.min_safety_slack_s

        # Decision Tree:

        # 1. Danger Zone: Is slack below the safety margin?
        # If so, the risk of missing the deadline is too high. Use On-Demand
        # to guarantee progress, even if Spot is available.
        if current_slack <= safety_margin:
            return ClusterType.ON_DEMAND

        # 2. Not in Danger Zone: Is the cheap option available?
        # If Spot is available, use it. It's the most cost-effective way to
        # make progress, and we have enough slack to absorb a potential preemption.
        if has_spot:
            return ClusterType.SPOT

        # 3. No Spot: Caution Zone vs. Safe Zone
        # The decision is between waiting (NONE) and paying for progress (ON_DEMAND).
        # This depends on whether our slack is below the cushion margin.
        if current_slack <= cushion_margin:
            # Caution Zone: We can't afford to lose more slack by waiting.
            # Use On-Demand to preserve our slack buffer.
            return ClusterType.ON_DEMAND
        else:
            # Safe Zone: We have abundant slack. It's cost-effective to save
            # money and wait for Spot to become available again.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required class method for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)