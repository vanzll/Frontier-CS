import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        # --- Tunable Parameters ---

        # Safety buffer for panic mode, as a factor of restart_overhead.
        # A factor of 1.0 means we enter panic mode if a single preemption
        # would leave us with less than one restart_overhead worth of slack.
        self.panic_buffer_factor = 1.0

        # Slack buffer threshold in hours. When spot is unavailable, we use
        # On-Demand if current slack is below this threshold; otherwise, we wait.
        self.slack_buffer_hours = 6.0

        # --- Internal State ---
        self._panic_buffer_seconds = None
        self._slack_buffer_seconds = None

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
        if self._panic_buffer_seconds is None:
            self._panic_buffer_seconds = self.panic_buffer_factor * self.restart_overhead
            self._slack_buffer_seconds = self.slack_buffer_hours * 3600

        # 1. Calculate current state
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_time = self.env.elapsed_seconds
        time_until_deadline = self.deadline - elapsed_time

        if work_remaining >= time_until_deadline:
            return ClusterType.ON_DEMAND

        # 2. Panic Mode Check: Switch to On-Demand if a single preemption is too risky.
        time_needed_after_preemption = work_remaining + self.restart_overhead
        time_available_after_next_step = time_until_deadline - self.env.gap_seconds

        if time_needed_after_preemption + self._panic_buffer_seconds >= time_available_after_next_step:
            return ClusterType.ON_DEMAND

        # 3. Normal Operation: Prioritize Spot if available and safe.
        if has_spot:
            return ClusterType.SPOT

        # 4. No Spot: Decide between On-Demand and None based on slack.
        current_slack = time_until_deadline - work_remaining

        if current_slack < self._slack_buffer_seconds:
            # Slack is low; preserve it by using On-Demand.
            return ClusterType.ON_DEMAND
        else:
            # Slack is high; save cost by waiting for Spot.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """Instantiates the strategy from command-line arguments."""
        args, _ = parser.parse_known_args()
        return cls(args)