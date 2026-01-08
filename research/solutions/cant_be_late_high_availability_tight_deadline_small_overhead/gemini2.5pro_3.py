import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the tunable parameters for the strategy.
        """
        # Based on problem spec: 0.05 hours * 3600 s/hr = 180s
        restart_overhead_seconds = 180

        # Absolute slack safety buffer for panic mode. If slack drops below this,
        # we use ON_DEMAND unconditionally. A value of 30 minutes (10x restart
        # overhead) provides a robust buffer.
        self.absolute_slack_panic_seconds = 10 * restart_overhead_seconds

        # Threshold for the slack ratio (slack / work_remaining).
        # When spot is unavailable, if our current slack ratio is above this
        # threshold, we wait (NONE). If it's below, we use ON_DEMAND.
        # Initial ratio is (52-48)/48 = ~8.3%. A 6% threshold is a safe choice.
        self.wait_for_spot_slack_ratio_threshold = 0.06

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decision logic for each time step.
        """
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is finished, do nothing to save cost.
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # This is the total time we can be idle and still finish by running
        # exclusively on On-Demand instances from this point forward.
        current_absolute_slack = time_to_deadline - work_remaining

        # 1. Panic Mode: If absolute slack is critically low, use On-Demand.
        if current_absolute_slack <= self.absolute_slack_panic_seconds:
            return ClusterType.ON_DEMAND

        # 2. Opportunistic Spot: If Spot is available, always use it.
        if has_spot:
            return ClusterType.SPOT

        # 3. Wait vs. Work (when Spot is unavailable):
        # Decide based on the health of our schedule, measured by slack ratio.
        current_slack_ratio = current_absolute_slack / work_remaining

        if current_slack_ratio > self.wait_for_spot_slack_ratio_threshold:
            # Schedule is healthy; we can afford to wait for Spot.
            return ClusterType.NONE
        else:
            # Schedule health is deteriorating; use On-Demand to make progress.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """
        Instantiates the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)