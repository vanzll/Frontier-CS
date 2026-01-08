import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """
    A strategy that balances cost and completion risk by maintaining a "safety buffer".
    The safety buffer is defined by the ratio of time remaining to work remaining.

    The core logic is as follows:
    1.  If Spot instances are available, always use them for cost savings.
    2.  If Spot is not available, the decision to use expensive On-Demand or wait (NONE)
        depends on the current safety buffer.
    3.  The safety buffer is quantified by the ratio `R = time_to_deadline / work_remaining`.
        This `R` can be interpreted as the "slowness factor" we can afford.
    4.  A threshold, `safety_factor`, is set (e.g., 1.20). If `R` drops below this
        threshold, it means our time slack is getting uncomfortably low (e.g., less than 20%).
    5.  When the buffer is low (`R < safety_factor`), we use On-Demand to guarantee progress
        and preserve our remaining slack.
    6.  When the buffer is sufficient (`R >= safety_factor`), we can afford to wait for Spot
        to become available again, so we choose NONE to minimize costs.
    7.  An absolute "panic mode" triggers if `time_to_deadline <= work_remaining`, forcing
        the use of On-Demand as it's the only hope to meet the deadline.
    """
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy with a safety factor.
        """
        # This factor determines the minimum time buffer we aim to maintain.
        # A value of 1.20 means we will use On-Demand to prevent our time buffer
        # from dropping below 20% of the remaining work time.
        self.safety_factor = 1.20
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Implements the decision-making logic for each time step.
        """
        work_remaining = self.task_duration - self.work_done

        # If the job is finished, do nothing.
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # --- Panic Mode ---
        # If the remaining time is less than or equal to the remaining work,
        # we must use On-Demand instances to have any chance of finishing.
        # A small buffer equal to one time step is added for robustness.
        if time_to_deadline <= work_remaining + self.env.gap_seconds:
            return ClusterType.ON_DEMAND

        # --- Standard Operation ---
        if has_spot:
            # Always prefer cheap Spot instances when they are available.
            return ClusterType.SPOT
        else:
            # Spot is unavailable. Decide between On-Demand (costly) and NONE (risky).
            # The choice is based on our "slowness budget".
            slowness_budget = time_to_deadline / work_remaining

            if slowness_budget < self.safety_factor:
                # The time buffer is too low. We cannot afford to wait.
                # Use On-Demand to make progress and protect the remaining buffer.
                return ClusterType.ON_DEMAND
            else:
                # The time buffer is sufficient. We can afford to wait for Spot
                # to become available, thus saving costs.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> "Solution":
        """
        Required classmethod to instantiate the strategy for the evaluator.
        """
        args, _ = parser.parse_known_args()
        return cls(args)