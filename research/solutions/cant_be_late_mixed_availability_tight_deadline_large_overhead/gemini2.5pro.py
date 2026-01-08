import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """
    This strategy operates on a simple but robust principle:
    1.  It calculates the total available "slack" time it has. Slack is defined
        as the amount of time that can be wasted (e.g., by waiting for a spot
        instance) before the job is at risk of missing the deadline, even if it
        switches to a reliable on-demand instance.
    2.  It defines a "safety buffer" based on the restart overhead. This buffer
        is the minimum amount of slack we want to preserve to handle a worst-case
        event (a preemption) near the deadline.
    3.  If the current slack is greater than the safety buffer, the strategy
        acts greedily to minimize cost: it uses Spot if available, otherwise it
        waits (NONE) since it can afford the delay.
    4.  If the slack drops below the safety buffer, the strategy switches to a
        "safe" mode, using On-Demand to guarantee progress and ensure it
        finishes before the deadline.

    This approach avoids complex predictions about spot availability, focusing
    instead on a reactive, risk-managed process. The key tunable parameter is
    the `safety_buffer_multiplier`, which adjusts how conservative the strategy is.
    """
    NAME = "my_solution"

    # A tunable hyperparameter for our safety margin. It's a multiplier for the
    # restart_overhead. A value of 1.0 means we switch to On-Demand when our
    # slack time is <= the time to recover from one preemption. A value
    # slightly > 1.0 provides an additional margin of safety.
    DEFAULT_SAFETY_MULTIPLIER = 1.1

    def __init__(self, args):
        super().__init__(args)
        # Set the safety multiplier from parsed arguments or use the default.
        self.safety_multiplier = getattr(args, 'safety_buffer_multiplier',
                                         self.DEFAULT_SAFETY_MULTIPLIER)

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        No complex initialization is needed for this reactive strategy.
        """
        return self

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Calculate the total work completed and remaining.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is finished, do nothing to save cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate the latest possible time we can start running on On-Demand
        # continuously and still finish exactly at the deadline.
        must_start_od_time = self.deadline - work_remaining

        # Calculate the available time slack. This is how much time we can
        # afford to "waste" before we are forced to use On-Demand.
        slack = must_start_od_time - self.env.elapsed_seconds

        # Define a safety buffer. We switch to a safe strategy (On-Demand)
        # before our slack runs out completely. The time lost to a single
        # preemption is a sensible basis for this buffer.
        safety_buffer = self.safety_multiplier * self.restart_overhead

        # --- Decision Logic ---

        # 1. CAUTION / PANIC MODE
        # If our time slack is less than or equal to the safety buffer, we can't
        # risk any more delays from waiting or potential preemptions. We must
        # switch to On-Demand for guaranteed progress.
        if slack <= safety_buffer:
            return ClusterType.ON_DEMAND

        # 2. NORMAL (GREEDY) MODE
        # If we have enough slack, we prioritize cost savings.
        else:
            if has_spot:
                # Spot is available and is the cheapest option for making progress.
                return ClusterType.SPOT
            else:
                # Spot is not available. Since we have sufficient slack, it's
                # better to wait (cost-free) for it to potentially become
                # available, rather than paying for expensive On-Demand.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Adds strategy-specific arguments to the parser and instantiates the class.
        This allows for tuning the strategy's parameters from the command line.
        """
        parser.add_argument(
            '--safety-buffer-multiplier',
            type=float,
            default=cls.DEFAULT_SAFETY_MULTIPLIER,
            help=
            'Multiplier for the restart overhead to determine the safety buffer.'
        )
        args, _ = parser.parse_known_args()
        return cls(args)