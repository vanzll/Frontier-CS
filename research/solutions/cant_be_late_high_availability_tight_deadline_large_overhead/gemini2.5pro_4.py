import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    A strategy that balances cost-saving with a strong guarantee of meeting the deadline.

    The core idea is two-fold:
    1. A hard safety net based on slack time: It calculates if there's enough time
       left to finish on a reliable on-demand instance, even after accounting for one
       potential restart overhead. If the time is too tight (slack <= restart_overhead),
       it forces the use of On-Demand to prevent failure.
    2. A rate-based heuristic for economic decisions: When not in the critical safety
       zone, it compares the current work progress against a target linear progress
       rate. If ahead of schedule, it's conservative with spending, waiting for cheap
       Spot instances. If behind, it uses On-Demand to catch up.
    """
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy by calculating the target progress rate.
        This rate defines a linear schedule to complete the task exactly at the deadline.
        """
        if self.deadline > 0:
            self.target_rate = self.task_duration / self.deadline
        else:
            # Fallback for the edge case of a zero deadline.
            self.target_rate = 1.0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Determines the cluster type to use for the next time step.
        """
        # Calculate current progress by summing up completed work segments.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_left = self.task_duration - work_done

        # If the task is finished, do nothing to save costs.
        if work_left <= 0:
            return ClusterType.NONE

        # Calculate time remaining and the current slack.
        time_left = self.deadline - self.env.elapsed_seconds
        slack = time_left - work_left

        # --- Safety Net: Enforce Deadline ---
        # If the available slack is less than or equal to the time lost in one
        # restart, we can't risk another preemption. We must switch to the
        # reliable On-Demand instance to guarantee timely completion.
        if slack <= self.restart_overhead:
            return ClusterType.ON_DEMAND

        # --- Cost Optimization: Main Logic ---
        # If we have a comfortable safety margin, we can focus on minimizing cost.

        # Always use Spot when available, as it's the cheapest option.
        if has_spot:
            return ClusterType.SPOT

        # Spot is not available. We must choose between spending on On-Demand or
        # waiting and hoping Spot returns.
        
        # We determine if we are "ahead of schedule" or "behind schedule".
        # The schedule is a linear line from (0, 0) to (deadline, task_duration).
        target_work_done = self.env.elapsed_seconds * self.target_rate

        if work_done < target_work_done:
            # We are behind schedule. Use On-Demand to catch up and reduce risk.
            return ClusterType.ON_DEMAND
        else:
            # We are on or ahead of schedule. We can afford to wait for Spot to
            # become available again, incurring no cost in this step.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for the evaluator to instantiate the solution.
        """
        args, _ = parser.parse_known_args()
        return cls(args)