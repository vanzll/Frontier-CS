import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """
    This strategy uses a dynamic slack-based threshold to decide when to switch
    from cheap Spot instances to reliable On-Demand instances.

    The core idea is to maintain a "safety slack" of time. This slack is the
    difference between the time available until the deadline and the time strictly
    required to finish the remaining work on an On-Demand instance.

    - If the slack is high, it indicates we can afford the risk of preemption,
      so we aggressively use Spot instances to save cost.
    - If the slack drops below a certain threshold, we switch to On-Demand to
      guarantee completion before the deadline.

    The key feature is that this threshold is dynamic:
    - It is high at the beginning of the task, making the strategy more
      conservative when there is a lot of work left and the impact of falling
      behind is significant.
    - It decreases as the task progresses, allowing the strategy to be more
      aggressive with cost-saving Spot instances towards the end.

    This approach balances cost-saving with the hard requirement of finishing
    before the deadline.
    """
    NAME = "DynamicSlackStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize strategy-specific parameters. This method is called once
        before the simulation starts.
        """
        # Multiplier for the slack threshold at the start of the task (progress=0).
        # A higher value makes the strategy more conservative initially, reserving
        # a larger time buffer for potential preemptions.
        self.C_initial = 10.0

        # Multiplier for the slack threshold near the end of the task (progress=1).
        # This value should be at least 1.0 to ensure there is enough time to
        # recover from at least one last-minute preemption.
        self.C_final = 1.5

        # A small epsilon for floating-point comparisons to check if work is done.
        self.WORK_DONE_EPSILON = 1e-6
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        The main decision-making function, called at each time step of the
        simulation.
        """
        # 1. Calculate the current state of the job.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is effectively complete, we can idle to save costs.
        if work_remaining <= self.WORK_DONE_EPSILON:
            return ClusterType.NONE

        # Calculate the time required to finish all remaining work using a
        # guaranteed On-Demand instance from this point forward. This includes
        # any pending restart overhead.
        time_needed_on_demand = work_remaining + self.env.remaining_restart_overhead

        # Calculate the total time available until the hard deadline.
        time_available = self.deadline - self.env.elapsed_seconds

        # The "slack" is our safety buffer: the amount of time we have beyond
        # what is strictly necessary to finish with On-Demand.
        slack = time_available - time_needed_on_demand

        # 2. Determine the dynamic safety threshold for the current step.
        progress = work_done / self.task_duration if self.task_duration > 0 else 1.0
        # Clamp progress to the [0, 1] range.
        progress = max(0.0, min(1.0, progress))

        # The threshold's multiplier is linearly interpolated from C_initial to
        # C_final based on job progress.
        c_multiplier = self.C_initial * (1 - progress) + self.C_final * progress
        slack_threshold = c_multiplier * self.restart_overhead

        # 3. Make the scheduling decision.
        # If our safety slack is below the dynamically calculated threshold,
        # it's too risky to use Spot. We must switch to On-Demand.
        if slack <= slack_threshold:
            return ClusterType.ON_DEMAND
        else:
            # If we have a comfortable safety margin, we prefer the cheaper Spot
            # instance whenever it's available.
            if has_spot:
                return ClusterType.SPOT
            else:
                # If Spot is unavailable, we use On-Demand to make progress
                # rather than waiting and losing valuable time (and slack).
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required classmethod for the evaluator to instantiate the class.
        """
        args, _ = parser.parse_known_args()
        return cls(args)