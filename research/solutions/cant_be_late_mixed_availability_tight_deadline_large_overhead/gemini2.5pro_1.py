from argparse import ArgumentParser
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "SlackAwareStrategy"  # REQUIRED: unique identifier

    def __init__(self, args):
        super().__init__(args)
        # This factor determines the safety buffer. A value of 1.1 means we
        # switch to a safe strategy (On-Demand) when our available slack time
        # drops below 110% of the time cost of a single preemption. This
        # provides a small margin for error.
        self.safety_buffer_factor = 1.1

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Must return self.
        """
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
        # Calculate current progress and time remaining.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is already done, stop incurring costs.
        if work_remaining <= 0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds

        # The core of the strategy is the concept of "slack".
        # Slack is the amount of time we can afford to be idle or spend on
        # overheads and still finish by the deadline if we work continuously
        # on a reliable instance from this point forward.
        current_slack = time_left - work_remaining

        # The safety buffer is the minimum slack we're comfortable having.
        # It's based on the restart_overhead, which is the primary source of
        # unexpected time loss when using Spot instances.
        safety_buffer = self.restart_overhead * self.safety_buffer_factor

        # --- Decision Logic ---

        # If our current slack is less than the safety buffer, we are in a
        # critical state. We cannot risk a preemption, as it might cause us
        # to miss the deadline. Therefore, we must use a reliable On-Demand
        # instance to guarantee progress.
        if current_slack < safety_buffer:
            return ClusterType.ON_DEMAND

        # Otherwise, we have enough slack to be in "economy mode". We can
        # prioritize saving costs.
        else:
            # If a cheap Spot instance is available, we use it. We have enough
            # slack to absorb the time penalty if we get preempted.
            if has_spot:
                return ClusterType.SPOT
            # If Spot is not available, we can afford to wait. We use our
            # slack time to wait for Spot to become available again, incurring
            # zero cost in the meantime.
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: ArgumentParser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)