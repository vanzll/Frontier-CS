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
        self.initial_slack = self.deadline - self.task_duration
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
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_time = self.env.elapsed_seconds
        time_to_deadline = self.deadline - elapsed_time

        slack = time_to_deadline - work_remaining

        if slack <= self.restart_overhead:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # If spot is not available, decide between ON_DEMAND and NONE.
        # We use a dynamic threshold that makes the strategy more conservative
        # as time progresses.

        time_progress = elapsed_time / self.deadline if self.deadline > 0 else 1.0

        max_wait_buffer = self.initial_slack - self.restart_overhead

        # The wait threshold `s_wait` increases from `restart_overhead` at t=0
        # to `initial_slack` at t=deadline.
        s_wait = self.restart_overhead + max_wait_buffer * time_progress

        if slack > s_wait:
            # If we have a large slack buffer, wait for spot to become available.
            return ClusterType.NONE
        else:
            # If slack is getting tight, use On-Demand to guarantee progress.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)