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
        initial_slack = self.deadline - self.task_duration
        
        # The critical buffer is a safety margin. If the remaining slack time
        # drops below this value, the strategy will switch to using only
        # on-demand instances to guarantee completion before the deadline.
        # It's calculated based on a fraction of the initial slack plus
        # a buffer for several potential restart overheads. This balances
        # cost-saving with the high penalty for missing the deadline.
        self.critical_buffer = 0.10 * initial_slack + 5 * self.restart_overhead
        
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
        work_done = sum(self.task_done_time)

        if work_done >= self.task_duration:
            return ClusterType.NONE

        work_remaining = self.task_duration - work_done
        time_remaining_until_deadline = self.deadline - self.env.elapsed_seconds

        # Failsafe: if it's impossible to finish, stop incurring costs.
        if time_remaining_until_deadline < work_remaining:
            return ClusterType.NONE

        # Slack is the available time beyond what's minimally required.
        slack = time_remaining_until_deadline - work_remaining

        # "Panic mode": If slack is below the critical buffer, use on-demand
        # to guarantee finishing.
        if slack <= self.critical_buffer:
            return ClusterType.ON_DEMAND
        
        # "Standard mode": Prioritize cost savings while making progress.
        # Use cheap Spot instances whenever they are available.
        if has_spot:
            return ClusterType.SPOT
        
        # If Spot is not available, use On-Demand. Waiting (NONE) is too risky
        # with low spot availability as it consumes slack without making progress.
        # Using On-Demand preserves slack for future Spot opportunities.
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)