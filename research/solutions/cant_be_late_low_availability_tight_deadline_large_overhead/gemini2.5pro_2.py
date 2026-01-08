import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "SafeSlackStrategy"

    def __init__(self, args):
        super().__init__(args)
        # This multiplier determines how conservative the strategy is.
        # A higher value means it will switch to On-Demand earlier,
        # increasing the safety margin at the cost of potential savings.
        # A value of 2.0 means we want to keep a slack buffer of at least
        # two full restart overheads.
        self.safety_buffer_multiplier = 2.0

    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Core decision logic for choosing the cluster type at each step.
        The strategy is based on maintaining a "safety slack".
        """
        # 1. Calculate progress and remaining work.
        total_work_done = sum(self.task_done_time)
        work_left = self.task_duration - total_work_done

        # 2. If the job is complete, do nothing to minimize cost.
        if work_left <= 0:
            return ClusterType.NONE

        # 3. Calculate time remaining and the critical path.
        time_left_until_deadline = self.deadline - self.env.elapsed_seconds

        # If remaining work is more than time left, we cannot finish.
        # Use On-Demand to make maximum possible progress.
        if work_left > time_left_until_deadline:
            return ClusterType.ON_DEMAND

        # 4. Calculate the current "safety slack".
        # This is the buffer we have if we were to run On-Demand from now on.
        current_slack = time_left_until_deadline - work_left

        # 5. Define the safety buffer threshold.
        # This is the minimum slack we are willing to risk, proportional to the
        # time penalty of a spot preemption (restart_overhead).
        safety_buffer = self.safety_buffer_multiplier * self.restart_overhead

        # 6. Decision-making based on the safety slack.
        if current_slack <= safety_buffer:
            # If slack is low, we are in a "danger zone".
            # We must use the reliable On-Demand instance to guarantee progress.
            return ClusterType.ON_DEMAND
        
        # If slack is comfortable, we can be cost-effective.
        if has_spot:
            # If the cheap spot instance is available, use it.
            return ClusterType.SPOT
        else:
            # If spot is unavailable but we have plenty of slack, we can afford
            # to wait. Pausing (NONE) costs nothing and "spends" our time slack
            # to save money, waiting for spot to become available again.
            return ClusterType.NONE