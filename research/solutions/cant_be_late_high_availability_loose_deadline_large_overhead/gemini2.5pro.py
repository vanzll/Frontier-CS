import math
from argparse import ArgumentParser

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.

        This strategy implements a "just-in-time" approach for using On-Demand instances.
        It prioritizes cost-effective Spot instances and waiting, only switching to
        expensive On-Demand instances when absolutely necessary to meet the deadline.

        The core logic is as follows:
        1. Calculate the time required to finish all remaining work using a guaranteed
           On-Demand instance. This calculation includes a potential restart overhead if
           the system is currently idle.
        2. Compare this `time_needed` with the `time_available` until the deadline.
        3. If `time_available` is less than or equal to `time_needed`, we are in the
           critical path and must use On-Demand to guarantee completion.
        4. If there is slack (`time_available` > `time_needed`), we can afford to be
           cost-conscious:
           - Use Spot if available, as it's the cheapest way to make progress.
           - If Spot is unavailable, wait (NONE) to avoid On-Demand costs, effectively
             using our time slack to wait for cheaper resources.
        """
        # Calculate the total work remaining.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is complete, do nothing to save cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate the worst-case time required to finish from this point forward
        # using only On-Demand instances.
        time_needed_on_demand = work_remaining

        # If we are not currently running a cluster, starting one will incur a
        # restart overhead. This must be factored into our time calculation.
        if self.env.cluster_type == ClusterType.NONE:
            time_needed_on_demand += self.restart_overhead

        # Calculate the total time remaining until the hard deadline.
        time_available = self.deadline - self.env.elapsed_seconds

        # This is the critical decision point. If the time we have left is not
        # sufficient to complete the remaining work on a guaranteed instance,
        # we have no choice but to use On-Demand immediately.
        if time_available <= time_needed_on_demand:
            return ClusterType.ON_DEMAND

        # If we are not in the critical phase, we have a time buffer.
        # We should use the most cost-effective option.
        if has_spot:
            # Spot is available and cheaper, so we use it.
            return ClusterType.SPOT
        else:
            # Spot is not available, but since we have slack time, we can afford
            # to wait for it to become available again. This avoids paying for
            # expensive On-Demand instances before it's necessary.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: ArgumentParser):
        """
        Required method for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)