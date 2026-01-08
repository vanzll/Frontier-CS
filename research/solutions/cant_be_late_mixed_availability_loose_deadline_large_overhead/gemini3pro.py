from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CostAwareDeadlineStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialization method. Returns self as required.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides which cluster type to use at the current time step.
        Implements a strategy that prioritizes Spot instances for cost savings
        but switches to On-Demand when the deadline approaches the critical path time.
        """
        # Extract environment state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Calculate remaining work
        # task_done_time is a list of completed work segments
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If work is completed, do nothing (env should terminate, but for safety)
        if work_remaining <= 0:
            return ClusterType.NONE

        # Define safety buffer
        # We need a buffer to account for the discrete time steps (gap_seconds)
        # and ensure we don't overshoot the decision boundary.
        # Default to 300s if gap is unavailable, otherwise 2 steps.
        safe_gap = gap if gap is not None else 300.0
        buffer = 2.0 * safe_gap

        # Calculate time remaining before the hard deadline
        time_available = deadline - elapsed

        # Calculate the time required to finish the task if we commit to On-Demand NOW.
        # - If currently On-Demand: We can continue without paying overhead.
        # - If currently Spot or None: We must pay restart overhead to switch/start On-Demand.
        if last_cluster_type == ClusterType.ON_DEMAND:
            time_needed_on_demand = work_remaining
        else:
            time_needed_on_demand = work_remaining + overhead

        # CRITICAL ZONE CHECK:
        # If the available time is approaching the minimum time needed to finish using On-Demand,
        # we must force On-Demand usage to guarantee meeting the deadline.
        if time_available <= (time_needed_on_demand + buffer):
            return ClusterType.ON_DEMAND

        # SAFE ZONE STRATEGY:
        # If we have sufficient slack, we optimize for cost.
        # - Prefer Spot instances as they are significantly cheaper (~1/3 price).
        # - If Spot is unavailable, return NONE (wait) instead of using expensive On-Demand.
        #   Since we are not in the critical zone, we can afford to wait for Spot availability.
        if has_spot:
            return ClusterType.SPOT
        
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)