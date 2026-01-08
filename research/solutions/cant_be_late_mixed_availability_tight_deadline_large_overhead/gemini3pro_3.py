from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_solution"

    def __init__(self, args):
        self.args = args

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Calculate total work done and remaining work
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If the task is effectively finished, stop using resources
        if work_remaining <= 0:
            return ClusterType.NONE
            
        # Calculate time remaining relative to the deadline
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Determine the safety threshold for forcing On-Demand.
        # We need enough time to potentially restart (pay overhead) and then complete the work.
        # We add a safety margin (2 steps) to account for discrete time stepping logic 
        # and ensure we switch before it's mathematically impossible.
        safety_margin = 2 * self.env.gap_seconds
        min_time_needed = work_remaining + self.restart_overhead + safety_margin
        
        # Panic Logic:
        # If the remaining time is dropping dangerously close to the minimum required time,
        # we must use On-Demand immediately to guarantee meeting the deadline.
        if time_remaining < min_time_needed:
            return ClusterType.ON_DEMAND
            
        # Cost Optimization Logic:
        # If we have sufficient slack (time buffer), we try to minimize cost.
        if has_spot:
            # Spot is available and we have time, use it (cheapest option).
            return ClusterType.SPOT
        else:
            # Spot is unavailable, but we still have slack.
            # We choose to WAIT (NONE) rather than run On-Demand.
            # Rationale: Running On-Demand now costs the same as running it later,
            # but waiting gives us a chance for Spot to become available again,
            # potentially saving significant cost. The Panic Logic ensures we 
            # don't wait too long.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)