from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        """
        No initialization needed for this strategy.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the next cluster type to use based on availability, slack, and cost.
        """
        # Calculate current state
        done_work = sum(self.task_done_time)
        remaining_work = self.task_duration - done_work
        
        # If work is effectively done, stop
        if remaining_work <= 1e-6:
            return ClusterType.NONE
            
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        # Calculate slack: the amount of time we can afford to waste (overhead or pause)
        # Total Slack = 4 hours (14400s)
        slack = remaining_time - remaining_work
        
        # Threshold for switching from On-Demand (safe) to Spot (cheap)
        # Switching incurs a restart overhead (0.2h = 720s).
        # We should only switch if we have a healthy slack buffer to absorb the overhead
        # and potential future interruptions.
        # A 1-hour buffer (3600s) allows for approx 5 restart overheads.
        switch_to_spot_threshold = 3600.0
        
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                # We are currently running On-Demand.
                # Only switch to Spot if we have enough slack to justify the overhead risk.
                if slack > switch_to_spot_threshold:
                    return ClusterType.SPOT
                else:
                    # Slack is tight; stay on safe On-Demand instances.
                    return ClusterType.ON_DEMAND
            else:
                # Currently on Spot or None. Spot is available and cheaper.
                return ClusterType.SPOT
        else:
            # Spot is unavailable.
            # With low availability and tight deadline, we cannot afford to wait (NONE).
            # We must use On-Demand to ensure progress.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)