from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateOptimized"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Configuration Thresholds (in seconds)
        # CRITICAL_BUFFER: If slack drops below this, force ON_DEMAND to avoid preemption risk near deadline.
        # 20 minutes = 1200 seconds.
        CRITICAL_BUFFER = 20 * 60
        
        # WAIT_BUFFER: If Spot is unavailable, we wait (return NONE) to save money, 
        # provided our slack is above this threshold.
        # If slack drops below this, we stop waiting and use ON_DEMAND.
        # 1 hour = 3600 seconds.
        WAIT_BUFFER = 60 * 60

        # Calculate Work State
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # Sanity check if job is finished
        if work_remaining <= 0:
            return ClusterType.NONE

        # Calculate Time State
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Calculate Slack
        # Slack = Time_Available - Time_Required
        # Time_Required includes work_remaining + restart_overhead.
        # We include overhead conservatively to account for the cost of starting/switching instances.
        slack = time_remaining - (work_remaining + self.restart_overhead)

        # 1. Critical Safety Check
        # If we are dangerously close to the deadline (low slack), we cannot tolerate 
        # the unreliability/preemption risk of Spot. Force On-Demand.
        if slack < CRITICAL_BUFFER:
            return ClusterType.ON_DEMAND

        # 2. Prefer Spot Instances
        # If we have enough buffer, always prefer Spot to minimize cost.
        if has_spot:
            return ClusterType.SPOT
        
        # 3. Handle Spot Unavailability
        else:
            # Spot is not available. We must decide whether to wait or pay for On-Demand.
            
            # If we have a healthy amount of slack, we return NONE (wait).
            # This consumes slack but costs $0, preserving the chance to use Spot later.
            if slack > WAIT_BUFFER:
                return ClusterType.NONE
            else:
                # If slack has decayed below our wait threshold, we must ensure progress.
                # Switch to On-Demand.
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)