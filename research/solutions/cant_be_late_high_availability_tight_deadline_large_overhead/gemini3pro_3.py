from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLate"

    def __init__(self, args):
        super().__init__()
        self.args = args

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current progress and remaining work
        # task_done_time is a list of durations of completed work
        progress = sum(self.task_done_time)
        work_rem = self.task_duration - progress
        
        # Calculate time remaining until deadline
        time_rem = self.deadline - self.env.elapsed_seconds
        
        # Calculate Slack: The amount of time we can afford to waste (idle or overhead)
        slack = time_rem - work_rem
        
        # Define Safety Buffer
        # We need to preserve enough slack to cover the restart overhead if we are forced 
        # to switch to On-Demand. We also add padding for discrete simulation steps.
        # buffer = overhead + (2 * step_size)
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        safe_buffer = overhead + (2.0 * gap)
        
        # Decision Logic
        
        # 1. Panic Mode: Slack is critically low.
        # We must use On-Demand to guarantee completion before the deadline.
        if slack < safe_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Spot Availability Logic
        if has_spot:
            # Hysteresis optimization:
            # If we are currently running On-Demand (perhaps due to a previous dip in slack),
            # switching back to Spot incurs a restart overhead.
            # We should only switch if we have enough slack to pay that cost and still stay above the safety buffer.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > safe_buffer + overhead:
                    return ClusterType.SPOT
                else:
                    # Stay on On-Demand to avoid thrashing or dipping into danger zone
                    return ClusterType.ON_DEMAND
            
            # If currently None or Spot, use Spot
            return ClusterType.SPOT
            
        # 3. Wait Mode: Spot is unavailable, but we have slack (slack >= safe_buffer).
        # We return NONE (Wait).
        # Rationale: Waiting costs slack but saves money. Using OD saves slack but costs money.
        # Since we have excess slack, we spend it to wait for cheap Spot instances.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)