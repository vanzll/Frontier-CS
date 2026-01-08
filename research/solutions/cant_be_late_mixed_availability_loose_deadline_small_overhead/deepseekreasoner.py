import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_threshold"

    def __init__(self, args):
        super().__init__(args)
        self.safety_margin = None
        self.spot_price = None
        self.ondemand_price = None
        
    def solve(self, spec_path: str) -> "Solution":
        # Read configuration
        # Default values if parsing fails
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        
        # Calculate safety margin based on price ratio
        # More aggressive spot usage when price difference is larger
        price_ratio = self.spot_price / self.ondemand_price
        self.safety_margin = max(0.1, min(0.5, 1.0 - price_ratio)) * self.deadline
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        
        # Calculate remaining work and time
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - current_time
        
        # If no work left, do nothing
        if work_remaining <= 0:
            return ClusterType.NONE
        
        # Critical condition: must use on-demand if we're running out of time
        # Account for restart overhead in worst case
        min_time_needed = work_remaining + self.restart_overhead
        if time_remaining <= min_time_needed * 1.2:  # 20% safety factor
            return ClusterType.ON_DEMAND
        
        # Calculate efficiency thresholds
        # Base threshold on how much time we can afford to lose
        time_slack = time_remaining - work_remaining
        urgency = work_remaining / max(1, time_remaining)
        
        # Adaptive threshold based on progress
        progress_ratio = work_done / self.task_duration
        
        # Use spot when available and we have enough slack
        if has_spot:
            # Be more aggressive with spot usage early on
            if progress_ratio < 0.8:  # First 80% of work
                # Use spot if we have sufficient slack
                required_slack = self.restart_overhead * (1 + urgency * 3)
                if time_slack > required_slack:
                    return ClusterType.SPOT
                # Otherwise use on-demand to catch up
                elif urgency > 0.7:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE
            else:  # Last 20% - be more conservative
                required_slack = self.restart_overhead * (1 + urgency * 5)
                if time_slack > required_slack:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        
        # No spot available
        if last_cluster_type == ClusterType.SPOT:
            # Just lost spot, wait briefly to see if it comes back
            # But not too long if we're behind schedule
            if urgency < 0.8 and np.random.random() < 0.3:
                return ClusterType.NONE
        
        # Use on-demand if we're behind schedule or no spot
        if urgency > 0.6:
            return ClusterType.ON_DEMAND
        
        # Otherwise wait for spot
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)