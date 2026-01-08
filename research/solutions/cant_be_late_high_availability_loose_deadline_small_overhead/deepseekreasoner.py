import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_threshold"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.spot_available_history = []
        self.conservative_threshold = 0.65
        self.aggressive_threshold = 0.35
        self.min_required_progress = 0.0
        self.safety_margin = 3600  # 1 hour in seconds
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.spot_available_history.append(has_spot)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)
        
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE
        
        spot_availability = sum(self.spot_available_history) / len(self.spot_available_history) if self.spot_available_history else 0.5
        
        # Calculate minimum time needed with on-demand
        min_time_needed = work_remaining
        if last_cluster_type == ClusterType.SPOT:
            min_time_needed += self.restart_overhead
        
        # Calculate aggressive and conservative progress thresholds
        time_ratio = elapsed / self.deadline if self.deadline > 0 else 0
        required_progress = time_ratio
        
        # Adjust thresholds based on spot availability
        adjusted_conservative = self.conservative_threshold + (0.5 - spot_availability) * 0.2
        adjusted_aggressive = self.aggressive_threshold + (0.5 - spot_availability) * 0.2
        
        # Calculate current progress
        current_progress = work_done / self.task_duration if self.task_duration > 0 else 0
        
        # Calculate safety buffer needed
        estimated_spot_efficiency = spot_availability * 0.8  # conservative estimate considering restarts
        time_needed_spot = work_remaining / estimated_spot_efficiency if estimated_spot_efficiency > 0 else float('inf')
        
        # Make decision
        if not has_spot:
            # No spot available
            if remaining_time <= min_time_needed + self.safety_margin:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
        else:
            # Spot is available
            if remaining_time <= time_needed_spot + self.safety_margin:
                # Running out of time, switch to on-demand
                return ClusterType.ON_DEMAND
            elif current_progress < required_progress - adjusted_conservative:
                # Behind schedule, use on-demand
                return ClusterType.ON_DEMAND
            elif current_progress < required_progress - adjusted_aggressive:
                # Slightly behind, use spot
                return ClusterType.SPOT
            else:
                # Ahead of schedule, use spot
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                else:
                    # Consider restart overhead when switching to spot
                    effective_remaining = remaining_time - self.restart_overhead
                    if effective_remaining > time_needed_spot:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)