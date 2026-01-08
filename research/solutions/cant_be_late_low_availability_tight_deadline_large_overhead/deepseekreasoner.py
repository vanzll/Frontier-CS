from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import argparse
from typing import List

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.spot_available_history: List[bool] = []
        self.spot_unavailable_streak = 0
        self.consecutive_spot_usage = 0
        self.safety_margin_factor = 1.1
        
    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Track spot availability history
        self.spot_available_history.append(has_spot)
        if len(self.spot_available_history) > 10:
            self.spot_available_history.pop(0)
        
        # Track spot usage streak
        if last_cluster_type == ClusterType.SPOT:
            self.consecutive_spot_usage += 1
        else:
            self.consecutive_spot_usage = 0
        
        # Track spot unavailability streak
        if has_spot:
            self.spot_unavailable_streak = 0
        else:
            self.spot_unavailable_streak += 1
        
        # Calculate remaining work and time
        completed_work = sum(self.task_done_time)
        remaining_work = self.task_duration - completed_work
        time_elapsed = self.env.elapsed_seconds
        time_left = self.deadline - time_elapsed
        
        # If work is done, stop
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Emergency mode: if we're running out of time, use on-demand
        # Calculate safe completion time considering restart overhead
        estimated_completion_time = remaining_work
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            estimated_completion_time += self.restart_overhead
        
        # Conservative safety margin considering spot availability history
        avg_spot_available = sum(self.spot_available_history) / len(self.spot_available_history) if self.spot_available_history else 0
        safety_adjustment = 1.0 + (1.0 - avg_spot_available) * 0.5  # More conservative if spot is often unavailable
        
        if time_left < estimated_completion_time * safety_adjustment * self.safety_margin_factor:
            return ClusterType.ON_DEMAND
        
        # Calculate the minimum time we need with on-demand to meet deadline
        min_ondemand_time_needed = max(0, remaining_work - (time_left - self.restart_overhead))
        
        # If we've used spot for too long continuously, switch to on-demand briefly
        # to reduce risk of restart overhead at critical moment
        if self.consecutive_spot_usage > 5 and has_spot:
            # Check if we can afford a brief on-demand period
            if time_left > remaining_work * 1.2:
                return ClusterType.ON_DEMAND
        
        # If spot is available and we're not in emergency mode, use it
        if has_spot:
            # But be more conservative if spot has been recently unreliable
            if self.spot_unavailable_streak > 0 and avg_spot_available < 0.3:
                # Recent spot outage, use on-demand for a while
                if time_left > remaining_work * 1.3:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT
        
        # Spot is not available
        # Calculate how long we can wait for spot to come back
        max_wait_time = time_left - (remaining_work + self.restart_overhead)
        
        # If we can't wait long, use on-demand
        if max_wait_time < self.restart_overhead * 2:
            return ClusterType.ON_DEMAND
        
        # If we've been waiting too long for spot, use on-demand
        if self.spot_unavailable_streak > 3:
            return ClusterType.ON_DEMAND
        
        # Otherwise wait for spot to become available
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)