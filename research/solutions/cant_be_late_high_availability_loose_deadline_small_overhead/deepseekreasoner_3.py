import math
from typing import List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.work_segments = []
        self.spot_available_history = []
        self.current_segment_start = None
        self.current_segment_type = ClusterType.NONE
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Record availability history
        self.spot_available_history.append(has_spot)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)
        
        # Track work segments
        if last_cluster_type != self.current_segment_type:
            if self.current_segment_start is not None:
                self.work_segments.append((self.current_segment_start, self.env.elapsed_seconds, self.current_segment_type))
            self.current_segment_start = self.env.elapsed_seconds
            self.current_segment_type = last_cluster_type
        
        # Calculate remaining work and time
        work_done = sum(end - start for start, end, _ in self.work_segments)
        if last_cluster_type != ClusterType.NONE and self.current_segment_start is not None:
            work_done += self.env.elapsed_seconds - self.current_segment_start
        
        work_left = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If work is done, stop
        if work_left <= 0:
            return ClusterType.NONE
        
        # Calculate time needed for on-demand completion
        time_needed_ondemand = work_left
        
        # Calculate conservative time needed for spot
        # Assume worst-case availability pattern from history
        spot_availability = sum(self.spot_available_history) / max(len(self.spot_available_history), 1)
        if spot_availability < 0.5:
            spot_availability = 0.5  # Conservative estimate
        
        # Account for restart overheads - assume one overhead per segment
        segment_count = len([seg for seg in self.work_segments if seg[2] == ClusterType.SPOT])
        future_segments_est = max(1, math.ceil(work_left / (self.env.gap_seconds * 10)))
        restart_penalty = self.restart_overhead * future_segments_est
        
        time_needed_spot = work_left / spot_availability + restart_penalty
        
        # Emergency mode: if we're running out of time, go on-demand
        safety_margin = self.restart_overhead * 3
        if time_left < time_needed_ondemand + safety_margin:
            return ClusterType.ON_DEMAND
        
        # Calculate cost-benefit ratio
        spot_price = 0.97 / 3600  # $ per second
        ondemand_price = 3.06 / 3600  # $ per second
        
        spot_cost = work_left * spot_price / spot_availability
        ondemand_cost = work_left * ondemand_price
        
        # If spot is significantly cheaper and we have time, use it
        if has_spot and time_left > time_needed_spot + safety_margin:
            # Check if we should avoid frequent switching
            if last_cluster_type == ClusterType.SPOT and self._should_continue_spot():
                return ClusterType.SPOT
            elif last_cluster_type != ClusterType.SPOT:
                # Only switch to spot if we'll use it for a while
                min_spot_duration = self.restart_overhead * 2
                expected_spot_time = min(work_left / spot_availability, time_left - safety_margin)
                if expected_spot_time > min_spot_duration:
                    return ClusterType.SPOT
        
        # Use on-demand if spot is unavailable or not worthwhile
        if not has_spot or time_left < time_needed_spot + safety_margin:
            return ClusterType.ON_DEMAND
        
        # Default: pause if we're ahead of schedule
        ahead_schedule = time_left > time_needed_ondemand * 1.5
        if ahead_schedule and has_spot:
            # Wait for better spot conditions
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT
    
    def _should_continue_spot(self) -> bool:
        """Check if we should continue using spot based on recent history"""
        if len(self.spot_available_history) < 10:
            return True
        
        # Check recent availability
        recent = self.spot_available_history[-10:]
        availability = sum(recent) / len(recent)
        
        # Continue if availability is good
        return availability > 0.7
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)