import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
from collections import deque


class Solution(Strategy):
    NAME = "my_solution"
    
    def solve(self, spec_path: str) -> "Solution":
        # Read configuration if needed
        try:
            with open(spec_path, 'r') as f:
                # Parse configuration if provided
                pass
        except:
            pass
        
        # Initialize state
        self.spot_history = deque(maxlen=100)
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.last_spot_availability = None
        self.spot_availability_rate = 0.5
        self.emergency_mode = False
        self.on_demand_start_time = None
        self.last_work_progress = 0
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update spot availability tracking
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 0:
            self.spot_availability_rate = sum(self.spot_history) / len(self.spot_history)
        
        # Update consecutive counters
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0
        
        self.last_spot_availability = has_spot
        
        # Calculate progress and time remaining
        total_done = sum(duration for _, duration in self.task_done_time)
        remaining_work = self.task_duration - total_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Calculate work rate (accounting for restart overhead)
        effective_time_left = time_left
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            effective_time_left -= self.restart_overhead
        
        # Emergency mode: if we're running out of time, use on-demand
        required_rate = remaining_work / max(effective_time_left, 0.1)
        if required_rate > 0.95:  # Need to work >95% of remaining time
            self.emergency_mode = True
        
        # Calculate progress since last step
        current_progress = total_done
        progress_delta = current_progress - self.last_work_progress
        self.last_work_progress = current_progress
        
        # Track if we're making progress
        if progress_delta < self.env.gap_seconds * 0.1 and last_cluster_type != ClusterType.NONE:
            # Not making good progress, consider switching
            if self.on_demand_start_time is None:
                self.on_demand_start_time = self.env.elapsed_seconds
        else:
            self.on_demand_start_time = None
        
        # Strategy decision logic
        if self.emergency_mode:
            # Emergency: must use on-demand to finish on time
            return ClusterType.ON_DEMAND
        
        # Calculate if we have time buffer
        buffer_needed = self.restart_overhead * 2  # Safety buffer
        time_buffer = time_left - remaining_work
        
        if time_buffer < buffer_needed:
            # Not enough buffer, use on-demand
            return ClusterType.ON_DEMAND
        
        # Check spot availability patterns
        if has_spot:
            # Good spot availability streak
            if self.consecutive_spot_available >= 10:
                # Long streak of availability, use spot
                return ClusterType.SPOT
            elif self.spot_availability_rate > 0.7:
                # High historical availability
                return ClusterType.SPOT
            else:
                # Mixed availability, be conservative
                if time_buffer > self.restart_overhead * 3:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        else:
            # No spot available
            if self.consecutive_spot_unavailable >= 5:
                # Long unavailability streak, use on-demand
                return ClusterType.ON_DEMAND
            elif time_buffer < self.restart_overhead * 4:
                # Low time buffer, use on-demand
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)