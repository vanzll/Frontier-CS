import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import argparse

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize state variables
        self.spot_history = []
        self.spot_availability_rate = 0.5  # Initial estimate
        self.min_spot_window = 1  # Minimum consecutive spot steps to use spot
        self.safety_margin = 2.0  # Hours of safety margin for deadline
        self.conservative_threshold = 0.8  # When to switch to OD
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Track spot availability history
        self.spot_history.append(has_spot)
        if len(self.spot_history) > 100:
            self.spot_history.pop(0)
            self.spot_availability_rate = np.mean(self.spot_history)

        # Calculate critical metrics
        elapsed = self.env.elapsed_seconds / 3600.0  # Convert to hours
        deadline_hours = self.deadline / 3600.0
        task_hours = self.task_duration / 3600.0
        restart_hours = self.restart_overhead / 3600.0
        
        # Calculate work completed
        if self.task_done_time:
            completed = sum(self.task_done_time) / 3600.0
        else:
            completed = 0.0
        
        remaining_work = task_hours - completed
        time_left = deadline_hours - elapsed
        
        # If work is done, stop everything
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Calculate required completion rate
        if time_left <= 0:
            return ClusterType.ON_DEMAND  # Emergency mode
        
        required_rate = remaining_work / time_left
        
        # Critical situation: must use on-demand
        if time_left < self.safety_margin or required_rate > 0.9:
            if has_spot and time_left > restart_hours * 2:
                # In critical mode but still try spot if safe
                recent_spot = np.mean(self.spot_history[-10:]) if len(self.spot_history) >= 10 else self.spot_availability_rate
                if recent_spot > 0.7:
                    return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Calculate efficiency of spot vs on-demand
        # Account for restart overhead in effective spot rate
        if self.spot_availability_rate > 0:
            effective_spot_rate = self.spot_availability_rate / (1 + restart_hours * required_rate)
        else:
            effective_spot_rate = 0
        
        # Decide based on current state and conditions
        if last_cluster_type == ClusterType.SPOT:
            # Currently on spot
            if has_spot:
                # Spot still available
                if required_rate > self.conservative_threshold:
                    # Need higher reliability
                    if self.spot_availability_rate > 0.85:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
                else:
                    return ClusterType.SPOT
            else:
                # Spot lost, consider restart or switch
                if time_left > restart_hours * 3 and remaining_work > restart_hours:
                    # Enough time for potential restart
                    recent_unavailable = len([x for x in self.spot_history[-5:] if not x]) if len(self.spot_history) >= 5 else 0
                    if recent_unavailable <= 2:
                        # Temporary glitch, wait for spot
                        return ClusterType.NONE
                return ClusterType.ON_DEMAND
        
        elif last_cluster_type == ClusterType.ON_DEMAND:
            # Currently on on-demand
            if required_rate < 0.5 and has_spot and time_left > restart_hours * 4:
                # Ahead of schedule, try to save cost with spot
                if self.spot_availability_rate > 0.6:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        else:  # NONE
            # Currently paused
            if required_rate > 0.1:  # Have work to do
                if has_spot and time_left > restart_hours * 3:
                    if self.spot_availability_rate > 0.65:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
                else:
                    return ClusterType.ON_DEMAND
            else:
                # No urgent work
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)