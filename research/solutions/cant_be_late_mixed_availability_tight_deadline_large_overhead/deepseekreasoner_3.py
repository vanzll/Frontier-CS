import argparse
import math
from enum import Enum
from typing import List, Tuple

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class InstanceType(Enum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"
    NONE = "none"


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self._state = "IDLE"
        self._restart_remaining = 0.0
        self._work_done = 0.0
        self._spot_availability_history = []
        self._time_history = []
        self._prediction_horizon = 3600  # 1 hour for prediction

    def solve(self, spec_path: str) -> "Solution":
        # Initialize any needed state
        self._state = "IDLE"
        self._restart_remaining = 0.0
        self._work_done = 0.0
        self._spot_availability_history = []
        self._time_history = []
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Record history
        current_time = self.env.elapsed_seconds
        self._spot_availability_history.append(has_spot)
        self._time_history.append(current_time)
        
        # Keep only recent history for efficiency
        if len(self._spot_availability_history) > 1000:
            self._spot_availability_history = self._spot_availability_history[-1000:]
            self._time_history = self._time_history[-1000:]

        # Update work done
        if self.task_done_time:
            self._work_done = sum(end - start for start, end in self.task_done_time)
        work_left = self.task_duration - self._work_done
        
        # Update restart state
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            # Preemption occurred
            self._state = "RESTARTING"
            self._restart_remaining = self.restart_overhead
        elif self._state == "RESTARTING":
            self._restart_remaining -= self.env.gap_seconds
            if self._restart_remaining <= 0:
                self._state = "IDLE"
                self._restart_remaining = 0.0
        elif last_cluster_type != ClusterType.NONE:
            self._state = "RUNNING"
        else:
            self._state = "IDLE"

        # If in restart, wait
        if self._state == "RESTARTING":
            return ClusterType.NONE

        time_left = self.deadline - current_time
        
        # Calculate safe threshold
        spot_price = 0.97 / 3600  # $/second
        ondemand_price = 3.06 / 3600  # $/second
        price_ratio = spot_price / ondemand_price
        
        # Estimate spot availability probability from history
        if self._spot_availability_history:
            spot_prob = sum(self._spot_availability_history) / len(self._spot_availability_history)
        else:
            spot_prob = 0.5  # Default
        
        # Conservative safety margin
        safety_margin = 2.0 * self.restart_overhead
        
        # Calculate minimum time needed
        if work_left <= 0:
            return ClusterType.NONE
        
        # If running out of time, use on-demand
        if time_left < work_left + safety_margin:
            return ClusterType.ON_DEMAND
        
        # If spot is not available, wait if we have time, otherwise use on-demand
        if not has_spot:
            # Check if we can afford to wait
            wait_time_estimate = self._estimate_wait_time()
            if time_left > work_left + wait_time_estimate + safety_margin:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND
        
        # Calculate risk-adjusted cost benefit
        expected_spot_time = work_left / spot_prob if spot_prob > 0 else float('inf')
        expected_cost_spot = expected_spot_time * spot_price
        expected_cost_ondemand = work_left * ondemand_price
        
        # Use spot if it's significantly cheaper and we have buffer for potential preemptions
        buffer_needed = self.restart_overhead * (work_left / (self.env.gap_seconds * 10))  # Estimate number of preemptions
        if (expected_cost_spot < expected_cost_ondemand * 0.7 and 
            time_left > work_left + buffer_needed + safety_margin):
            return ClusterType.SPOT
        
        # Default: use on-demand for reliability
        return ClusterType.ON_DEMAND
    
    def _estimate_wait_time(self) -> float:
        """Estimate time until spot becomes available based on history"""
        if not self._spot_availability_history:
            return self.restart_overhead * 2
        
        # Look for patterns in availability
        recent_window = min(50, len(self._spot_availability_history))
        recent = self._spot_availability_history[-recent_window:]
        
        # If spot was recently available, assume it will return soon
        if any(recent):
            avg_gap = self._calculate_average_gap()
            return min(avg_gap * 2, 3600)  # Cap at 1 hour
        
        return self.restart_overhead * 3
    
    def _calculate_average_gap(self) -> float:
        """Calculate average gap between spot availability periods"""
        if len(self._spot_availability_history) < 2:
            return self.env.gap_seconds * 10
        
        gaps = []
        last_available = None
        
        for i, avail in enumerate(self._spot_availability_history):
            if avail:
                if last_available is not None:
                    gap = self._time_history[i] - last_available
                    gaps.append(gap)
                last_available = self._time_history[i]
        
        if not gaps:
            return self.env.gap_seconds * 10
        
        return sum(gaps) / len(gaps)

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)