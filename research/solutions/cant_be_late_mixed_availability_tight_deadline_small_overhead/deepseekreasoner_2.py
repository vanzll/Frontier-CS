import sys
import os
import json
import math
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Import required types from the provided API
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

@dataclass
class JobState:
    """Track job progress and state"""
    work_done: float = 0.0
    remaining_time: float = 0.0
    current_cluster: ClusterType = ClusterType.NONE
    in_restart: bool = False
    restart_remaining: float = 0.0
    consecutive_none: int = 0
    
class Solution(Strategy):
    NAME = "optimized_spot_scheduler"
    
    def __init__(self, args):
        super().__init__(args)
        self.state = JobState()
        self.config = {}
        self.spot_history = []
        self.spot_availability_rate = 0.5
        self.critical_threshold = 0.2
        self.safety_margin = 0.1
        self.min_consecutive_spot = 3
        self.od_price = 3.06 / 3600  # $/second
        self.spot_price = 0.97 / 3600  # $/second
        self.step_count = 0
        self.max_steps = 1000000
        
    def solve(self, spec_path: str) -> "Solution":
        """Initialize from specification file"""
        try:
            if os.path.exists(spec_path):
                with open(spec_path, 'r') as f:
                    self.config = json.load(f)
        except:
            pass
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Main decision logic"""
        self.step_count += 1
        if self.step_count > self.max_steps:
            return ClusterType.ON_DEMAND
            
        # Update spot history
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 100:
            self.spot_history.pop(0)
        if len(self.spot_history) > 0:
            self.spot_availability_rate = sum(self.spot_history) / len(self.spot_history)
        
        # Calculate progress
        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        work_remaining = self.task_duration - work_done
        
        # Check if we're in trouble
        if work_remaining <= 0:
            return ClusterType.NONE
            
        # Calculate safe completion time
        safe_time_needed = work_remaining + self.restart_overhead * 2
        
        # Emergency mode: we're running out of time
        if time_left <= safe_time_needed * (1 + self.safety_margin):
            # Must use on-demand to guarantee completion
            if last_cluster_type != ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        # Calculate efficiency metrics
        spot_efficiency = self.spot_availability_rate
        time_ratio = time_left / self.deadline
        work_ratio = work_remaining / self.task_duration
        
        # Determine if we should use spot
        can_use_spot = has_spot and spot_efficiency > 0.3
        
        if can_use_spot:
            # Check if we have enough time for spot
            expected_spot_time = work_remaining / spot_efficiency
            expected_restarts = max(1, expected_spot_time * (1 - spot_efficiency) / 3600)
            total_expected_time = expected_spot_time + expected_restarts * self.restart_overhead
            
            if total_expected_time < time_left * 0.8:
                # Use spot if we have good availability and time buffer
                if last_cluster_type == ClusterType.SPOT:
                    # Stay on spot
                    return ClusterType.SPOT
                elif last_cluster_type == ClusterType.NONE:
                    # Start spot if we've been waiting
                    if self.state.consecutive_none > 2:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.NONE
                else:
                    # Switch to spot from on-demand
                    if spot_efficiency > 0.6 and time_left > total_expected_time * 1.5:
                        return ClusterType.SPOT
                    else:
                        return last_cluster_type
            else:
                # Not enough time for spot, use on-demand
                if last_cluster_type != ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.ON_DEMAND
        else:
            # Spot not available
            if has_spot and spot_efficiency <= 0.3:
                # Spot available but unreliable
                if work_ratio > 0.7 or time_ratio < 0.3:
                    # Much work remaining or little time left
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE
            elif not has_spot:
                # No spot available
                if work_ratio > self.critical_threshold or time_ratio < 0.4:
                    # Critical: use on-demand
                    if last_cluster_type != ClusterType.ON_DEMAND:
                        return ClusterType.ON_DEMAND
                    else:
                        return ClusterType.ON_DEMAND
                else:
                    # Wait for spot
                    return ClusterType.NONE
            else:
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)