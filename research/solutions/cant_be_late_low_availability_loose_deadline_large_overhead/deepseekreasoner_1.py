import math
import json
from typing import List, Tuple, Optional
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.config = None
        self.min_cost_per_hour = None
        self.max_cost_per_hour = None
        self.restart_hours = None
        self.deadline_hours = None
        self.task_hours = None
        self.horizon = None
        self.spot_price = None
        self.ondemand_price = None
        self.state = {}
        
    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path, 'r') as f:
            self.config = json.load(f)
            
        self.spot_price = 0.97  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.restart_hours = 0.20  # hours
        self.task_hours = 48.0  # hours
        self.deadline_hours = 70.0  # hours
        self.horizon = int(self.deadline_hours * 3600)  # seconds
        
        self.min_cost_per_hour = self.spot_price
        self.max_cost_per_hour = self.ondemand_price
        
        self.state = {
            'remaining_work': self.task_hours * 3600,  # seconds
            'restart_timer': 0,
            'current_type': None,
            'spot_availability_history': [],
            'time_elapsed': 0,
            'work_done': 0,
            'backup_plan_triggered': False
        }
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        step_seconds = self.env.gap_seconds
        current_time = self.env.elapsed_seconds
        
        # Update state
        self.state['time_elapsed'] = current_time
        self.state['current_type'] = last_cluster_type
        self.state['spot_availability_history'].append(has_spot)
        
        # Update restart timer
        if self.state['restart_timer'] > 0:
            self.state['restart_timer'] = max(0, self.state['restart_timer'] - step_seconds)
        
        # Calculate remaining work from environment
        total_work_done = 0
        for start, end in self.task_done_time:
            total_work_done += (end - start)
        self.state['work_done'] = total_work_done
        self.state['remaining_work'] = max(0, self.task_duration - total_work_done)
        
        # Calculate time remaining
        time_remaining = self.deadline - current_time
        work_needed = self.state['remaining_work']
        
        # If work is done, use NONE
        if work_needed <= 0:
            return ClusterType.NONE
        
        # If restart timer is active, we can't do work
        if self.state['restart_timer'] > 0:
            return ClusterType.NONE
        
        # Check if we need to trigger emergency mode
        # Estimate time needed for remaining work with various strategies
        work_hours_needed = work_needed / 3600
        
        # Conservative estimate: all work on on-demand
        ondemand_time_needed = work_hours_needed
        
        # Optimistic estimate: all work on spot (no interruptions)
        spot_time_needed = work_hours_needed
        
        # Account for restart overhead in spot estimate
        # Based on historical availability pattern
        recent_availability = self.state['spot_availability_history'][-min(50, len(self.state['spot_availability_history'])):]
        if recent_availability:
            spot_availability = sum(recent_availability) / len(recent_availability)
        else:
            spot_availability = 0.5  # conservative default
            
        # Estimate interruptions per hour based on availability
        # Lower availability means more frequent interruptions
        expected_interruptions = max(0, (1 - spot_availability) * (work_hours_needed / 0.5))
        spot_time_with_overhead = work_hours_needed + expected_interruptions * self.restart_hours
        
        # Determine if we need to switch to on-demand to meet deadline
        safety_margin = 2.0  # hours
        critical_time = time_remaining / 3600
        
        # If we're running out of time, switch to on-demand
        if critical_time < ondemand_time_needed + safety_margin:
            self.state['backup_plan_triggered'] = True
            return ClusterType.ON_DEMAND
        
        # If spot is available and we have time buffer, use spot
        if has_spot and critical_time > spot_time_with_overhead + safety_margin:
            return ClusterType.SPOT
        
        # If no spot but we have time, wait for spot
        if not has_spot and critical_time > ondemand_time_needed + safety_margin * 2:
            return ClusterType.NONE
        
        # Otherwise use on-demand
        return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)