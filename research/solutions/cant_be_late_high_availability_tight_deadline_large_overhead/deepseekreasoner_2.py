import argparse
from enum import Enum
from typing import List, Optional
import math

class ClusterType(Enum):
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"
    NONE = "NONE"

class Solution:
    NAME = "my_solution"
    
    def __init__(self, args):
        self.args = args
        self.env = None
        self.task_duration = None
        self.deadline = None
        self.restart_overhead = None
        self.task_done_time = None
        
        # Internal state
        self.state = {
            'spot_available_history': [],
            'current_restart_overhead': 0,
            'total_work_done': 0,
            'last_work_time': 0,
            'consecutive_spot_failures': 0,
            'spot_reliability': 0.5,
            'emergency_mode': False,
            'time_in_current_session': 0,
            'last_cluster_type': ClusterType.NONE
        }
        
        # Constants
        self.ON_DEMAND_PRICE = 3.06 / 3600  # $/second
        self.SPOT_PRICE = 0.97 / 3600       # $/second
        self.SAFETY_MARGIN = 3600           # 1 hour in seconds
        self.MAX_SPOT_FAILURES = 5
        self.MIN_SPOT_SESSION = 1800        # Minimum spot session: 30 minutes
    
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update internal state
        self.state['last_cluster_type'] = last_cluster_type
        self.state['spot_available_history'].append(has_spot)
        if len(self.state['spot_available_history']) > 100:
            self.state['spot_available_history'].pop(0)
        
        # Update spot reliability estimate
        if len(self.state['spot_available_history']) >= 10:
            available_count = sum(self.state['spot_available_history'][-50:])
            self.state['spot_reliability'] = available_count / min(50, len(self.state['spot_available_history']))
        
        # Calculate remaining work and time
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Check if we're in emergency mode (must finish soon)
        time_needed_on_demand = remaining_work
        time_needed_with_overhead = remaining_work + (self.restart_overhead * 2)
        
        # Emergency mode: switch to on-demand if we're running out of time
        if time_left < time_needed_on_demand + self.SAFETY_MARGIN:
            self.state['emergency_mode'] = True
            return ClusterType.ON_DEMAND
        
        # If we have plenty of time, be conservative
        if time_left > time_needed_with_overhead * 2:
            # Try to use spot if reliable
            if has_spot and self.state['spot_reliability'] > 0.6:
                return ClusterType.SPOT
            elif not has_spot and self.state['last_cluster_type'] == ClusterType.SPOT:
                # Spot was just lost, wait a bit before trying again
                self.state['consecutive_spot_failures'] += 1
                if self.state['consecutive_spot_failures'] < 3:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND if time_left < time_needed_on_demand * 1.5 else ClusterType.NONE
            else:
                return ClusterType.NONE if self.state['consecutive_spot_failures'] < 2 else ClusterType.ON_DEMAND
        
        # Normal mode: balance risk and cost
        if has_spot:
            # Reset failure counter when spot is available
            self.state['consecutive_spot_failures'] = 0
            
            # Calculate risk-adjusted value of spot
            spot_value = self.ON_DEMAND_PRICE - self.SPOT_PRICE
            risk_penalty = (1 - self.state['spot_reliability']) * self.restart_overhead * self.ON_DEMAND_PRICE
            
            # Use spot if it's worth the risk
            if spot_value > risk_penalty * 0.5:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            # Spot not available
            self.state['consecutive_spot_failures'] += 1
            
            if self.state['consecutive_spot_failures'] > self.MAX_SPOT_FAILURES:
                # Too many failures, switch to on-demand
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to come back
                return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)