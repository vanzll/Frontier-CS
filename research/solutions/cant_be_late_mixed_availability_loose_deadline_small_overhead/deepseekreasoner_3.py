import json
import os
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
from typing import Optional
import math

class Solution(Strategy):
    NAME = "adaptive_safety_margin"
    
    def __init__(self, args):
        super().__init__(args)
        self.initialized = False
        self.safety_margin = None
        self.min_safety_margin = None
        self.max_safety_margin = None
        self.spot_reliability = None
        self.spot_availability_history = []
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.spot_switch_count = 0
        self.last_spot_state = None
        self.time_since_last_switch = 0
        self.work_remaining = None
        self.time_remaining = None
        self.total_work_done = 0
        self.last_action = ClusterType.NONE
        self.restart_pending = False
        self.restart_time_remaining = 0
        
    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
        except:
            spec = {}
        
        # Initialize adaptive parameters
        self.safety_margin = 3600  # Start with 1 hour safety margin
        self.min_safety_margin = 1800  # 30 minutes minimum
        self.max_safety_margin = 7200  # 2 hours maximum
        self.spot_reliability = 0.5  # Initial estimate
        self.spot_availability_history = []
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.spot_switch_count = 0
        self.last_spot_state = None
        self.time_since_last_switch = 0
        self.work_remaining = self.task_duration
        self.total_work_done = 0
        self.last_action = ClusterType.NONE
        self.restart_pending = False
        self.restart_time_remaining = 0
        self.initialized = True
        
        return self
    
    def _update_spot_history(self, has_spot: bool):
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        # Update consecutive counters
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0
            
        # Update spot reliability estimate
        if len(self.spot_availability_history) > 0:
            available_count = sum(self.spot_availability_history)
            self.spot_reliability = available_count / len(self.spot_availability_history)
    
    def _update_safety_margin(self):
        # Adjust safety margin based on spot reliability
        if self.spot_reliability > 0.7:
            # High reliability - can use smaller safety margin
            self.safety_margin = max(self.min_safety_margin, 
                                   3600 - int((self.spot_reliability - 0.7) * 2000))
        elif self.spot_reliability < 0.3:
            # Low reliability - need larger safety margin
            self.safety_margin = min(self.max_safety_margin,
                                   3600 + int((0.3 - self.spot_reliability) * 3000))
        else:
            # Medium reliability
            self.safety_margin = 3600
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized:
            self.initialized = True
            self.work_remaining = self.task_duration
            self.total_work_done = 0
            self.safety_margin = 3600
        
        # Update spot availability history
        self._update_spot_history(has_spot)
        self._update_safety_margin()
        
        # Update work remaining
        if len(self.task_done_time) > 0:
            recent_work = sum(segment[1] - segment[0] for segment in self.task_done_time[-10:])
            if recent_work > 0:
                self.total_work_done += recent_work
                self.work_remaining = max(0, self.task_duration - self.total_work_done)
        
        # Update time tracking
        self.time_remaining = self.deadline - self.env.elapsed_seconds
        
        # Update restart tracking
        if self.restart_pending:
            self.restart_time_remaining = max(0, self.restart_time_remaining - self.env.gap_seconds)
            if self.restart_time_remaining <= 0:
                self.restart_pending = False
        
        # Check if we need to handle restart overhead
        if (last_cluster_type != ClusterType.SPOT and 
            self.env.cluster_type == ClusterType.SPOT):
            # Starting spot after non-spot, need restart
            self.restart_pending = True
            self.restart_time_remaining = self.restart_overhead
        
        # Emergency mode: if we're running out of time, switch to on-demand
        critical_time = self.work_remaining + (self.restart_overhead if not self.restart_pending else 0)
        if self.time_remaining < critical_time * 1.2:
            return ClusterType.ON_DEMAND
        
        # Calculate time pressure
        time_pressure = self.work_remaining / max(1, self.time_remaining - self.safety_margin)
        
        # Aggressive spot use when we have lots of time
        if time_pressure < 0.7 and has_spot:
            # Use spot if available and we're not in restart
            if not self.restart_pending:
                return ClusterType.SPOT
            # If in restart, consider waiting or switching
            elif self.time_remaining - self.work_remaining > self.restart_time_remaining + 1800:
                return ClusterType.SPOT
        
        # Conservative approach when time is tighter
        if time_pressure >= 0.7:
            if time_pressure > 0.9 or not has_spot:
                return ClusterType.ON_DEMAND
            elif has_spot and not self.restart_pending:
                return ClusterType.SPOT
        
        # Default: use spot if available, otherwise on-demand
        if has_spot and not self.restart_pending:
            return ClusterType.SPOT
        
        return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)