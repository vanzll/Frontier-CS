import os
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

@dataclass
class WorkSegment:
    start: float
    end: float
    type: ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.work_history: List[WorkSegment] = []
        self.current_segment: Optional[WorkSegment] = None
        self.last_decision: ClusterType = ClusterType.NONE
        self.restart_timer: float = 0.0
        self.total_work_done: float = 0.0
        self.safety_margin: float = 3600.0  # 1 hour safety margin
        self.spot_availability_history: List[bool] = []
        self.spot_unavailable_streak: int = 0
        self.panic_mode: bool = False
        
    def solve(self, spec_path: str) -> "Solution":
        """Optional initialization"""
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                config = json.load(f)
                if 'safety_margin' in config:
                    self.safety_margin = config['safety_margin']
        return self
    
    def _update_work_segment(self, cluster_type: ClusterType, current_time: float):
        """Update current work segment based on cluster type"""
        if cluster_type == ClusterType.NONE:
            if self.current_segment:
                self.current_segment.end = current_time
                self.work_history.append(self.current_segment)
                self.current_segment = None
        else:
            if self.current_segment and self.current_segment.type == cluster_type:
                self.current_segment.end = current_time
            else:
                if self.current_segment:
                    self.current_segment.end = current_time
                    self.work_history.append(self.current_segment)
                self.current_segment = WorkSegment(current_time, current_time, cluster_type)
    
    def _calculate_remaining_time(self, current_time: float) -> Tuple[float, float]:
        """Calculate remaining work and time until deadline"""
        total_done = sum(seg.end - seg.start for seg in self.work_history)
        if self.current_segment:
            total_done += current_time - self.current_segment.start
        
        remaining_work = self.task_duration - total_done
        time_until_deadline = self.deadline - current_time
        
        return remaining_work, time_until_deadline
    
    def _get_required_rate(self, remaining_work: float, time_until_deadline: float) -> float:
        """Calculate required work rate to finish before deadline"""
        if time_until_deadline <= 0:
            return float('inf')
        return remaining_work / time_until_deadline
    
    def _should_panic(self, remaining_work: float, time_until_deadline: float) -> bool:
        """Determine if we need to switch to on-demand to meet deadline"""
        if remaining_work <= 0:
            return False
        
        # Conservative estimate: account for potential restart overhead
        conservative_work = remaining_work + self.restart_overhead
        conservative_time = time_until_deadline - self.safety_margin
        
        if conservative_time <= 0:
            return True
        
        required_rate = conservative_work / conservative_time
        
        # If required rate is too high, go to panic mode
        return required_rate > 0.95  # Need to work 95% of remaining time
    
    def _analyze_spot_pattern(self) -> Tuple[float, float]:
        """Analyze spot availability pattern for decision making"""
        if len(self.spot_availability_history) < 10:
            return 0.5, 0.0  # Default values if not enough history
        
        available_count = sum(self.spot_availability_history[-100:])  # Last 100 steps
        availability = available_count / min(100, len(self.spot_availability_history))
        
        # Calculate streak of unavailability
        current_streak = 0
        for available in reversed(self.spot_availability_history[-20:]):  # Last 20 steps
            if not available:
                current_streak += 1
            else:
                break
        
        return availability, current_streak
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        
        # Update spot availability history
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 1000:
            self.spot_availability_history.pop(0)
        
        # Update current streak of spot unavailability
        if not has_spot:
            self.spot_unavailable_streak += 1
        else:
            self.spot_unavailable_streak = 0
        
        # Update work segment based on last decision
        self._update_work_segment(last_cluster_type, current_time)
        
        # Calculate remaining work and time
        remaining_work, time_until_deadline = self._calculate_remaining_time(current_time)
        
        # Check if we're done
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Handle restart overhead
        if self.restart_timer > 0:
            self.restart_timer -= self.env.gap_seconds
            return ClusterType.NONE
        
        # Analyze spot pattern
        spot_availability, current_unavailable_streak = self._analyze_spot_pattern()
        
        # Check if we need to panic (switch to on-demand to meet deadline)
        self.panic_mode = self._should_panic(remaining_work, time_until_deadline)
        
        # Decision logic
        decision = ClusterType.NONE
        
        if self.panic_mode:
            # In panic mode, use on-demand to ensure we meet deadline
            decision = ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.NONE:
                self.restart_timer = self.restart_overhead
                return ClusterType.NONE
        else:
            # Normal mode: try to use spot when available
            if has_spot:
                # Use spot if it's reasonably available
                if spot_availability > 0.3 or current_unavailable_streak == 0:
                    decision = ClusterType.SPOT
                    if last_cluster_type == ClusterType.NONE:
                        self.restart_timer = self.restart_overhead
                        return ClusterType.NONE
                else:
                    # Spot has been unreliable recently, pause and wait
                    decision = ClusterType.NONE
            else:
                # Spot not available
                if current_unavailable_streak > 5 and time_until_deadline > remaining_work + self.restart_overhead + 3600:
                    # Spot has been unavailable for a while, but we have time
                    decision = ClusterType.NONE
                else:
                    # Use on-demand if we can't wait
                    decision = ClusterType.ON_DEMAND
                    if last_cluster_type == ClusterType.NONE:
                        self.restart_timer = self.restart_overhead
                        return ClusterType.NONE
        
        # Safety check: don't return SPOT if not available
        if decision == ClusterType.SPOT and not has_spot:
            decision = ClusterType.NONE
        
        self.last_decision = decision
        return decision
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)