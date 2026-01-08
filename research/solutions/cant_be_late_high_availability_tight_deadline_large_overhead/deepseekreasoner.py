import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.config = None
        self.spot_price = 0.97
        self.od_price = 3.06
        self.price_ratio = self.od_price / self.spot_price
        self.restart_hours = 0.20
        self.task_hours = 48
        self.deadline_hours = 52
        self.slack_hours = 4
        self.required_rate = 1.0
        self.safety_margin = 1.5
        self.spot_availability = None
        self.conservative = True
        self.plan = None
        self.current_phase = 0
        self.phase_start_time = 0
        self.overhead_remaining = 0
        self.last_action = ClusterType.NONE

    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    self.config = json.load(f)
            except:
                pass
        return self

    def _compute_required_rate(self, elapsed: float, work_done: float) -> float:
        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - elapsed
        if remaining_time <= 0:
            return float('inf')
        return remaining_work / remaining_time

    def _get_work_done(self) -> float:
        return sum(self.task_done_time) if self.task_done_time else 0

    def _should_switch_to_od(self, elapsed: float, work_done: float, has_spot: bool) -> bool:
        if not has_spot:
            return True
            
        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - elapsed
        
        if remaining_time <= 0:
            return True
            
        time_needed = remaining_work
        if self.last_action == ClusterType.NONE:
            time_needed += self.restart_overhead
            
        safety_threshold = remaining_time * 0.9
        
        if time_needed > safety_threshold:
            return True
            
        if self.conservative and remaining_time < self.slack_hours * 3600 * 0.5:
            return True
            
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        work_done = self._get_work_done()
        
        if work_done >= self.task_duration:
            return ClusterType.NONE
            
        if self.overhead_remaining > 0:
            self.overhead_remaining -= self.env.gap_seconds
            if self.overhead_remaining > 0:
                return ClusterType.NONE
        
        if last_cluster_type == ClusterType.NONE and self.last_action != ClusterType.NONE:
            self.overhead_remaining = self.restart_overhead
            return ClusterType.NONE
        
        required_rate = self._compute_required_rate(elapsed, work_done)
        
        if required_rate > 1.2:
            return ClusterType.ON_DEMAND
            
        if self._should_switch_to_od(elapsed, work_done, has_spot):
            if has_spot:
                remaining_work = self.task_duration - work_done
                remaining_time = self.deadline - elapsed
                buffer_time = remaining_time - remaining_work
                
                if buffer_time > self.restart_overhead * 2:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        if has_spot:
            self.last_action = ClusterType.SPOT
            return ClusterType.SPOT
        else:
            remaining_work = self.task_duration - work_done
            remaining_time = self.deadline - elapsed
            
            if remaining_work / remaining_time > 0.95:
                return ClusterType.ON_DEMAND
                
            self.last_action = ClusterType.NONE
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)