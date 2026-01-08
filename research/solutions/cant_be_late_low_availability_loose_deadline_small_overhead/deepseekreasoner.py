import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def solve(self, spec_path: str) -> "Solution":
        self._init_state()
        return self
    
    def _init_state(self):
        self.spot_price = 0.97
        self.od_price = 3.06
        self.spot_od_ratio = self.spot_price / self.od_price
        self.restart_penalty_hours = 0.05
        self.task_hours = 48
        self.deadline_hours = 70
        
        self.min_spot_frac = 0.15
        self.min_od_frac = 0.01
        
        self.state = {
            'work_done': 0.0,
            'time_used': 0.0,
            'cost': 0.0,
            'in_restart': False,
            'restart_remaining': 0.0,
            'consecutive_spot_unavailable': 0,
            'spot_available_history': [],
            'last_was_spot': False,
            'spot_availability_rate': 0.5,
            'emergency_od_mode': False
        }
        
        self.plan = {
            'target_spot_fraction': 0.85,
            'od_budget_hours': 5.0,
            'reserved_for_emergency': 2.0
        }
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        gap_hours = self.env.gap_seconds / 3600.0
        elapsed_hours = self.env.elapsed_seconds / 3600.0
        work_done_hours = sum(duration for _, duration in self.task_done_time)
        remaining_hours = self.task_hours - work_done_hours
        time_left_hours = self.deadline_hours - elapsed_hours
        
        self.state['spot_available_history'].append(1 if has_spot else 0)
        if len(self.state['spot_available_history']) > 100:
            self.state['spot_available_history'].pop(0)
        
        if self.state['spot_available_history']:
            self.state['spot_availability_rate'] = np.mean(self.state['spot_available_history'])
        
        if not has_spot:
            self.state['consecutive_spot_unavailable'] += 1
        else:
            self.state['consecutive_spot_unavailable'] = 0
        
        if last_cluster_type == ClusterType.SPOT:
            self.state['last_was_spot'] = True
        else:
            self.state['last_was_spot'] = False
        
        if self.state['in_restart']:
            self.state['restart_remaining'] -= gap_hours
            if self.state['restart_remaining'] <= 0:
                self.state['in_restart'] = False
                self.state['restart_remaining'] = 0.0
        
        if self.state['in_restart']:
            return ClusterType.NONE
        
        if remaining_hours <= 0:
            return ClusterType.NONE
        
        if time_left_hours <= 0:
            return ClusterType.NONE
        
        critical_ratio = remaining_hours / time_left_hours
        
        if critical_ratio > 1.0:
            self.state['emergency_od_mode'] = True
        
        if critical_ratio > 1.1 or time_left_hours < 4.0:
            self.state['emergency_od_mode'] = True
        
        if self.state['emergency_od_mode']:
            return ClusterType.ON_DEMAND
        
        effective_spot_available = has_spot
        
        if not effective_spot_available:
            if self.state['consecutive_spot_unavailable'] > 5:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
        
        spot_priority = self._calculate_spot_priority(
            remaining_hours, time_left_hours, critical_ratio
        )
        
        od_priority = self._calculate_od_priority(
            remaining_hours, time_left_hours, critical_ratio
        )
        
        if od_priority > spot_priority:
            return ClusterType.ON_DEMAND
        elif effective_spot_available:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE
    
    def _calculate_spot_priority(self, remaining_hours, time_left_hours, critical_ratio):
        base_priority = 5.0
        
        time_pressure = max(0, 1.0 - time_left_hours / 20.0)
        base_priority -= time_pressure * 3.0
        
        spot_avail_factor = self.state['spot_availability_rate']
        base_priority += spot_avail_factor * 2.0
        
        if self.state['consecutive_spot_unavailable'] > 0:
            base_priority -= min(self.state['consecutive_spot_unavailable'] * 0.5, 3.0)
        
        if critical_ratio > 0.9:
            base_priority -= 4.0
        elif critical_ratio > 0.7:
            base_priority -= 2.0
        
        if remaining_hours < 5.0:
            base_priority -= 2.0
        
        return base_priority
    
    def _calculate_od_priority(self, remaining_hours, time_left_hours, critical_ratio):
        base_priority = 1.0
        
        if critical_ratio > 1.0:
            base_priority += 10.0
        elif critical_ratio > 0.95:
            base_priority += 8.0
        elif critical_ratio > 0.9:
            base_priority += 6.0
        elif critical_ratio > 0.8:
            base_priority += 4.0
        elif critical_ratio > 0.7:
            base_priority += 2.0
        
        time_pressure = max(0, 1.0 - time_left_hours / 15.0)
        base_priority += time_pressure * 3.0
        
        if remaining_hours < 10.0:
            base_priority += 2.0
        
        if self.state['consecutive_spot_unavailable'] > 10:
            base_priority += 3.0
        
        return base_priority
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)