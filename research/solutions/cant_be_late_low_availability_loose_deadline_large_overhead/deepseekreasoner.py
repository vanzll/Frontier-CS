import math
import sys
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cost_aware_scheduler"
    
    def __init__(self, args):
        super().__init__(args)
        self.spec_config = {}
        self.work_done = 0.0
        self.last_work_check = 0.0
        self.in_restart_overhead = False
        self.restart_end_time = 0.0
        self.spot_history = []
        self.last_decision = None
        self.time_since_last_work = 0.0
        self.expected_work_rate = 1.0
        self.consecutive_no_work = 0
        self.conservative_mode = False
        self.emergency_mode = False
        self.switched_to_od = False
        self.min_slack_for_spot = 0.0
        
    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        self.spec_config[key.strip()] = value.strip()
        except:
            pass
        
        self.min_slack_for_spot = float(self.spec_config.get('min_slack_for_spot', '0.5'))
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        restart_overhead = self.restart_overhead
        total_work_needed = self.task_duration
        
        if current_time == 0:
            self.work_done = 0.0
            self.last_work_check = 0.0
            self.in_restart_overhead = False
            self.restart_end_time = 0.0
            self.spot_history = []
            self.last_decision = None
            self.time_since_last_work = 0.0
            self.expected_work_rate = 1.0
            self.consecutive_no_work = 0
            self.conservative_mode = False
            self.emergency_mode = False
            self.switched_to_od = False
        
        self.spot_history.append(has_spot)
        if len(self.spot_history) > 10:
            self.spot_history.pop(0)
        
        if last_cluster_type == ClusterType.NONE:
            self.time_since_last_work += gap
            self.consecutive_no_work += 1
        elif last_cluster_type != ClusterType.NONE:
            self.time_since_last_work = 0.0
            self.consecutive_no_work = 0
        
        if self.task_done_time:
            total_done = sum(self.task_done_time)
            if total_done > self.work_done:
                self.work_done = total_done
                self.last_work_check = current_time
                self.expected_work_rate = max(0.1, self.work_done / max(1.0, current_time - self.time_since_last_work))
        
        work_remaining = total_work_needed - self.work_done
        time_remaining = deadline - current_time
        
        if self.in_restart_overhead and current_time >= self.restart_end_time:
            self.in_restart_overhead = False
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.in_restart_overhead = True
            self.restart_end_time = current_time + restart_overhead
        
        if work_remaining <= 0:
            return ClusterType.NONE
        
        if time_remaining <= 0:
            return ClusterType.ON_DEMAND
        
        time_needed = work_remaining / self.expected_work_rate
        
        critical_threshold = time_needed * 1.1
        
        if time_remaining < critical_threshold:
            self.emergency_mode = True
        
        if self.emergency_mode:
            return ClusterType.ON_DEMAND
        
        available_slack = time_remaining - time_needed
        
        if available_slack < 2 * restart_overhead:
            self.conservative_mode = True
        
        if self.conservative_mode:
            if has_spot and available_slack > restart_overhead * 1.5:
                if not self.in_restart_overhead:
                    return ClusterType.SPOT
                else:
                    if current_time >= self.restart_end_time:
                        return ClusterType.SPOT
                    return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND
        
        spot_availability = sum(self.spot_history) / len(self.spot_history) if self.spot_history else 0.0
        
        if has_spot and available_slack > restart_overhead * 3:
            if spot_availability > 0.3:
                if self.in_restart_overhead:
                    if current_time >= self.restart_end_time:
                        return ClusterType.SPOT
                    return ClusterType.NONE
                else:
                    if last_cluster_type != ClusterType.SPOT:
                        self.in_restart_overhead = True
                        self.restart_end_time = current_time + restart_overhead
                        return ClusterType.SPOT
                    else:
                        return ClusterType.SPOT
            elif spot_availability > 0.1:
                if not self.in_restart_overhead and available_slack > restart_overhead * 4:
                    if last_cluster_type != ClusterType.SPOT:
                        self.in_restart_overhead = True
                        self.restart_end_time = current_time + restart_overhead
                        return ClusterType.SPOT
                    else:
                        return ClusterType.SPOT
        
        if has_spot and available_slack > restart_overhead * 6:
            if not self.in_restart_overhead:
                if last_cluster_type != ClusterType.SPOT:
                    self.in_restart_overhead = True
                    self.restart_end_time = current_time + restart_overhead
                    return ClusterType.SPOT
                else:
                    return ClusterType.SPOT
            else:
                if current_time >= self.restart_end_time:
                    return ClusterType.SPOT
                return ClusterType.NONE
        
        if time_remaining > time_needed * 1.5 and not has_spot:
            return ClusterType.NONE
        
        if time_remaining <= time_needed * 1.2:
            return ClusterType.ON_DEMAND
        
        return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)