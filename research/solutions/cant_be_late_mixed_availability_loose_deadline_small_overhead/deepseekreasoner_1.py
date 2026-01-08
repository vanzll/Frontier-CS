import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        remaining_work = self.task_duration - sum(self.task_done_time)
        time_left = self.deadline - elapsed
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        if time_left <= 0:
            return ClusterType.ON_DEMAND
        
        required_rate = remaining_work / time_left if time_left > 0 else float('inf')
        
        if required_rate > 1.0:
            return ClusterType.ON_DEMAND
        
        spot_price = 0.97
        ondemand_price = 3.06
        restart_cost = self.restart_overhead * spot_price
        
        remaining_steps = time_left / self.env.gap_seconds if self.env.gap_seconds > 0 else 0
        
        if remaining_steps < 10:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            
            if remaining_work / time_left > 0.7:
                safety_margin = max(self.restart_overhead * 2, 3600)
                if time_left - remaining_work < safety_margin:
                    return ClusterType.ON_DEMAND
            
            if remaining_work <= self.env.gap_seconds:
                return ClusterType.ON_DEMAND
            
            spot_expected_progress = self.env.gap_seconds
            ondemand_expected_progress = self.env.gap_seconds
            
            expected_spot_cost = spot_price * self.env.gap_seconds / 3600
            expected_ondemand_cost = ondemand_price * self.env.gap_seconds / 3600
            
            conservative_threshold = 0.4
            aggressive_threshold = 0.15
            
            time_slack_ratio = (time_left - remaining_work) / remaining_work if remaining_work > 0 else float('inf')
            
            if time_slack_ratio < conservative_threshold:
                if time_slack_ratio < aggressive_threshold:
                    return ClusterType.ON_DEMAND
                elif last_cluster_type != ClusterType.SPOT:
                    return ClusterType.SPOT
                else:
                    return ClusterType.SPOT
            else:
                if last_cluster_type == ClusterType.NONE:
                    restart_risk = min(0.3, 1.0 / max(1, remaining_steps / 10))
                    if elapsed % (3600 * 4) < self.env.gap_seconds * 2:
                        restart_risk *= 1.5
                    
                    if restart_risk * restart_cost > expected_spot_cost:
                        return ClusterType.ON_DEMAND
                
                return ClusterType.SPOT
        else:
            if last_cluster_type == ClusterType.SPOT:
                if remaining_work / time_left > 0.6:
                    return ClusterType.ON_DEMAND
                else:
                    wait_steps = min(5, int(remaining_steps * 0.1))
                    if elapsed % (self.env.gap_seconds * wait_steps) < self.env.gap_seconds:
                        return ClusterType.NONE
                    else:
                        return ClusterType.ON_DEMAND
            elif remaining_work / time_left > 0.8:
                return ClusterType.ON_DEMAND
            else:
                if remaining_steps > 20 and elapsed % (self.env.gap_seconds * 3) < self.env.gap_seconds * 2:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)