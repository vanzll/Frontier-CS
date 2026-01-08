import os
import json
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                self.spec = json.load(f)
        else:
            self.spec = {}
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work and time
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # If no work left, do nothing
        if work_remaining <= 0:
            return ClusterType.NONE
            
        # If we're in danger of missing deadline, use on-demand
        safety_factor = 1.1  # 10% safety margin
        min_time_needed = work_remaining * safety_factor
        
        # Account for restart overhead if switching from non-spot to spot
        restart_needed = (last_cluster_type != ClusterType.SPOT and 
                         has_spot and 
                         self.env.elapsed_seconds > 0)
        
        if restart_needed:
            min_time_needed += self.restart_overhead
            
        # Emergency mode: must use on-demand to finish on time
        if time_remaining < min_time_needed:
            return ClusterType.ON_DEMAND
            
        # Calculate aggressive threshold - use on-demand earlier to avoid risk
        # Based on remaining work and time
        progress_ratio = work_done / self.task_duration
        time_ratio = self.env.elapsed_seconds / self.deadline
        
        # If we're falling behind schedule, be more conservative
        if progress_ratio < time_ratio * 0.8:  # 20% behind schedule
            return ClusterType.ON_DEMAND
            
        # Normal operation - use spot when available
        if has_spot:
            # Avoid frequent switching to minimize restart overhead
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            elif last_cluster_type == ClusterType.ON_DEMAND:
                # Once on-demand, stay on it for a while to avoid restart overhead
                # unless we have plenty of time
                if time_remaining > work_remaining + 2 * self.restart_overhead:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:  # NONE or initial state
                return ClusterType.SPOT
        else:
            # Spot not available
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                # If we have time, wait for spot to become available
                # Otherwise use on-demand
                if time_remaining > work_remaining + self.restart_overhead:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)