import argparse
import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_safety_margin"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.spot_availability_history = []
        self.consecutive_unavailable = 0
        self.last_action = ClusterType.NONE
        self.safety_margin = 0
        self.phase = "normal"  # normal, critical, recovery
        self.critical_threshold = 0
    
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    def _compute_work_left(self):
        if not self.task_done_time:
            return self.task_duration
        return self.task_duration - sum(self.task_done_time)
    
    def _compute_time_left(self):
        return self.deadline - self.env.elapsed_seconds
    
    def _should_enter_critical_mode(self, work_left, time_left):
        # Enter critical if we can't finish even with on-demand considering restart overhead
        if work_left <= 0:
            return False
        
        # Conservative: assume 1 restart might be needed
        effective_time_needed = work_left + self.restart_overhead
        
        # Add safety margin (10% of remaining time)
        safety_buffer = time_left * 0.1
        
        return effective_time_needed + safety_buffer > time_left
    
    def _update_availability_history(self, has_spot):
        self.spot_availability_history.append(1 if has_spot else 0)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        if not has_spot:
            self.consecutive_unavailable += 1
        else:
            self.consecutive_unavailable = 0
    
    def _estimate_spot_reliability(self):
        if not self.spot_availability_history:
            return 0.5
        return sum(self.spot_availability_history) / len(self.spot_availability_history)
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.last_action = last_cluster_type
        self._update_availability_history(has_spot)
        
        work_left = self._compute_work_left()
        time_left = self._compute_time_left()
        
        if work_left <= 0:
            return ClusterType.NONE
        
        # Critical mode decision
        if self._should_enter_critical_mode(work_left, time_left):
            # In critical mode, use on-demand if we need to make progress
            if last_cluster_type == ClusterType.NONE:
                return ClusterType.ON_DEMAND
            elif last_cluster_type == ClusterType.SPOT:
                # If spot fails in critical mode, switch to on-demand
                if not has_spot:
                    return ClusterType.ON_DEMAND
                else:
                    # Stay on spot if available and we have buffer
                    remaining_buffer = time_left - work_left
                    if remaining_buffer > self.restart_overhead * 2:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        # Normal mode decision
        if has_spot:
            reliability = self._estimate_spot_reliability()
            
            # If spot has been reliable recently, use it
            if reliability > 0.7 or self.consecutive_unavailable == 0:
                # Only switch to spot if we're not currently in a restart
                if last_cluster_type == ClusterType.NONE:
                    # Wait a bit if spot was recently unreliable
                    if self.consecutive_unavailable > 0 and reliability < 0.5:
                        return ClusterType.NONE
                    return ClusterType.SPOT
                elif last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                else:  # Switching from on-demand to spot
                    # Only switch if we have time buffer and spot looks good
                    time_needed = work_left + self.restart_overhead
                    if time_left - time_needed > self.restart_overhead * 3 and reliability > 0.8:
                        return ClusterType.SPOT
                    else:
                        return ClusterType.ON_DEMAND
            else:
                # Spot unreliable recently, be conservative
                if last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                elif last_cluster_type == ClusterType.SPOT:
                    # Continue if we're already on spot and it's available
                    return ClusterType.SPOT
                else:
                    # Wait to see if spot stabilizes
                    return ClusterType.NONE
        else:
            # No spot available
            if last_cluster_type == ClusterType.SPOT:
                # Spot failed, switch to on-demand if we have tight deadline
                if time_left - work_left < self.restart_overhead * 2:
                    return ClusterType.ON_DEMAND
                else:
                    # Pause and wait for spot to return
                    return ClusterType.NONE
            elif last_cluster_type == ClusterType.ON_DEMAND:
                # Stay on on-demand if we're already there
                return ClusterType.ON_DEMAND
            else:
                # We're paused, wait a bit for spot
                if self.consecutive_unavailable < 10:  # Don't wait too long
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)