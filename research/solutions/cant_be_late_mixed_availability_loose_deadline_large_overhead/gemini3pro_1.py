import argparse
from typing import List, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Current state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done
        overhead = self.restart_overhead
        deadline = self.deadline
        
        # Calculate the latest safe time to rely on On-Demand.
        # We need time for:
        # 1. Remaining work (work_rem)
        # 2. Restart overhead (overhead) - assumed incurred if we switch/start OD
        # 3. Safety buffer (safety_margin) - covers time step quantization and slight delays
        safety_margin = 2.0 * gap + 0.1 * overhead
        time_needed_for_od = work_rem + overhead + safety_margin
        
        time_remaining = deadline - elapsed
        
        # Panic Logic: If we are close to the deadline, force On-Demand to ensure completion.
        if time_remaining <= time_needed_for_od:
            return ClusterType.ON_DEMAND
            
        # Slack Logic: If we have plenty of time
        if has_spot:
            # Optimization: If we are already on On-Demand and the remaining work is very small 
            # (less than the overhead time), it is cheaper/faster to just finish on OD 
            # rather than paying the overhead to switch to Spot.
            if last_cluster_type == ClusterType.ON_DEMAND and work_rem < overhead:
                return ClusterType.ON_DEMAND
                
            # Otherwise, prefer Spot to save cost
            return ClusterType.SPOT
            
        # If no Spot is available and we are not in panic mode, wait (NONE) to save money.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)