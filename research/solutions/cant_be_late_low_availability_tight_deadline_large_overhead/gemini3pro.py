from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def __init__(self, args):
        super().__init__(args)

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current progress and time
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_rem = self.task_duration - work_done
        
        # If task is already done, do nothing
        if work_rem <= 0:
            return ClusterType.NONE
            
        elapsed = self.env.elapsed_seconds
        time_rem = self.deadline - elapsed
        slack = time_rem - work_rem
        
        overhead = self.restart_overhead
        
        # Safety buffer (15 minutes) to account for time step granularity 
        # and ensure we don't miss the deadline due to small delays.
        buffer = 900.0
        
        # Thresholds definition:
        # 1. Panic Threshold: We must be running (or start) On-Demand if slack drops below this.
        #    We account for the overhead time required to start the instance.
        thresh_panic = overhead + buffer
        
        # 2. Spot Entry Threshold: If starting fresh or recovering, only choose Spot
        #    if we have enough slack to pay the entry overhead AND potentially fall back 
        #    to OD (paying overhead again) if Spot fails immediately.
        thresh_spot_entry = 2 * overhead + buffer
        
        # 3. Switch Threshold: If currently safe on OD, only switch to Spot if slack is 
        #    very high, justifying the overhead cost of switching and the risk.
        thresh_switch_from_od = 3 * overhead + buffer

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                # Keep using Spot if we are already on it
                return ClusterType.SPOT
                
            elif last_cluster_type == ClusterType.ON_DEMAND:
                # If on OD, only switch if we have abundance of slack
                if slack > thresh_switch_from_od:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
                    
            else: # NONE (Idle or Interrupted)
                # Try Spot if we have enough slack to handle potential failure
                if slack > thresh_spot_entry:
                    return ClusterType.SPOT
                else:
                    # Slack is tight; forced to use OD to ensure completion
                    return ClusterType.ON_DEMAND
        else:
            # Spot instances are not available
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Continue OD to avoid paying restart overhead again later
                return ClusterType.ON_DEMAND
            else:
                # We are interrupted or waiting
                if slack < thresh_panic:
                    # Critical: Must start OD to meet deadline
                    return ClusterType.ON_DEMAND
                else:
                    # Plenty of slack: Wait for Spot to become available to save cost
                    return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)