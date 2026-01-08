import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "optimal_slack_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the cluster type for the next time step based on a slack-based strategy.
        Prioritizes meeting the deadline (On-Demand) when slack is low, and cost savings (Spot/None) otherwise.
        """
        # 1. Gather environment state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        
        # 2. Calculate remaining work
        # task_done_time is a list of completed work segment durations
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done
        
        # If work is effectively done, stop
        if work_rem <= 1e-7:
            return ClusterType.NONE
            
        time_remaining = deadline - elapsed

        # 3. Calculate "Panic Threshold"
        # We calculate the time required to finish if we switch to (or stay on) On-Demand NOW.
        # OD is the reliable fallback. We must switch to it before it's too late.
        
        # If we are not currently on OD, we pay the restart overhead to switch.
        # If we are already on OD, we assume no new overhead is incurred to continue.
        switch_cost = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead
        
        time_needed_od = work_rem + switch_cost
        slack = time_remaining - time_needed_od
        
        # 4. Define Safety Buffer
        # We need a buffer to account for the discrete time steps (gap_seconds).
        # We decision is made at t, but covers [t, t+gap].
        # We also want enough buffer to absorb one failed Spot attempt (which costs time + overhead).
        # Buffer = 2 * gap + overhead ensures robust safety against deadline misses.
        safety_buffer = 2 * gap + overhead
        
        # 5. Decision Logic
        
        # CRITICAL: If slack is running out, force On-Demand to guarantee deadline.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # COST-SAVING: If we have plenty of slack, prefer Spot.
        if has_spot:
            return ClusterType.SPOT
            
        # WAIT: If Spot is unavailable but we have slack, wait (NONE) to save money
        # rather than burning expensive On-Demand hours unnecessarily.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)