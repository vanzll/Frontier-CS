from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # 1. Gather current state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        restart_overhead = self.restart_overhead
        
        # Calculate work progress
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If task is finished, stop incurring costs
        if work_remaining <= 0:
            return ClusterType.NONE
            
        time_remaining = deadline - elapsed
        
        # 2. Safety Logic (Panic Mode)
        # Determine the latest possible moment we must switch to On-Demand (OD) to guarantee completion.
        # If we switch to OD now, do we incur a restart overhead?
        # - If already on OD: No overhead.
        # - If on Spot or None: Yes, overhead applies.
        
        overhead_penalty = 0.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            overhead_penalty = restart_overhead
            
        # Total time required if we commit to OD immediately
        time_needed_for_od = work_remaining + overhead_penalty
        
        # Safety Buffer calculation:
        # We operate in discrete time steps defined by 'gap'.
        # If we delay switching to OD by one step, we lose 'gap' time.
        # We must ensure T_remaining > T_needed at the NEXT step.
        # Buffer = 2.0 * gap provides safety against step granularity.
        # We also set a minimum constant buffer (300s) to handle edge cases or small variances.
        buffer = max(2.0 * gap, 300.0)
        
        # If remaining time is critically low, force On-Demand usage
        if time_remaining <= time_needed_for_od + buffer:
            return ClusterType.ON_DEMAND
            
        # 3. Cost Optimization
        # If we are not in panic mode, we have sufficient slack.
        # Prioritize Spot instances for cost efficiency.
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot unavailable, but we have slack.
            # Wait (NONE) to save money rather than paying for OD.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)