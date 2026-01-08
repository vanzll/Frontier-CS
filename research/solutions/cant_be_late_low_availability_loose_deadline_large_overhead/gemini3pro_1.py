from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the cluster type for the next time step.
        Implements a slack-based strategy:
        1. Calculates remaining work and time.
        2. Enters 'Panic Mode' (forces On-Demand) if remaining time is close to required time + overhead.
        3. If safe, utilizes Spot instances if available.
        4. If safe but no Spot, waits (NONE) to save money.
        """
        
        # 1. Calculate remaining work
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_remaining = self.task_duration - work_done

        # If job is effectively done, do nothing
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        # 2. Calculate time constraints
        current_time = self.env.elapsed_seconds
        time_remaining = self.deadline - current_time
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Safety buffer definition:
        # We need enough buffer to cover:
        # - The restart overhead (if we need to switch suddenly)
        # - Discretization steps (gap_seconds)
        # - A safety margin for floating point or boundary issues
        # 3.0 * overhead gives ~36 minutes buffer (since overhead is 12 mins), which is safe but not wasteful given 22h slack.
        safety_buffer = (3.0 * overhead) + (2.0 * gap)

        # 3. Critical Path Analysis (Must use On-Demand?)
        # Calculate time needed to finish using On-Demand.
        # If we are not currently running On-Demand, we must account for the startup overhead.
        od_overhead_penalty = overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0
        time_needed_od = work_remaining + od_overhead_penalty

        # If time is running out, force On-Demand to guarantee completion.
        if time_remaining < time_needed_od + safety_buffer:
            return ClusterType.ON_DEMAND

        # 4. Cost Optimization Strategy (Not Critical)
        if has_spot:
            # Spot is available. 
            # If we are currently ON_DEMAND, switching to SPOT incurs an overhead.
            # We must check if paying that overhead is safe.
            spot_overhead_penalty = overhead if last_cluster_type != ClusterType.SPOT else 0.0
            time_needed_spot = work_remaining + spot_overhead_penalty
            
            # If switching to/starting Spot leaves us enough buffer, do it.
            if time_remaining >= time_needed_spot + safety_buffer:
                return ClusterType.SPOT
            else:
                # We have Spot, but switching is too risky for the deadline. 
                # Since we are not in critical panic yet (checked above), this usually implies 
                # we are currently ON_DEMAND and switching would burn our safety margin.
                return ClusterType.ON_DEMAND
        
        # 5. Waiting Strategy
        # Not critical, no Spot available -> Wait (NONE) to minimize cost
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)