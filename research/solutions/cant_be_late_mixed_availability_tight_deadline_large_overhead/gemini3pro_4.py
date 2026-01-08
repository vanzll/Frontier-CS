from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Retrieve environment and state variables
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        deadline = self.deadline
        
        # Calculate progress
        work_done = sum(self.task_done_time)
        work_total = self.task_duration
        work_rem = max(0.0, work_total - work_done)
        
        # Calculate time metrics
        time_rem = deadline - elapsed
        slack = time_rem - work_rem
        
        # Define Panic Threshold
        # We must switch to On-Demand if slack drops too low.
        # We need to cover:
        # 1. Potential restart overhead (if we are stopped or switching types)
        # 2. Step gap (to ensure we don't miss the deadline between steps)
        # 3. Safety margin
        # 2.0 * overhead provides generous buffer (covers switch cost + buffer).
        # 2.0 * gap ensures simulation granularity doesn't cause issues.
        panic_threshold = 2.0 * overhead + 2.0 * gap
        
        # 1. Panic Mode: If slack is critical, force On-Demand to meet deadline
        if slack < panic_threshold:
            return ClusterType.ON_DEMAND
        
        # 2. Standard Operation
        if has_spot:
            # Spot is available.
            # If we are currently on On-Demand (e.g., coming out of a panic or spot was just lost),
            # we should only switch to Spot if we have enough slack to justify the switch overhead.
            # Switching incurs 'overhead' time loss. We want to ensure we don't dip into panic zone immediately.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > panic_threshold + 1.5 * overhead:
                    return ClusterType.SPOT
                else:
                    # Too risky to switch, stay on On-Demand
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT
            
        else:
            # Spot unavailable and we have slack (not in panic mode).
            # Wait for Spot to return to save money.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)