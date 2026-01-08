from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cant_be_late_optimizer"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        total_duration = self.task_duration
        overhead = self.restart_overhead
        
        # Calculate remaining work
        # task_done_time is a list of completed segment durations
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        work_rem = max(0.0, total_duration - work_done)
        
        # If finished, stop
        if work_rem <= 1e-6:
            return ClusterType.NONE

        time_rem = deadline - elapsed
        
        # Calculate "Bailout Slack"
        # Determine how much time margin we have if we were to commit to On-Demand immediately.
        # If currently on OD, we continue seamlessly.
        # If not (Spot or None), we incur restart overhead to switch/start OD.
        needed_for_od = work_rem
        if last_cluster_type != ClusterType.ON_DEMAND:
            needed_for_od += overhead
            
        slack = time_rem - needed_for_od
        
        # Define Safety Buffer
        # We need to maintain enough slack to handle:
        # 1. The current time step duration (gap), as we can't act until next step.
        # 2. The overhead itself (already in needed_for_od, but we add margin for robustness).
        # 3. A general safety margin to handle floating point issues or simulation granularity.
        gap = self.env.gap_seconds
        # Buffer: gap + overhead + 10 minutes margin
        safety_buffer = gap + overhead + 600.0
        
        # Decision Logic
        
        # 1. Panic Mode: If slack is critically low, force OD usage to ensure deadline is met.
        if slack < safety_buffer:
            return ClusterType.ON_DEMAND
            
        # 2. Opportunistic Mode: We have sufficient slack.
        if has_spot:
            # Spot is available.
            
            # If we are currently on OD, avoid thrashing (switching back and forth).
            # Only switch OD -> SPOT if we have significant excess slack.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Require an extra 20 minutes buffer to justify leaving the safety of OD
                hysteresis = 1200.0 
                if slack > safety_buffer + hysteresis:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # If we are on Spot or None, use Spot immediately.
                return ClusterType.SPOT
        else:
            # Spot is NOT available.
            # Since slack > safety_buffer, we are not in danger yet.
            # We choose to WAIT (NONE) rather than burn money on OD.
            # This consumes slack but saves significant cost.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)