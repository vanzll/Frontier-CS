from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "slack_hysteresis_solver"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate progress and remaining work
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        work_done = sum(self.task_done_time)
        work_total = self.task_duration
        work_rem = work_total - work_done
        
        # If work is completed, stop
        if work_rem <= 1e-6:
            return ClusterType.NONE
            
        time_rem = deadline - elapsed
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Safety buffer to handle discrete time steps and potential delays
        # We need to ensure we detect the panic condition at least 'gap' seconds before it's too late
        safety_buffer = 2.0 * gap + 1.0  # +1s for epsilon safety
        
        # Panic Threshold:
        # Minimum time required to finish on On-Demand assuming we switch now.
        # We need: time_rem >= work_rem + overhead (for switch)
        # If we are close to this limit, we must switch to OD immediately.
        panic_threshold = work_rem + overhead + safety_buffer
        
        # Hysteresis Threshold:
        # If currently on On-Demand, only switch back to Spot if we have excess slack.
        # We must pay 'overhead' to switch, and remaining slack must still be safe (above panic_threshold).
        # Condition: time_rem - overhead > panic_threshold
        switch_to_spot_threshold = panic_threshold + overhead
        
        # Decision Logic
        if last_cluster_type == ClusterType.ON_DEMAND:
            # If we are already paying for OD, stay on it unless Spot is available 
            # and we have enough slack to justify the switch overhead + risk.
            if has_spot and time_rem > switch_to_spot_threshold:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
            
        else:
            # Currently on Spot or None
            # If we are running out of time (slack is exhausted), force OD
            if time_rem < panic_threshold:
                return ClusterType.ON_DEMAND
            
            # If we have slack, maximize cost savings
            if has_spot:
                return ClusterType.SPOT
            else:
                # Spot unavailable, but we have slack: Wait (spend time to save money)
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)