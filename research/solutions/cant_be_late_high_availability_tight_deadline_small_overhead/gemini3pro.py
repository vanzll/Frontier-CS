from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "CantBeLateStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate current progress
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If the task is already completed, do nothing
        if work_remaining <= 0:
            return ClusterType.NONE

        # Current state variables
        elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - elapsed
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Safety Threshold Calculation
        # We must switch to ON_DEMAND if the remaining time is close to the minimum required.
        # Minimum required = work_remaining + restart_overhead (time to switch).
        # We add a safety buffer to account for:
        # 1. Discrete time steps (gap_seconds)
        # 2. Potential delays or small variations
        # 3. High penalty for missing deadline (-100,000)
        # Buffer chosen: 10 minutes (600s) + 2 step intervals is conservative but efficient.
        safety_buffer = 600.0 + 2.0 * gap
        must_run_od_threshold = work_remaining + overhead + safety_buffer
        
        # 1. Panic Mode: If time is running out, force ON_DEMAND to guarantee completion.
        if time_remaining < must_run_od_threshold:
            return ClusterType.ON_DEMAND
        
        # 2. Normal Mode: We have sufficient slack.
        if has_spot:
            # Hysteresis Check:
            # If we are currently on ON_DEMAND, switching to SPOT incurs a restart overhead.
            # We only switch if we have significant excess slack to justify the time cost.
            # We require slack > threshold + 2 * overhead to switch back.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if time_remaining > must_run_od_threshold + 2.0 * overhead:
                    return ClusterType.SPOT
                else:
                    # Not enough excess slack to justify switching cost/risk
                    return ClusterType.ON_DEMAND
            
            # If not currently on OD (or satisfied hysteresis), use Spot
            return ClusterType.SPOT
        
        # 3. Wait Mode: Spot is unavailable, but we have slack.
        # Pause to save money (Cost=0) instead of burning expensive ON_DEMAND.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)