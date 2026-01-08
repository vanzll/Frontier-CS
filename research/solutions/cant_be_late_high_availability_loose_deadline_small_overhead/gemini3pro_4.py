from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "RobustSlackStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # Calculate remaining work based on completed segments
        completed_work = sum(self.task_done_time)
        remaining_work = self.task_duration - completed_work

        # If task is effectively done, do nothing
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        # Get environment step size (gap) for safety margins
        gap = getattr(self.env, 'gap_seconds', 60.0)
        
        # Safety margin: 30 minutes (1800s) + step gap.
        # This buffer ensures we don't cut it too close to the deadline.
        margin = 1800.0 + gap 
        
        # The base time required to finish the work if we ran uninterrupted
        # We add margin to define a "panic threshold"
        base_threshold = remaining_work + margin

        # Logic:
        # 1. If we are running out of slack (remaining_time < needed), force ON_DEMAND.
        # 2. If we have slack, prefer SPOT to save money.
        # 3. If SPOT is unavailable and we have slack, wait (NONE).
        
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Hysteresis Logic:
            # If we are already paying for On-Demand, be reluctant to switch off.
            # Switching off risks incurring new restart overheads later if Spot fails.
            # We only leave OD if slack is very high (Margin + 1 hour + 2*Overhead).
            hysteresis = 3600.0 + (2.0 * self.restart_overhead)
            
            if remaining_time > base_threshold + hysteresis:
                # Safe to try Spot or Wait
                if has_spot:
                    return ClusterType.SPOT
                return ClusterType.NONE
            else:
                # Keep using safe On-Demand
                return ClusterType.ON_DEMAND
        else:
            # Currently on SPOT or NONE
            # If we switch to OD now, we incur one restart overhead.
            switch_overhead = self.restart_overhead
            
            # Panic check: Do we have enough time to finish using OD if we start now?
            if remaining_time < base_threshold + switch_overhead:
                return ClusterType.ON_DEMAND
            
            # Not in panic mode
            if has_spot:
                return ClusterType.SPOT
            
            # Spot unavailable, but plenty of slack -> Wait to save money
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)