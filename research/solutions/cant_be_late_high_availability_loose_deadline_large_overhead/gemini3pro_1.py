from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "safe_cost_optimized_strategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # 1. Gather State
        current_time = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - work_done)
        time_remaining = self.deadline - current_time
        
        # 2. Define Panic Threshold
        # We must ensure we have enough time to finish the work using On-Demand.
        # We assume worst-case: we must incur restart_overhead to start On-Demand logic.
        # We add a buffer of 2 time steps to handle simulation quantization safely.
        buffer_time = 2.0 * self.env.gap_seconds
        panic_threshold = remaining_work + self.restart_overhead + buffer_time
        
        # 3. Apply Hysteresis for Stability
        # If we are currently on On-Demand, we should not switch back to Spot immediately
        # just because we are slightly above the panic threshold. We need enough slack
        # to justify the cost of switching (paying overhead) and risking Spot preemption.
        # Switching OD -> Spot costs 'restart_overhead' time.
        # So we require: time_remaining > panic_threshold + restart_overhead
        effective_threshold = panic_threshold
        if last_cluster_type == ClusterType.ON_DEMAND:
            effective_threshold += self.restart_overhead

        # 4. Make Decision
        # Priority 1: Meet the deadline (Safety)
        if time_remaining < effective_threshold:
            return ClusterType.ON_DEMAND
            
        # Priority 2: Minimize cost (Use Spot if available)
        if has_spot:
            return ClusterType.SPOT
            
        # Priority 3: Conserve slack (Wait for Spot)
        # If we are safe and Spot is unavailable, we wait (NONE) rather than burning money on OD.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)