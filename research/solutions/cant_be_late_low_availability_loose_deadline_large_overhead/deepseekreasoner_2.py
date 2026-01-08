import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_threshold"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Constants
        OD_PRICE = 3.06  # $/hr
        SPOT_PRICE = 0.97  # $/hr
        OD_COST_PER_STEP = OD_PRICE * self.env.gap_seconds / 3600
        SPOT_COST_PER_STEP = SPOT_PRICE * self.env.gap_seconds / 3600
        
        # Calculate remaining work
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If no work left, pause
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Calculate minimum steps needed
        min_steps_needed = remaining_work / self.env.gap_seconds
        
        # Calculate conservative estimate of steps needed with potential restarts
        # Assuming worst-case restart scenario
        safety_factor = 1.5
        effective_steps_needed = min_steps_needed * safety_factor
        
        # Time pressure calculation
        steps_can_afford = time_left / self.env.gap_seconds
        
        # Calculate cost-benefit threshold
        # When time is plentiful, prefer spot
        # When time is tight, switch to OD
        
        # Dynamic threshold based on time pressure
        time_pressure = max(0, 1 - (steps_can_afford / effective_steps_needed))
        
        # Cost saving potential
        cost_saving = OD_COST_PER_STEP - SPOT_COST_PER_STEP
        saving_ratio = cost_saving / OD_COST_PER_STEP if OD_COST_PER_STEP > 0 else 0
        
        # Combined decision factor
        # Higher threshold means more likely to use spot
        spot_threshold = 0.3 + 0.5 * (1 - time_pressure) * saving_ratio
        
        # Check if we should use on-demand due to time pressure
        if time_pressure > 0.7:  # High time pressure
            return ClusterType.ON_DEMAND
        
        # Check if we're in restart overhead
        # If we were using spot and now it's unavailable, we might be in overhead
        # Use simple heuristic: avoid frequent switches
        if last_cluster_type == ClusterType.NONE:
            # Just coming out of pause, prefer spot if available
            if has_spot and np.random.random() < spot_threshold:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        # Main decision logic
        if has_spot:
            # Use spot with probability based on threshold
            if np.random.random() < spot_threshold:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            # Spot not available
            if time_pressure > 0.4:  # Moderate time pressure
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)