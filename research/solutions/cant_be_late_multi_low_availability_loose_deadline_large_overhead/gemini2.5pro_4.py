import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that uses heuristics to balance cost and deadline satisfaction.

    The strategy is based on the following principles:
    1.  **Urgency-Based Fallback**: It calculates a "critical time" by which it must switch to
        reliable On-Demand instances to guarantee finishing before the deadline. If time runs
        short, it defaults to On-Demand.
    2.  **Adaptive Learning**: It maintains an Exponential Moving Average (EMA) of spot availability
        for each region to adapt to changing conditions and learn which regions are more reliable.
    3.  **Conservative Switching**: It only considers switching regions when the current region's
        spot instance is unavailable, and another region is estimated to be significantly better.
        This minimizes unnecessary restart overheads.
    4.  **Slack Management**: When spot is unavailable but there is ample time before the deadline,
        it will choose to wait (NONE) to save costs, rather than immediately using an expensive
        On-Demand instance.
    """

    NAME = "EMA_Balancer"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self.num_regions = self.env.get_num_regions()
        
        # Initialize with an optimistic belief about spot availability to encourage exploration.
        self.spot_ema = [0.75] * self.num_regions
        
        # --- Hyperparameters ---
        self.ema_alpha = 0.1
        self.switch_hysteresis = 0.15
        self.on_demand_safety_buffer_factor = 1.5
        self.wait_slack_threshold_factor = 1.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide the next action based on the current state.
        """
        # 1. Update belief about current region's spot availability using EMA.
        current_region = self.env.get_current_region()
        current_spot_observation = 1.0 if has_spot else 0.0
        self.spot_ema[current_region] = (
            self.ema_alpha * current_spot_observation +
            (1 - self.ema_alpha) * self.spot_ema[current_region]
        )

        # 2. Calculate remaining work and time criticality.
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        gap_seconds = self.env.gap_seconds
        
        # Calculate time needed to finish using only On-Demand from now.
        steps_needed_od = math.ceil(work_remaining / gap_seconds)
        time_needed_od = steps_needed_od * gap_seconds
        
        # The absolute latest time we can start using OD and still finish.
        on_demand_must_start_time = self.deadline - time_needed_od
        
        # Add a safety buffer to account for potential future spot preemptions.
        preemption_time_cost = gap_seconds + self.restart_overhead
        safety_buffer = self.on_demand_safety_buffer_factor * preemption_time_cost
        on_demand_trigger_time = on_demand_must_start_time - safety_buffer

        # Effective current time includes any pending restart overhead.
        current_effective_time = self.env.elapsed_seconds + self.remaining_restart_overhead

        # 3. Urgency Check: If past the trigger time, must use On-Demand.
        if current_effective_time >= on_demand_trigger_time:
            return ClusterType.ON_DEMAND

        # 4. Slack available. Decide based on spot availability.
        if has_spot:
            # Spot is available and we have slack, so use it.
            return ClusterType.SPOT
        else:
            # Spot is not available. Choose between switching, using On-Demand, or waiting.
            
            # 4a. Consider switching to a better region.
            if self.num_regions > 1:
                max_ema = max(self.spot_ema)
                best_region_idx = self.spot_ema.index(max_ema)
                
                if (best_region_idx != current_region and 
                    max_ema > self.spot_ema[current_region] + self.switch_hysteresis):
                    
                    self.env.switch_region(best_region_idx)
                    # After switching, we cannot return SPOT. Use ON_DEMAND to make progress.
                    return ClusterType.ON_DEMAND

            # 4b. No better region or single region. Decide between waiting (NONE) and On-Demand.
            slack = on_demand_trigger_time - current_effective_time
            
            if slack > self.wait_slack_threshold_factor * gap_seconds:
                # Enough slack to wait for spot to become available again.
                return ClusterType.NONE
            else:
                # Slack is low; better to make progress with On-Demand.
                return ClusterType.ON_DEMAND