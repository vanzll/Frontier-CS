import json
from argparse import Namespace
from typing import List
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_multi_region"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Additional initialization
        self.spot_history = {}
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.ondemand_price / self.spot_price
        self.safety_margin = 1.2  # 20% safety margin
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get current state
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - elapsed
        
        # Update spot availability history
        if current_region not in self.spot_history:
            self.spot_history[current_region] = []
        self.spot_history[current_region].append(has_spot)
        
        # Calculate minimum time needed with different strategies
        gap = self.env.gap_seconds
        
        # Calculate critical threshold - when we must use on-demand
        # Account for potential restart overhead
        min_time_needed = work_remaining
        if last_cluster_type != ClusterType.ON_DEMAND and last_cluster_type != ClusterType.NONE:
            min_time_needed += self.restart_overhead
        
        # Check if we're in critical state
        is_critical = time_remaining < min_time_needed * self.safety_margin
        
        # Check if spot is available in current region
        spot_available = has_spot
        
        # Calculate expected spot availability in current region
        spot_reliability = 0.0
        if current_region in self.spot_history and len(self.spot_history[current_region]) > 0:
            reliable_period = min(10, len(self.spot_history[current_region]))
            recent_history = self.spot_history[current_region][-reliable_period:]
            if recent_history:
                spot_reliability = sum(recent_history) / len(recent_history)
        
        # Calculate when we would finish if using spot vs on-demand
        steps_needed_spot = math.ceil(work_remaining / gap)
        steps_needed_ondemand = math.ceil(work_remaining / gap)
        
        # Estimate potential interruptions for spot
        if spot_reliability > 0:
            expected_interruptions = steps_needed_spot * (1 - spot_reliability)
            time_with_spot = steps_needed_spot * gap + expected_interruptions * self.restart_overhead
        else:
            time_with_spot = float('inf')
        
        time_with_ondemand = steps_needed_ondemand * gap
        
        # Check other regions for better spot availability
        best_region = current_region
        best_reliability = spot_reliability
        
        for region in range(num_regions):
            if region == current_region:
                continue
                
            if region in self.spot_history and len(self.spot_history[region]) > 0:
                region_history = self.spot_history[region][-5:]  # Last 5 steps
                if region_history:
                    region_reliability = sum(region_history) / len(region_history)
                    if region_reliability > best_reliability:
                        best_reliability = region_reliability
                        best_region = region
        
        # Decide whether to switch region
        should_switch = (
            best_region != current_region and 
            best_reliability > spot_reliability + 0.2 and  # Significant improvement
            not is_critical and
            time_remaining > work_remaining + self.restart_overhead * 2
        )
        
        if should_switch:
            self.env.switch_region(best_region)
            # After switching, we'll have restart overhead anyway
            # Use spot if available in new region (we'll check in next step)
            return ClusterType.NONE
        
        # Decision logic
        if is_critical:
            # In critical state, use on-demand to guarantee completion
            return ClusterType.ON_DEMAND
        
        if not spot_available:
            # If spot not available, check if we should wait or use on-demand
            # Calculate if waiting makes sense
            steps_can_wait = int(time_remaining / gap) - steps_needed_ondemand
            if steps_can_wait > 2 and time_remaining > work_remaining + gap * 2:
                # We can afford to wait a bit for spot
                return ClusterType.NONE
            else:
                # Need to start making progress
                return ClusterType.ON_DEMAND
        
        # Spot is available
        # Check if spot is reliable enough given time constraints
        expected_spot_time = time_with_spot if spot_reliability > 0 else float('inf')
        
        if expected_spot_time <= time_remaining and spot_reliability > 0.7:
            # Good spot reliability and enough time
            return ClusterType.SPOT
        elif expected_spot_time <= time_remaining * 1.1 and spot_reliability > 0.5:
            # Moderate reliability with some buffer
            return ClusterType.SPOT
        elif time_remaining > work_remaining * 2:
            # Plenty of time, try spot
            return ClusterType.SPOT
        else:
            # Conservative: use on-demand
            return ClusterType.ON_DEMAND