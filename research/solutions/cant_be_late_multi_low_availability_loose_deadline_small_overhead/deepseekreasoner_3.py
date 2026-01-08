import json
from argparse import Namespace
from typing import List
import heapq

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""
    NAME = "my_strategy"

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
        
        # Initialize tracking structures
        self.region_count = self.env.get_num_regions()
        self.work_done = 0.0
        self.last_work_update = 0.0
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.last_action = None
        
        # Region quality tracking
        self.region_spot_availability = [0] * self.region_count
        self.region_spot_attempts = [0] * self.region_count
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update work done
        current_work = sum(self.task_done_time)
        if current_work > self.work_done:
            self.work_done = current_work
            if last_cluster_type == ClusterType.SPOT:
                # Update spot availability for current region
                current_region = self.env.get_current_region()
                self.region_spot_availability[current_region] += 1
        
        # Update attempts for current region if we tried spot
        if last_cluster_type == ClusterType.SPOT:
            current_region = self.env.get_current_region()
            self.region_spot_attempts[current_region] += 1
        
        # Calculate remaining work and time
        remaining_work = self.task_duration - self.work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If no work left, return NONE
        if remaining_work <= 1e-9:
            return ClusterType.NONE
        
        # If we have pending restart overhead, wait
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
        
        # Calculate time needed for on-demand (guaranteed)
        time_needed_ondemand = remaining_work
        
        # Calculate time needed for spot (with overhead)
        # Conservative estimate: assume we'll get interrupted
        time_needed_spot = remaining_work + self.restart_overhead
        
        # Calculate slack ratio
        slack_ratio = time_left / remaining_work if remaining_work > 0 else float('inf')
        
        # Emergency mode: if very little time left, use on-demand
        if time_left < time_needed_ondemand * 1.2:
            return ClusterType.ON_DEMAND
        
        # If good spot availability and reasonable slack, try spot
        current_region = self.env.get_current_region()
        region_success_rate = 0
        if self.region_spot_attempts[current_region] > 0:
            region_success_rate = (
                self.region_spot_availability[current_region] / 
                self.region_spot_attempts[current_region]
            )
        
        # Determine best region
        best_region = current_region
        best_score = -1
        
        for region in range(self.region_count):
            attempts = self.region_spot_attempts[region]
            if attempts > 0:
                success_rate = self.region_spot_availability[region] / attempts
                score = success_rate
                if score > best_score:
                    best_score = score
                    best_region = region
        
        # Switch to better region if significantly better
        if (best_region != current_region and 
            best_score > max(0.1, region_success_rate + 0.2) and
            slack_ratio > 1.5):
            self.env.switch_region(best_region)
            # After switching, check spot availability in new region
            # (has_spot parameter is for the region we're switching FROM,
            # so we'll return NONE and let next step decide)
            return ClusterType.NONE
        
        # Decision logic
        if has_spot:
            # Use spot if we have good slack or good success rate in this region
            if slack_ratio > 1.8 or (region_success_rate > 0.7 and slack_ratio > 1.3):
                return ClusterType.SPOT
            elif slack_ratio > 1.3:
                # Moderate slack - mixed strategy
                # Use spot 70% of the time in this case
                import random
                if random.random() < 0.7:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                # Low slack, use on-demand
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            if slack_ratio > 2.0:
                # Plenty of time, wait for spot
                return ClusterType.NONE
            elif slack_ratio > 1.3:
                # Try another region
                next_region = (current_region + 1) % self.region_count
                self.env.switch_region(next_region)
                return ClusterType.NONE
            else:
                # Running out of time, use on-demand
                return ClusterType.ON_DEMAND