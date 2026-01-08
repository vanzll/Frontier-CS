import json
from argparse import Namespace
from typing import List
import numpy as np

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
        
        # Initialize tracking structures
        self.num_regions = self.env.get_num_regions()
        self.region_spot_history = [[] for _ in range(self.num_regions)]
        self.region_availability = [0.0] * self.num_regions
        self.last_region = self.env.get_current_region()
        self.switch_cooldown = 0
        self.consecutive_spot_failures = 0
        self.spot_success_streak = 0
        self.emergency_mode = False
        
        # Cost parameters (from problem description)
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.step_hours = self.env.gap_seconds / 3600.0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        
        # Update region spot history
        self.region_spot_history[current_region].append(1 if has_spot else 0)
        if len(self.region_spot_history[current_region]) > 24:  # Keep last 24 hours
            self.region_spot_history[current_region].pop(0)
        
        # Calculate region availability (moving average)
        if self.region_spot_history[current_region]:
            self.region_availability[current_region] = np.mean(self.region_spot_history[current_region])
        
        # Check if we switched regions
        if current_region != self.last_region:
            self.switch_cooldown = 3  # Steps to wait after switching
            self.last_region = current_region
        elif self.switch_cooldown > 0:
            self.switch_cooldown -= 1
        
        # Calculate progress metrics
        total_work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - total_work_done
        time_remaining = self.deadline - self.env.elapsed_seconds
        steps_remaining = time_remaining / self.env.gap_seconds if time_remaining > 0 else 0
        
        # Emergency check: if we're running out of time
        min_steps_needed = work_remaining / (self.env.gap_seconds - self.restart_overhead)
        if steps_remaining < min_steps_needed * 1.2:
            self.emergency_mode = True
        
        # Calculate conservative progress rate needed
        progress_needed = work_remaining / max(time_remaining, 1)
        step_capacity = self.env.gap_seconds - (self.remaining_restart_overhead if self.remaining_restart_overhead > 0 else 0)
        step_progress = min(step_capacity, self.env.gap_seconds)
        
        # Strategy decision logic
        if self.emergency_mode:
            # In emergency mode, use on-demand to ensure completion
            if has_spot and self.consecutive_spot_failures < 2:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Check if we should consider switching regions
        if (self.switch_cooldown == 0 and 
            not has_spot and 
            self.region_availability[current_region] < 0.4):
            
            best_region = -1
            best_availability = 0
            
            # Find region with best recent availability
            for i in range(self.num_regions):
                if i != current_region and self.region_availability[i] > best_availability:
                    best_availability = self.region_availability[i]
                    best_region = i
            
            if best_region != -1 and best_availability > self.region_availability[current_region] + 0.2:
                self.env.switch_region(best_region)
                # After switching, use NONE for this step to avoid immediate overhead
                return ClusterType.NONE
        
        # Main decision logic
        if has_spot:
            # Use spot if available and we have reasonable chance
            if (self.consecutive_spot_failures < 3 and 
                step_progress >= progress_needed * self.env.gap_seconds * 0.8):
                
                self.spot_success_streak += 1
                self.consecutive_spot_failures = 0
                return ClusterType.SPOT
            else:
                # Too risky, use on-demand
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            self.consecutive_spot_failures += 1
            self.spot_success_streak = 0
            
            # Decide between on-demand and pause
            if (step_progress >= progress_needed * self.env.gap_seconds or 
                self.consecutive_spot_failures >= 2 or
                time_remaining < work_remaining * 1.5):
                
                return ClusterType.ON_DEMAND
            else:
                # Can afford to wait
                return ClusterType.NONE