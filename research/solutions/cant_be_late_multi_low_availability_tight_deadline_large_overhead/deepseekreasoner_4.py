import json
from argparse import Namespace
from enum import IntEnum
from typing import List, Tuple
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Action(IntEnum):
    NONE = 0
    SPOT = 1
    ON_DEMAND = 2
    SWITCH = 3


class Solution(MultiRegionStrategy):
    NAME = "cautious_spot_seeker"
    
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
        
        # Cache expensive calculations
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.price_ratio = self.spot_price / self.ondemand_price
        self.gap_seconds = 3600.0  # Default time step
        
        # Initialize region tracking
        self.region_spot_history = None
        self.region_switch_times = None
        self.current_region_idx = 0
        self.last_action = None
        self.switch_count = 0
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize tracking structures on first call
        if self.region_spot_history is None:
            num_regions = self.env.get_num_regions()
            self.region_spot_history = [0] * num_regions
            self.region_switch_times = [0] * num_regions
            self.current_region_idx = self.env.get_current_region()
            self.last_action = Action.NONE
        
        # Update current region
        self.current_region_idx = self.env.get_current_region()
        
        # Calculate remaining work and time
        total_done = sum(self.task_done_time)
        remaining_work = self.task_duration - total_done
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        
        # If work is done, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Calculate effective time per step (considering overhead)
        step_seconds = self.gap_seconds
        
        # Emergency mode: must use on-demand to finish
        safe_margin = self.restart_overhead * 2
        time_needed_if_ondemand = remaining_work + (self.restart_overhead if last_cluster_type != ClusterType.ON_DEMAND else 0)
        
        if remaining_time <= time_needed_if_ondemand + safe_margin:
            # Must use on-demand to guarantee completion
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        # Calculate maximum affordable spot attempts
        time_for_spot_attempts = remaining_time - time_needed_if_ondemand - safe_margin
        max_spot_attempts = int(time_for_spot_attempts / (step_seconds + self.restart_overhead))
        
        # Update spot history for current region
        if has_spot:
            self.region_spot_history[self.current_region_idx] += 1
        
        # Count steps in current region
        steps_in_current = self.region_switch_times[self.current_region_idx] + 1
        
        # Decision logic
        if remaining_work <= step_seconds:
            # Almost done, use on-demand to avoid restart
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        # Calculate risk factor based on remaining time
        time_ratio = remaining_time / self.deadline
        work_ratio = remaining_work / self.task_duration
        
        # Adjust strategy based on progress
        progress_ratio = total_done / self.task_duration
        
        if progress_ratio < 0.3:
            # Early phase: be more aggressive with spot
            spot_aggression = 0.7
        elif progress_ratio < 0.7:
            # Middle phase: moderate
            spot_aggression = 0.5
        else:
            # Late phase: more conservative
            spot_aggression = 0.3
        
        # Consider switching regions if spot unavailable or poor history
        if not has_spot or (steps_in_current > 10 and self.region_spot_history[self.current_region_idx] / steps_in_current < 0.3):
            num_regions = self.env.get_num_regions()
            if num_regions > 1:
                # Find region with best spot availability history
                best_region = self.current_region_idx
                best_ratio = -1
                
                for i in range(num_regions):
                    if i == self.current_region_idx:
                        continue
                    region_steps = max(1, self.region_switch_times[i])
                    spot_ratio = self.region_spot_history[i] / region_steps
                    if spot_ratio > best_ratio:
                        best_ratio = spot_ratio
                        best_region = i
                
                # Only switch if significantly better
                current_steps = max(1, steps_in_current)
                current_ratio = self.region_spot_history[self.current_region_idx] / current_steps
                
                if best_ratio > current_ratio + 0.1 and self.switch_count < 3:
                    self.env.switch_region(best_region)
                    self.region_switch_times[self.current_region_idx] = 0
                    self.current_region_idx = best_region
                    self.switch_count += 1
                    self.last_action = Action.SWITCH
                    return ClusterType.NONE
        
        # Main decision: spot vs on-demand
        if has_spot:
            # Use spot if we have enough time buffer
            time_buffer_needed = self.restart_overhead * 2 + step_seconds
            
            # Adjust probability based on progress and remaining time
            spot_probability = min(0.9, spot_aggression * (1 + time_ratio - work_ratio))
            
            # Use random-like deterministic decision based on elapsed time
            decision_factor = (elapsed / step_seconds) % 1.0
            
            if decision_factor < spot_probability and remaining_time > time_buffer_needed:
                self.last_action = Action.SPOT
                return ClusterType.SPOT
        
        # Default to on-demand
        self.last_action = Action.ON_DEMAND
        return ClusterType.ON_DEMAND