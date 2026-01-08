import json
from argparse import Namespace
from typing import List
import math

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
        
        # Initialize strategy parameters
        self.od_ratio = 3.06 / 0.9701  # On-demand to spot price ratio
        self.switch_penalty = 0.20 * 3600  # Switching penalty in seconds
        self.required_work = float(config["duration"]) * 3600
        self.restart_overhead_sec = float(config["overhead"]) * 3600
        self.deadline_sec = float(config["deadline"]) * 3600
        
        # State tracking
        self.region_history = []
        self.spot_availability = []
        self.consecutive_failures = 0
        self.last_region = 0
        self.region_trials = {}
        self.patience_counter = 0
        self.fallback_triggered = False
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get current state
        current_region = self.env.get_current_region()
        elapsed = self.env.elapsed_seconds
        num_regions = self.env.get_num_regions()
        
        # Calculate remaining work and time
        completed_work = sum(self.task_done_time)
        remaining_work = self.required_work - completed_work
        remaining_time = self.deadline_sec - elapsed
        
        # Emergency mode: must finish
        if remaining_time <= remaining_work + self.restart_overhead_sec * 2:
            if last_cluster_type != ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND if has_spot else ClusterType.ON_DEMAND
        
        # Calculate safe threshold (25% of remaining time buffer)
        time_buffer = remaining_time - remaining_work
        safe_threshold = time_buffer * 0.25
        
        # If we're running out of buffer, be more aggressive
        if time_buffer < self.restart_overhead_sec * 3:
            # Use on-demand if we have less than 3 restart buffers
            if last_cluster_type != ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND if has_spot else ClusterType.ON_DEMAND
        
        # Track region performance
        if current_region not in self.region_trials:
            self.region_trials[current_region] = {'attempts': 0, 'successes': 0}
        
        # If spot is available and we have buffer, use it
        if has_spot and time_buffer > self.restart_overhead_sec:
            # Check if we should switch regions
            if last_cluster_type == ClusterType.SPOT:
                # Stay in current region if working well
                self.region_trials[current_region]['attempts'] += 1
                self.region_trials[current_region]['successes'] += 1
                self.consecutive_failures = 0
                return ClusterType.SPOT
            else:
                # Switching from something else to spot
                self.region_trials[current_region]['attempts'] += 1
                self.region_trials[current_region]['successes'] += 1
                self.consecutive_failures = 0
                return ClusterType.SPOT
        
        # No spot available in current region
        elif not has_spot:
            self.region_trials[current_region]['attempts'] += 1
            
            # If we've had too many failures in this region, switch
            if self.consecutive_failures > 2 or self.region_trials[current_region]['attempts'] > 5:
                # Find best alternative region
                best_region = current_region
                best_score = -1
                
                for region in range(num_regions):
                    if region == current_region:
                        continue
                    
                    attempts = self.region_trials.get(region, {'attempts': 0})['attempts']
                    successes = self.region_trials.get(region, {'successes': 0})['successes']
                    
                    if attempts == 0:
                        score = 1.0  # Untried regions get high priority
                    else:
                        score = successes / max(attempts, 1)
                    
                    if score > best_score:
                        best_score = score
                        best_region = region
                
                # Switch to best region
                if best_region != current_region:
                    self.env.switch_region(best_region)
                    self.consecutive_failures = 0
                    self.patience_counter = 0
                    # Don't run immediately after switch to avoid penalty
                    return ClusterType.NONE
            
            self.consecutive_failures += 1
            
            # Use on-demand if we can afford it and spot keeps failing
            if time_buffer > self.restart_overhead_sec * 2 and self.consecutive_failures > 1:
                return ClusterType.ON_DEMAND
            else:
                # Wait and try again
                self.patience_counter += 1
                if self.patience_counter > 3:
                    self.patience_counter = 0
                    return ClusterType.NONE
                else:
                    return ClusterType.NONE
        
        # Default: use on-demand if we have reasonable buffer
        if time_buffer > self.restart_overhead_sec * 1.5:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE