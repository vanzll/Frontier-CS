import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

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
        
        # Initialize strategy state
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.time_step = 3600.0  # 1 hour in seconds
        
        # Store region statistics
        self.region_stats = {}
        self.current_region = 0
        self.consecutive_failures = 0
        self.last_success_type = None
        
        # Deadline buffer (12 hours slack as per problem)
        self.deadline_buffer = 12 * 3600
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # Calculate time pressure
        time_remaining = self.deadline - elapsed
        time_needed = work_remaining + (0 if work_remaining == 0 else self.restart_overhead)
        
        # If we're out of time, use on-demand to try to finish
        if time_remaining <= self.restart_overhead + self.time_step:
            if work_remaining > 0:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE
        
        # If work is done, do nothing
        if work_remaining <= 0:
            return ClusterType.NONE
        
        # Initialize region stats if needed
        if current_region not in self.region_stats:
            self.region_stats[current_region] = {
                'spot_attempts': 0,
                'spot_success': 0,
                'last_used': elapsed
            }
        
        # Update region stats
        if last_cluster_type == ClusterType.SPOT:
            self.region_stats[current_region]['spot_attempts'] += 1
            if has_spot:
                self.region_stats[current_region]['spot_success'] += 1
                self.consecutive_failures = 0
                self.last_success_type = ClusterType.SPOT
            else:
                self.consecutive_failures += 1
        
        # Calculate region reliability
        region_reliability = {}
        for region in range(self.env.get_num_regions()):
            if region in self.region_stats and self.region_stats[region]['spot_attempts'] > 0:
                reliability = self.region_stats[region]['spot_success'] / self.region_stats[region]['spot_attempts']
            else:
                reliability = 0.5  # Default assumption
            region_reliability[region] = reliability
        
        # Determine if we should switch regions
        should_switch = False
        best_region = current_region
        best_reliability = region_reliability[current_region]
        
        # Find best region based on reliability
        for region, reliability in region_reliability.items():
            if reliability > best_reliability + 0.1:  # Only switch for significant improvement
                best_reliability = reliability
                best_region = region
        
        # Also consider unexplored regions
        for region in range(self.env.get_num_regions()):
            if region not in self.region_stats:
                best_region = region
                should_switch = True
                break
        
        # Check if current region is performing poorly
        if (self.consecutive_failures >= 2 or 
            (region_reliability[current_region] < 0.3 and self.region_stats[current_region]['spot_attempts'] >= 3)):
            should_switch = True
        
        # Switch if needed and not in critical time
        if should_switch and best_region != current_region and time_remaining > time_needed * 2:
            self.env.switch_region(best_region)
            self.current_region = best_region
            self.consecutive_failures = 0
            
            # After switching, use on-demand for one step to be safe
            return ClusterType.ON_DEMAND
        
        # Calculate safe time for spot usage
        time_safety_margin = max(2 * self.restart_overhead, self.time_step * 2)
        can_use_spot = time_remaining > time_needed + time_safety_margin
        
        # Determine action based on conditions
        if has_spot and can_use_spot:
            # Use spot if available and we have time buffer
            return ClusterType.SPOT
        elif not has_spot and can_use_spot:
            # No spot available but have time - try another region or wait
            if self.consecutive_failures < 2:
                return ClusterType.NONE
            else:
                # Too many failures, use on-demand
                return ClusterType.ON_DEMAND
        else:
            # Time is tight, use on-demand to ensure completion
            return ClusterType.ON_DEMAND