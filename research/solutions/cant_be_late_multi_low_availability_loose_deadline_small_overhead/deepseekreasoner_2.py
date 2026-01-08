import json
from argparse import Namespace
import math
from typing import List, Tuple

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
        self.spot_price = 0.9701  # $/hr
        self.on_demand_price = 3.06  # $/hr
        self.spot_safety_margin = 2  # hours
        self.min_spot_confidence = 0.3
        self.region_stats = {}
        self.current_region_loyalty = 0
        self.region_switch_threshold = 3
        self.consecutive_failures = 0
        self.max_consecutive_failures = 2
        self.last_action = ClusterType.NONE
        self.work_done = 0.0
        self.spot_availability_history = {}
        
        return self

    def _update_stats(self, region_idx: int, has_spot: bool):
        """Update statistics for region tracking."""
        if region_idx not in self.region_stats:
            self.region_stats[region_idx] = {
                'spot_count': 0,
                'total_steps': 0,
                'recent_spots': []
            }
        
        stats = self.region_stats[region_idx]
        stats['total_steps'] += 1
        if has_spot:
            stats['spot_count'] += 1
            stats['recent_spots'].append(1)
        else:
            stats['recent_spots'].append(0)
        
        # Keep only last 20 steps for recent history
        if len(stats['recent_spots']) > 20:
            stats['recent_spots'].pop(0)

    def _get_region_spot_probability(self, region_idx: int) -> float:
        """Get spot probability for a region based on history."""
        if region_idx not in self.region_stats:
            return 0.0
        
        stats = self.region_stats[region_idx]
        if stats['total_steps'] == 0:
            return 0.0
        
        # Weight recent history more heavily
        if stats['recent_spots']:
            recent_prob = sum(stats['recent_spots']) / len(stats['recent_spots'])
            overall_prob = stats['spot_count'] / stats['total_steps']
            return 0.7 * recent_prob + 0.3 * overall_prob
        else:
            return stats['spot_count'] / stats['total_steps']

    def _should_switch_region(self, current_region: int, has_spot: bool) -> Tuple[bool, int]:
        """Determine if we should switch regions and to which one."""
        if not has_spot and self.current_region_loyalty >= self.region_switch_threshold:
            # Try to find a better region
            best_region = current_region
            best_prob = self._get_region_spot_probability(current_region)
            
            for region_idx in range(self.env.get_num_regions()):
                if region_idx == current_region:
                    continue
                
                region_prob = self._get_region_spot_probability(region_idx)
                if region_prob > best_prob + 0.1:  # Significant improvement threshold
                    best_region = region_idx
                    best_prob = region_prob
            
            if best_region != current_region and best_prob > self.min_spot_confidence:
                return True, best_region
        
        return False, current_region

    def _calculate_safety_margin(self) -> float:
        """Calculate safety margin based on remaining time and work."""
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        if time_left <= 0 or remaining_work <= 0:
            return 0.0
        
        # Convert to hours for easier reasoning
        remaining_work_hours = remaining_work / 3600.0
        time_left_hours = time_left / 3600.0
        
        # Base safety margin scales with remaining work and time pressure
        base_margin = min(4.0, max(1.0, remaining_work_hours * 0.2))
        
        # Adjust based on time pressure
        time_ratio = time_left_hours / remaining_work_hours
        if time_ratio < 1.5:
            # Very tight schedule
            return 0.5
        elif time_ratio < 2.0:
            # Tight schedule
            return base_margin * 0.5
        else:
            # Comfortable schedule
            return base_margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update current region stats
        current_region = self.env.get_current_region()
        self._update_stats(current_region, has_spot)
        
        # Update work progress
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        # If no work left or past deadline, pause
        if remaining_work <= 0 or time_left <= 0:
            self.current_region_loyalty = max(0, self.current_region_loyalty - 1)
            self.last_action = ClusterType.NONE
            return ClusterType.NONE
        
        # Check if we need to be conservative due to approaching deadline
        safety_margin_hours = self._calculate_safety_margin()
        safety_margin_seconds = safety_margin_hours * 3600.0
        
        # Calculate minimum time needed with on-demand (no overhead after restart)
        min_time_needed = remaining_work + self.remaining_restart_overhead
        
        # If we're in a critical situation, use on-demand
        if time_left - min_time_needed < safety_margin_seconds:
            self.consecutive_failures = 0
            self.current_region_loyalty += 1
            self.last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        # Check if we should switch region
        should_switch, target_region = self._should_switch_region(current_region, has_spot)
        if should_switch and target_region != current_region:
            self.env.switch_region(target_region)
            self.current_region_loyalty = 0
            self.consecutive_failures = 0
            # After switching, we need to restart anyway, so use spot if available
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
        
        # If we have spot available and not in critical zone, use it
        if has_spot:
            # But if we've had too many consecutive failures recently, be cautious
            if self.consecutive_failures >= self.max_consecutive_failures:
                # Use on-demand for a while to build up progress
                if remaining_work > 4 * 3600:  # More than 4 hours left
                    self.consecutive_failures = max(0, self.consecutive_failures - 1)
                    self.current_region_loyalty += 1
                    self.last_action = ClusterType.ON_DEMAND
                    return ClusterType.ON_DEMAND
            
            self.consecutive_failures = 0
            self.current_region_loyalty += 1
            self.last_action = ClusterType.SPOT
            return ClusterType.SPOT
        else:
            # No spot available in current region
            self.consecutive_failures += 1
            self.current_region_loyalty = max(0, self.current_region_loyalty - 1)
            
            # If we're building loyalty to a region and spot isn't available,
            # pause briefly rather than immediately switching or using expensive on-demand
            if (self.current_region_loyalty > 0 and 
                time_left - min_time_needed > safety_margin_seconds * 2):
                self.last_action = ClusterType.NONE
                return ClusterType.NONE
            else:
                self.last_action = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND