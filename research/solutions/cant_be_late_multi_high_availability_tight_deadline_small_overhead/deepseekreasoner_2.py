import json
from argparse import Namespace
from typing import List, Dict, Tuple
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"

    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.spot_availability_history = {}
        self.region_stats = {}
        self.current_step = 0
        self.remaining_work_history = []
        self.last_action = None
        self.consecutive_spot_failures = 0
        self.max_consecutive_spot_failures = 3

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
        
        # Initialize region stats
        for i in range(self.env.get_num_regions()):
            self.region_stats[i] = {
                'spot_available_count': 0,
                'total_steps': 0,
                'last_spot_available': True
            }
        
        return self

    def _update_spot_history(self, region_idx: int, has_spot: bool):
        if region_idx not in self.spot_availability_history:
            self.spot_availability_history[region_idx] = []
        
        # Keep only recent history to save memory
        if len(self.spot_availability_history[region_idx]) > 1000:
            self.spot_availability_history[region_idx].pop(0)
        
        self.spot_availability_history[region_idx].append(has_spot)
        
        # Update region stats
        self.region_stats[region_idx]['total_steps'] += 1
        if has_spot:
            self.region_stats[region_idx]['spot_available_count'] += 1
            self.consecutive_spot_failures = 0
        else:
            self.consecutive_spot_failures += 1
        self.region_stats[region_idx]['last_spot_available'] = has_spot

    def _get_region_spot_reliability(self, region_idx: int) -> float:
        if region_idx not in self.region_stats:
            return 0.0
        stats = self.region_stats[region_idx]
        if stats['total_steps'] == 0:
            return 0.0
        return stats['spot_available_count'] / stats['total_steps']

    def _get_best_spot_region(self) -> int:
        best_region = self.env.get_current_region()
        best_reliability = self._get_region_spot_reliability(best_region)
        
        for region in range(self.env.get_num_regions()):
            reliability = self._get_region_spot_reliability(region)
            if reliability > best_reliability:
                best_reliability = reliability
                best_region = region
        
        return best_region

    def _calculate_urgency(self) -> float:
        total_work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - total_work_done
        remaining_time = self.deadline - self.env.elapsed_seconds
        
        if remaining_time <= 0:
            return float('inf')
        
        # Factor in restart overhead
        if self.remaining_restart_overhead > 0:
            remaining_time -= self.remaining_restart_overhead
        
        if remaining_time <= 0:
            return float('inf')
        
        # How many time steps we have (conservative estimate)
        time_steps_available = remaining_time / self.env.gap_seconds
        
        # Minimum steps needed if we work every step
        min_steps_needed = remaining_work / self.env.gap_seconds
        
        if time_steps_available <= 0:
            return float('inf')
        
        urgency = min_steps_needed / time_steps_available
        return urgency

    def _should_use_ondemand(self, urgency: float, has_spot: bool) -> bool:
        # Critical condition: if we might miss deadline
        if urgency > 0.9:
            return True
        
        # If spot has been failing consecutively
        if self.consecutive_spot_failures >= self.max_consecutive_spot_failures:
            return True
        
        # If we're close to deadline and spot is unreliable
        if urgency > 0.7 and not has_spot:
            return True
        
        # Conservative approach near deadline
        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_time < 2 * self.restart_overhead:
            return True
        
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.current_step += 1
        current_region = self.env.get_current_region()
        
        # Update spot availability history
        self._update_spot_history(current_region, has_spot)
        
        # Calculate urgency
        urgency = self._calculate_urgency()
        
        # Check if we need to switch region for better spot availability
        if has_spot:
            self.last_action = 'spot_available'
        else:
            best_region = self._get_best_spot_region()
            if best_region != current_region and urgency < 0.8:
                self.env.switch_region(best_region)
                # After switching, we need to re-evaluate spot availability
                # We'll return NONE for this step since we just switched
                return ClusterType.NONE
        
        # Check if we should use on-demand
        if self._should_use_ondemand(urgency, has_spot):
            return ClusterType.ON_DEMAND
        
        # Use spot if available
        if has_spot:
            return ClusterType.SPOT
        
        # If spot not available and we're not urgent, wait
        if urgency < 0.6:
            return ClusterType.NONE
        
        # Otherwise use on-demand
        return ClusterType.ON_DEMAND