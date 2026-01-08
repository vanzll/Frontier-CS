import json
from argparse import Namespace
import heapq
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "efficient_mixed"

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
        
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.time_step = self.env.gap_seconds
        
        self.region_history = []
        self.last_decision = ClusterType.NONE
        self.consecutive_spots = 0
        self.urgency_threshold = 0.3
        
        return self

    def _get_remaining_work(self):
        """Calculate remaining work in seconds."""
        done = sum(self.task_done_time)
        return max(0.0, self.task_duration - done)

    def _get_time_until_deadline(self):
        """Calculate time until deadline in seconds."""
        return max(0.0, self.deadline - self.env.elapsed_seconds)

    def _get_completion_urgency(self):
        """Calculate urgency factor (0-1) for completion."""
        remaining_work = self._get_remaining_work()
        time_until_deadline = self._get_time_until_deadline()
        
        if time_until_deadline <= 0:
            return 1.0
        
        base_time_needed = remaining_work + (self.restart_overhead if self.remaining_restart_overhead > 0 else 0)
        return min(1.0, base_time_needed / max(time_until_deadline, 0.001))

    def _should_switch_to_ondemand(self, has_spot):
        """Determine if we should switch to on-demand based on urgency."""
        urgency = self._get_completion_urgency()
        remaining_work = self._get_remaining_work()
        time_until_deadline = self._get_time_until_deadline()
        
        if urgency > self.urgency_threshold:
            if has_spot and remaining_work < self.time_step * 2:
                return False
            return True
        
        if time_until_deadline < remaining_work + self.restart_overhead * 2:
            return True
            
        return False

    def _find_best_region(self, current_region, has_spot):
        """Find the best region to switch to based on recent history."""
        num_regions = self.env.get_num_regions()
        
        if num_regions <= 1:
            return current_region
        
        if has_spot and self.consecutive_spots >= 2:
            return current_region
        
        best_region = current_region
        best_score = -1
        
        for region in range(num_regions):
            if region == current_region:
                continue
                
            score = 0
            for hist in reversed(self.region_history[-5:]):
                if hist.get('region') == region and hist.get('had_spot'):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_region = region
        
        return best_region if best_score > 0 else current_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        remaining_work = self._get_remaining_work()
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        time_until_deadline = self._get_time_until_deadline()
        
        if time_until_deadline <= 0:
            return ClusterType.ON_DEMAND
        
        current_region = self.env.get_current_region()
        
        self.region_history.append({
            'region': current_region,
            'had_spot': has_spot,
            'cluster_type': last_cluster_type
        })
        
        if len(self.region_history) > 20:
            self.region_history.pop(0)
        
        if last_cluster_type == ClusterType.SPOT and has_spot:
            self.consecutive_spots += 1
        else:
            self.consecutive_spots = 0
        
        urgency = self._get_completion_urgency()
        
        if urgency > 0.8:
            return ClusterType.ON_DEMAND
        
        if self._should_switch_to_ondemand(has_spot):
            return ClusterType.ON_DEMAND
        
        if has_spot:
            if self.consecutive_spots >= 1 or urgency < 0.5:
                return ClusterType.SPOT
        
        best_region = self._find_best_region(current_region, has_spot)
        
        if best_region != current_region:
            self.env.switch_region(best_region)
            
            if has_spot and urgency < 0.6:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND if urgency > 0.4 else ClusterType.NONE
        
        if not has_spot and urgency < 0.4:
            return ClusterType.NONE
        
        if has_spot:
            return ClusterType.SPOT
        
        return ClusterType.ON_DEMAND if urgency > 0.3 else ClusterType.NONE