import json
from argparse import Namespace
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
import math


class Solution(MultiRegionStrategy):
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
        
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.ondemand_price / self.spot_price
        
        self.spot_history = []
        self.region_history = []
        self.consecutive_spot_failures = 0
        self.max_consecutive_spot_failures = 3
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        gap = self.env.gap_seconds
        overhead = self.restart_overhead
        
        completed = sum(self.task_done_time)
        remaining_work = self.task_duration - completed
        time_left = deadline - elapsed
        
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        self.spot_history.append(has_spot)
        self.region_history.append(current_region)
        if len(self.spot_history) > 10:
            self.spot_history.pop(0)
            self.region_history.pop(0)
        
        if has_spot:
            self.consecutive_spot_failures = 0
        else:
            self.consecutive_spot_failures += 1
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        if time_left <= 0:
            return ClusterType.ON_DEMAND
        
        effective_time_left = time_left - self.remaining_restart_overhead
        
        critical_threshold = 0.15
        is_critical = (remaining_work / effective_time_left) > (1.0 / gap) * 1.2
        
        if is_critical or (effective_time_left - remaining_work) < overhead * 2:
            if has_spot and self.consecutive_spot_failures < self.max_consecutive_spot_failures:
                recent_spot_success = any(self.spot_history[-3:])
                if recent_spot_success:
                    return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                spot_available_ratio = sum(self.spot_history) / len(self.spot_history) if self.spot_history else 0
                
                if spot_available_ratio > 0.7:
                    return ClusterType.SPOT
                
                remaining_gaps = int(time_left / gap)
                expected_work_with_spot = remaining_gaps * gap * spot_available_ratio
                
                if expected_work_with_spot >= remaining_work:
                    return ClusterType.SPOT
                
                safety_margin = overhead * 2
                if (remaining_work / spot_available_ratio + overhead * 3) < effective_time_left - safety_margin:
                    return ClusterType.SPOT
            
            return ClusterType.SPOT
        
        if last_cluster_type == ClusterType.NONE:
            recent_spot_availability = self._check_other_regions_availability(current_region, num_regions)
            if recent_spot_availability >= 0:
                self.env.switch_region(recent_spot_availability)
                return ClusterType.NONE
        
        if self.consecutive_spot_failures >= self.max_consecutive_spot_failures:
            recent_spot_availability = self._check_other_regions_availability(current_region, num_regions)
            if recent_spot_availability >= 0:
                self.env.switch_region(recent_spot_availability)
                self.consecutive_spot_failures = 0
                return ClusterType.NONE
        
        remaining_gaps = int(time_left / gap)
        needed_gaps = math.ceil(remaining_work / gap)
        
        if needed_gaps + 2 >= remaining_gaps:
            return ClusterType.ON_DEMAND
        
        return ClusterType.NONE
    
    def _check_other_regions_availability(self, current_region: int, num_regions: int) -> int:
        best_region = -1
        max_recent_spot = -1
        
        for i in range(num_regions):
            if i == current_region:
                continue
            
            recent_in_region = 0
            total_in_region = 0
            
            for idx, region in enumerate(self.region_history):
                if region == i:
                    total_in_region += 1
                    if self.spot_history[idx]:
                        recent_in_region += 1
            
            if total_in_region > 0:
                ratio = recent_in_region / total_in_region
                if ratio > max_recent_spot:
                    max_recent_spot = ratio
                    best_region = i
        
        if max_recent_spot > 0.5:
            return best_region
        
        return -1