import json
import math
from argparse import Namespace
from typing import List, Tuple
from enum import IntEnum
from collections import defaultdict

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_optimizer"

    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.gap_seconds = 3600.0
        self.regions = []
        self.region_data = []
        self.region_spot_history = defaultdict(list)
        self.region_switches = 0
        self.consecutive_spot_failures = 0
        self.last_action = None
        self.time_in_region = 0
        self.spot_success_streak = 0
        self.current_best_region = 0
        self.region_availability_score = {}
        self.emergency_mode = False
        self.spot_attempts_in_current_region = 0
        self.critical_time_threshold = 0

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
        
        self.critical_time_threshold = self.restart_overhead * 2
        self.regions = list(range(self.env.get_num_regions()))
        for region in self.regions:
            self.region_availability_score[region] = 0.0
            
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        if remaining_work <= 0:
            return ClusterType.NONE
            
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
            
        self.time_in_region += self.gap_seconds
        
        if remaining_time <= 0:
            return ClusterType.NONE
            
        if work_done == 0 and self.time_in_region < self.gap_seconds * 2:
            if has_spot:
                self.spot_attempts_in_current_region += 1
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        time_per_work_unit = remaining_work / remaining_time if remaining_time > 0 else float('inf')
        critical_level = 1.0 / (time_per_work_unit + 1e-6)
        
        if remaining_time < remaining_work + self.restart_overhead * 2:
            self.emergency_mode = True
            
        if self.emergency_mode:
            if remaining_time < remaining_work + self.restart_overhead:
                return ClusterType.ON_DEMAND
            elif has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = max(0, self.consecutive_spot_failures - 1)
            
        if last_cluster_type == ClusterType.SPOT and has_spot:
            self.spot_success_streak += 1
            self.region_availability_score[current_region] += 1.0
        else:
            self.spot_success_streak = max(0, self.spot_success_streak - 1)
            
        self.spot_attempts_in_current_region += 1
        self.region_spot_history[current_region].append(1 if has_spot else 0)
        if len(self.region_spot_history[current_region]) > 10:
            self.region_spot_history[current_region].pop(0)
            
        if self.consecutive_spot_failures >= 2 or self.spot_attempts_in_current_region >= 3:
            if self._should_switch_region(current_region, has_spot, remaining_work, remaining_time):
                self._perform_region_switch()
                self.spot_attempts_in_current_region = 0
                self.consecutive_spot_failures = 0
                return ClusterType.NONE
                
        if remaining_time < remaining_work * 1.5:
            if has_spot and self.spot_success_streak >= 1:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
                
        if remaining_time > remaining_work * 3 and has_spot:
            if self.consecutive_spot_failures <= 1:
                return ClusterType.SPOT
                
        if has_spot:
            recent_history = self.region_spot_history[current_region][-3:] if len(self.region_spot_history[current_region]) >= 3 else self.region_spot_history[current_region]
            if len(recent_history) == 0 or sum(recent_history) / max(len(recent_history), 1) >= 0.5:
                if self.consecutive_spot_failures <= 1:
                    return ClusterType.SPOT
                    
        return ClusterType.ON_DEMAND if remaining_time < remaining_work * 2 else ClusterType.NONE

    def _should_switch_region(self, current_region: int, has_spot: bool, remaining_work: float, remaining_time: float) -> bool:
        if self.env.get_num_regions() <= 1:
            return False
            
        if remaining_time < remaining_work + self.restart_overhead * 3:
            return False
            
        if self.time_in_region < self.gap_seconds * 2:
            return False
            
        best_score = self.region_availability_score[current_region]
        best_region = current_region
        
        for region in self.regions:
            if region == current_region:
                continue
                
            history = self.region_spot_history[region]
            if len(history) > 0:
                score = sum(history) / len(history) * 2 + self.region_availability_score[region]
            else:
                score = self.region_availability_score[region] + 1.0
                
            if score > best_score:
                best_score = score
                best_region = region
                
        if best_region != current_region and best_score > self.region_availability_score[current_region] + 0.2:
            self.current_best_region = best_region
            return True
            
        return False

    def _perform_region_switch(self):
        if self.current_best_region != self.env.get_current_region():
            self.env.switch_region(self.current_best_region)
            self.region_switches += 1
            self.time_in_region = 0
            self.consecutive_spot_failures = 0
            self.spot_success_streak = 0
            self.spot_attempts_in_current_region = 0