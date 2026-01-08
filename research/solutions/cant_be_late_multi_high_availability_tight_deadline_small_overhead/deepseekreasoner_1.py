import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


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
        self.on_demand_price = 3.06
        self.spot_value = self.on_demand_price / self.spot_price
        self.safety_factor = 1.1
        
        self.spot_history = {}
        self.current_region = 0
        self.num_regions = 0
        self.time_step = 0
        self.best_regions = []
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.num_regions == 0:
            self.num_regions = self.env.get_num_regions()
            self.current_region = self.env.get_current_region()
        
        self.time_step += 1
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds
        
        if work_left <= 0:
            return ClusterType.NONE
        
        current_region = self.env.get_current_region()
        
        if current_region not in self.spot_history:
            self.spot_history[current_region] = []
        self.spot_history[current_region].append(has_spot)
        
        if len(self.spot_history[current_region]) > 100:
            self.spot_history[current_region] = self.spot_history[current_region][-100:]
        
        if time_left <= 0 or work_left > time_left:
            return ClusterType.ON_DEMAND
        
        min_time_needed = work_left + (0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead)
        
        if time_left < min_time_needed * self.safety_factor:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            recent_availability = sum(self.spot_history.get(current_region, [True][-20:])) / min(20, len(self.spot_history.get(current_region, [True])))
            
            if recent_availability > 0.8 or time_left > work_left * 1.5:
                return ClusterType.SPOT
            
            if last_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead <= 0:
                return ClusterType.SPOT
        
        if not has_spot and time_left > min_time_needed * 1.2:
            best_region = self._find_best_region()
            if best_region != current_region:
                self.env.switch_region(best_region)
                return ClusterType.NONE
        
        return ClusterType.ON_DEMAND if time_left < min_time_needed * 1.3 else ClusterType.NONE

    def _find_best_region(self) -> int:
        current = self.env.get_current_region()
        best_score = -1
        best_region = current
        
        for region in range(self.num_regions):
            if region == current:
                continue
            
            history = self.spot_history.get(region, [])
            if len(history) < 5:
                score = 0.5
            else:
                recent = history[-10:] if len(history) >= 10 else history
                score = sum(recent) / len(recent)
            
            if score > best_score:
                best_score = score
                best_region = region
        
        return best_region