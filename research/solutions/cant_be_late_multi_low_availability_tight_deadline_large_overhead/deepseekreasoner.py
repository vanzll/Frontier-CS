import json
from argparse import Namespace
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
from typing import List, Tuple
import heapq


class Solution(MultiRegionStrategy):
    NAME = "cost_aware_scheduler"

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
        
        self.traces = []
        for trace_file in config["trace_files"]:
            with open(trace_file, 'r') as f:
                self.traces.append([bool(int(line.strip())) for line in f])
        
        self.num_regions = len(self.traces)
        self.spot_price = 0.9701
        self.on_demand_price = 3.06
        self.gap_hours = self.env.gap_seconds / 3600.0
        
        return self

    def _calculate_safety_margin(self, work_done: float, time_elapsed: float) -> float:
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - time_elapsed
        if work_remaining <= 0:
            return float('inf')
        return (time_remaining - work_remaining) / self.gap_hours

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        time_elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
        
        safety_margin = self._calculate_safety_margin(work_done, time_elapsed)
        
        if safety_margin < 3:
            if has_spot:
                return ClusterType.SPOT
            else:
                best_region = current_region
                best_spot_availability = 0
                
                for region in range(self.num_regions):
                    if region == current_region:
                        continue
                        
                    current_step = int(time_elapsed / self.gap_hours)
                    if current_step < len(self.traces[region]) and self.traces[region][current_step]:
                        future_steps = min(5, len(self.traces[region]) - current_step)
                        spot_count = sum(1 for i in range(future_steps) 
                                       if self.traces[region][current_step + i])
                        
                        if spot_count > best_spot_availability:
                            best_spot_availability = spot_count
                            best_region = region
                
                if best_spot_availability > 0 and best_region != current_region:
                    self.env.switch_region(best_region)
                    return ClusterType.SPOT if self.traces[best_region][current_step] else ClusterType.ON_DEMAND
                else:
                    return ClusterType.ON_DEMAND
        
        if has_spot:
            return ClusterType.SPOT
        
        if safety_margin > 10:
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND