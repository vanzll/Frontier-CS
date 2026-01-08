import json
from argparse import Namespace
import math
from collections import defaultdict

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "optimized_spot_strategy"

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
        
        self.regions = self.env.get_num_regions()
        self.step_hours = self.env.gap_seconds / 3600.0
        self.deadline_hours = self.deadline / 3600.0
        self.task_hours = self.task_duration / 3600.0
        self.overhead_hours = self.restart_overhead / 3600.0
        
        self.trace_files = config["trace_files"]
        self.spot_availability = self._load_traces()
        
        return self

    def _load_traces(self):
        """Load and preprocess spot availability traces."""
        availability = {}
        for region_idx, trace_file in enumerate(self.trace_files):
            with open(trace_file, 'r') as f:
                lines = f.readlines()
            availability[region_idx] = [int(line.strip()) for line in lines[:int(self.deadline_hours/self.step_hours)+1]]
        return availability

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        elapsed_hours = self.env.elapsed_seconds / 3600.0
        
        work_done_hours = sum(self.task_done_time) / 3600.0
        work_remaining_hours = self.task_hours - work_done_hours
        
        time_remaining_hours = self.deadline_hours - elapsed_hours
        
        if work_remaining_hours <= 0:
            return ClusterType.NONE
        
        effective_time_remaining = time_remaining_hours - self.remaining_restart_overhead / 3600.0
        
        if effective_time_remaining <= 0:
            return ClusterType.ON_DEMAND
        
        critical_ratio = work_remaining_hours / effective_time_remaining
        
        if critical_ratio > 1.0:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            spot_score = self._calculate_spot_score(current_region, elapsed_hours, work_remaining_hours)
            
            future_spot_hours = self._estimate_future_spot(current_region, elapsed_hours, work_remaining_hours)
            
            if future_spot_hours >= work_remaining_hours * 0.8:
                return ClusterType.SPOT
            
            if spot_score > 0.7 and critical_ratio < 0.8:
                return ClusterType.SPOT
            
            if critical_ratio > 0.6:
                return ClusterType.ON_DEMAND
                
            if spot_score > 0.3:
                return ClusterType.SPOT
        
        if not has_spot:
            best_region = self._find_best_spot_region(elapsed_hours)
            if best_region != current_region:
                self.env.switch_region(best_region)
                return ClusterType.SPOT if self.spot_availability[best_region][int(elapsed_hours/self.step_hours)] else ClusterType.ON_DEMAND
        
        if critical_ratio < 0.4:
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND

    def _calculate_spot_score(self, region, elapsed_hours, work_remaining_hours):
        """Calculate how favorable spot is for this region."""
        current_step = int(elapsed_hours / self.step_hours)
        lookahead_steps = min(10, int(work_remaining_hours / self.step_hours) + 1)
        
        spot_count = 0
        for i in range(lookahead_steps):
            if current_step + i < len(self.spot_availability[region]):
                spot_count += self.spot_availability[region][current_step + i]
        
        return spot_count / lookahead_steps

    def _estimate_future_spot(self, region, elapsed_hours, work_remaining_hours):
        """Estimate available spot hours in the remaining time."""
        current_step = int(elapsed_hours / self.step_hours)
        steps_remaining = int(work_remaining_hours / self.step_hours) + 1
        
        total_steps = min(steps_remaining, len(self.spot_availability[region]) - current_step)
        if total_steps <= 0:
            return 0
        
        available = 0
        for i in range(total_steps):
            if current_step + i < len(self.spot_availability[region]):
                available += self.spot_availability[region][current_step + i]
        
        return available * self.step_hours

    def _find_best_spot_region(self, elapsed_hours):
        """Find region with best immediate spot availability."""
        current_step = int(elapsed_hours / self.step_hours)
        best_region = self.env.get_current_region()
        best_score = 0
        
        for region in range(self.regions):
            if region == self.env.get_current_region():
                continue
                
            future_score = 0
            lookahead = min(5, len(self.spot_availability[region]) - current_step)
            for i in range(lookahead):
                if current_step + i < len(self.spot_availability[region]):
                    future_score += self.spot_availability[region][current_step + i]
            
            if future_score > best_score:
                best_score = future_score
                best_region = region
        
        return best_region