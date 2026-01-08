import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):

    NAME = "ucb_bailout_strategy"

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

        num_regions = self.env.get_num_regions()
        
        self.region_stats = [{'total': 0, 'up': 0} for _ in range(num_regions)]
        self.consecutive_downtime = [0] * num_regions
        self.total_steps = 0
        
        self.MAX_WAIT_STEPS = 3
        self.UCB_C = 1.414
        self.BAILOUT_SAFETY_MARGIN = self.env.gap_seconds * 1.0

        return self

    def _find_best_region(self, current_region: int, num_regions: int) -> int:
        unvisited = [r for r in range(num_regions) if self.region_stats[r]['total'] == 0]
        if unvisited:
            return unvisited[0]

        best_region = -1
        max_score = -1.0
        for r in range(num_regions):
            if r == current_region:
                continue
            
            stats = self.region_stats[r]
            if stats['total'] > 0:
                uptime = stats['up'] / stats['total']
                if self.total_steps > 0 and stats['total'] > 0:
                    exploration = self.UCB_C * (math.log(self.total_steps) / stats['total'])**0.5
                else:
                    exploration = float('inf')

                score = uptime + exploration
                
                if score > max_score:
                    max_score = score
                    best_region = r
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.total_steps += 1
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()

        self.region_stats[current_region]['total'] += 1
        if has_spot:
            self.region_stats[current_region]['up'] += 1
            self.consecutive_downtime[current_region] = 0
        else:
            self.consecutive_downtime[current_region] += 1
        
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        if work_remaining > 0:
            steps_needed = math.ceil(work_remaining / self.env.gap_seconds)
            time_for_work = steps_needed * self.env.gap_seconds
            bailout_time_needed = time_for_work + self.restart_overhead
        else:
            bailout_time_needed = 0

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        if time_to_deadline <= bailout_time_needed + self.BAILOUT_SAFETY_MARGIN:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        
        if num_regions > 1 and self.consecutive_downtime[current_region] >= self.MAX_WAIT_STEPS:
            self.consecutive_downtime[current_region] = 0
            best_region_to_switch = self._find_best_region(current_region, num_regions)
            
            if best_region_to_switch != -1 and best_region_to_switch != current_region:
                self.env.switch_region(best_region_to_switch)
                return ClusterType.NONE
        
        return ClusterType.NONE