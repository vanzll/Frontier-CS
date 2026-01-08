import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_heuristic_scheduler"

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

        self.num_regions = len(config.get("trace_files", []))
        
        self.spot_history = [[1.0, 1.0] for _ in range(self.num_regions)]
        
        self.decay_factor = 0.998
        
        self.PROBABILITY_IMPROVEMENT_THRESHOLD = 0.15
        self.SWITCH_SLACK_BUFFER_FACTOR = 1.5

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        
        self.spot_history[current_region][0] *= self.decay_factor
        self.spot_history[current_region][1] *= self.decay_factor
        
        if has_spot:
            self.spot_history[current_region][0] += 1.0
        else:
            self.spot_history[current_region][1] += 1.0

        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_time = self.env.elapsed_seconds
        time_left = self.deadline - elapsed_time

        time_needed_od_worst_case = work_remaining + self.restart_overhead
        if time_needed_od_worst_case >= time_left:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        probas = []
        for i, (alpha, beta) in enumerate(self.spot_history):
            probas.append((alpha / (alpha + beta), i))

        current_proba, _ = probas[current_region]
        
        other_regions_probas = [p for p in probas if p[1] != current_region]
        
        if other_regions_probas:
            best_other_proba, best_other_region_idx = max(other_regions_probas)
            
            slack = time_left - work_remaining
            min_slack_for_switch = self.restart_overhead + self.env.gap_seconds * self.SWITCH_SLACK_BUFFER_FACTOR
            
            if (slack > min_slack_for_switch and
                    best_other_proba > current_proba + self.PROBABILITY_IMPROVEMENT_THRESHOLD):
                
                self.env.switch_region(best_other_region_idx)
                return ClusterType.SPOT

        slack = time_left - work_remaining
        patience_threshold = self.env.gap_seconds + self.restart_overhead
        
        if slack < patience_threshold:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE