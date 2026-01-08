import json
from argparse import Namespace
import os

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "precompute_and_panic"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self.num_regions = len(config['trace_files'])
        self.region_quality = []
        
        base_path = os.path.dirname(spec_path)

        for trace_file in config['trace_files']:
            full_trace_path = os.path.join(base_path, trace_file)
            try:
                if os.path.exists(full_trace_path):
                    with open(full_trace_path) as f:
                        trace_data = [int(line.strip()) for line in f if line.strip()]
                    if not trace_data:
                        quality = 0.0
                    else:
                        quality = sum(trace_data) / len(trace_data)
                else:
                    quality = 0.0 # Assume no availability if trace is missing
                self.region_quality.append(quality)
            except (IOError, ValueError):
                self.region_quality.append(0.0)
        
        if not self.region_quality and self.num_regions > 0:
             self.region_quality = [1.0] * self.num_regions

        if self.num_regions > 0:
            self.sorted_regions = sorted(range(self.num_regions), key=lambda i: self.region_quality[i], reverse=True)
        else:
            self.sorted_regions = []

        self.is_initialized = False
        self.wait_counter = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.is_initialized:
            self.is_initialized = True
            if self.sorted_regions:
                best_region_idx = self.sorted_regions[0]
                if self.env.get_current_region() != best_region_idx:
                    self.env.switch_region(best_region_idx)
                    return ClusterType.ON_DEMAND

        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        current_region_idx = self.env.get_current_region()

        time_needed_for_on_demand = work_remaining + self.restart_overhead
        if time_left <= time_needed_for_on_demand:
            return ClusterType.ON_DEMAND

        if has_spot:
            self.wait_counter = 0
            return ClusterType.SPOT
        else:
            slack_time = time_left - work_remaining
            max_wait_steps = 1
            
            is_current_region_the_best = self.sorted_regions and (current_region_idx == self.sorted_regions[0])
            can_afford_to_wait = slack_time > (self.env.gap_seconds * max_wait_steps + self.restart_overhead)
            
            if is_current_region_the_best and self.wait_counter < max_wait_steps and can_afford_to_wait:
                self.wait_counter += 1
                return ClusterType.NONE

            self.wait_counter = 0
            
            next_best_region_idx = -1
            if self.sorted_regions:
                for region_idx in self.sorted_regions:
                    if region_idx != current_region_idx:
                        next_best_region_idx = region_idx
                        break
            
            can_afford_to_switch = slack_time > self.restart_overhead
            
            if next_best_region_idx != -1 and can_afford_to_switch:
                self.env.switch_region(next_best_region_idx)
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND