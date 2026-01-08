import json
import os
from argparse import Namespace

import numpy as np

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

        self.traces = []
        spec_dir = os.path.dirname(spec_path)
        for trace_file in config["trace_files"]:
            full_path = trace_file
            if not os.path.isabs(full_path):
                full_path = os.path.join(spec_dir, trace_file)
            
            with open(full_path) as f:
                trace = np.array([bool(int(line.strip())) for line in f])
                self.traces.append(trace)

        self.num_regions = len(self.traces)
        self.trace_len = len(self.traces[0]) if self.num_regions > 0 else 0

        self.region_scores = []
        for i in range(self.num_regions):
            if self.trace_len == 0:
                self.region_scores.append(0)
                continue
            
            trace = self.traces[i]
            availability = np.mean(trace)
            
            switches = np.count_nonzero(np.diff(trace))
            instability = switches / self.trace_len if self.trace_len > 1 else 0
            
            score = availability - 0.5 * instability
            self.region_scores.append(score)

        self.ranked_regions = sorted(
            range(self.num_regions),
            key=lambda i: self.region_scores[i],
            reverse=True)
        
        self.initial_region_set = False
        
        self.SAFETY_BUFFER = 2 * self.restart_overhead
        
        if self.env.gap_seconds > 0:
            look_ahead_seconds = 3 * self.restart_overhead
            self.LOOK_AHEAD_STEPS = int(look_ahead_seconds / self.env.gap_seconds)
        else:
            self.LOOK_AHEAD_STEPS = 1

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initial_region_set and self.ranked_regions:
            best_region = self.ranked_regions[0]
            self.env.switch_region(best_region)
            self.initial_region_set = True

        progress = sum(self.task_done_time)
        remaining_work = self.task_duration - progress

        if remaining_work <= 0:
            return ClusterType.NONE

        elapsed_time = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed_time
        
        time_needed_on_demand = remaining_work + self.remaining_restart_overhead

        if remaining_time <= time_needed_on_demand + self.SAFETY_BUFFER:
            return ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        
        current_timestep = 0
        if self.env.gap_seconds > 0:
            current_timestep = int(elapsed_time / self.env.gap_seconds)

        if has_spot:
            return ClusterType.SPOT

        best_switch_region = -1
        max_future_spot = -1

        for region_idx in self.ranked_regions:
            if region_idx == current_region:
                continue

            if current_timestep < self.trace_len and self.traces[region_idx][current_timestep]:
                end_step = min(self.trace_len, current_timestep + self.LOOK_AHEAD_STEPS)
                future_spot_steps = np.sum(self.traces[region_idx][current_timestep:end_step])

                if future_spot_steps > max_future_spot:
                    max_future_spot = future_spot_steps
                    best_switch_region = region_idx
        
        if best_switch_region != -1:
            self.env.switch_region(best_switch_region)
            return ClusterType.SPOT

        return ClusterType.ON_DEMAND