import json
from argparse import Namespace
import math
import os

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
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

        # Pre-process traces
        self.num_timesteps = math.ceil(self.deadline / self.env.gap_seconds) + 5
        self.traces = []
        spec_dir = os.path.dirname(spec_path)
        
        if "trace_files" in config and config["trace_files"]:
            for trace_file in config["trace_files"]:
                trace_path = os.path.join(spec_dir, trace_file)
                trace = []
                try:
                    with open(trace_path) as tf:
                        for line in tf:
                            trace.append(bool(int(line.strip())))
                except FileNotFoundError:
                    trace = [True] * self.num_timesteps

                if len(trace) < self.num_timesteps:
                    trace.extend([False] * (self.num_timesteps - len(trace)))
                self.traces.append(trace[:self.num_timesteps])

        if not self.traces:
            self.traces = [[True] * self.num_timesteps for _ in range(9)]

        num_regions = len(self.traces)
        self.future_spot_streak = [[0] * self.num_timesteps for _ in range(num_regions)]
        self.time_to_next_spot = [[self.num_timesteps] * self.num_timesteps for _ in range(num_regions)]

        for r in range(num_regions):
            if self.traces[r][self.num_timesteps - 1]:
                self.future_spot_streak[r][-1] = 1
                self.time_to_next_spot[r][-1] = 0
            else:
                self.future_spot_streak[r][-1] = 0
                self.time_to_next_spot[r][-1] = 1

            for t in range(self.num_timesteps - 2, -1, -1):
                if self.traces[r][t]:
                    self.future_spot_streak[r][t] = self.future_spot_streak[r][t+1] + 1
                    self.time_to_next_spot[r][t] = 0
                else:
                    self.future_spot_streak[r][t] = 0
                    self.time_to_next_spot[r][t] = self.time_to_next_spot[r][t+1] + 1
        
        for r in range(num_regions):
            if self.time_to_next_spot[r][0] >= self.num_timesteps - 1:
                for t in range(self.num_timesteps):
                    if self.time_to_next_spot[r][t] >= self.num_timesteps - 1:
                        self.time_to_next_spot[r][t] = self.num_timesteps
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        progress_done = sum(self.task_done_time)
        if progress_done >= self.task_duration:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        timestep = min(self.num_timesteps - 1, int(current_time // self.env.gap_seconds))

        # --- Region Selection ---
        streaks = [self.future_spot_streak[r][timestep] for r in range(num_regions)]
        if not any(streaks):
             target_region = current_region
        else:
             target_region = max(range(num_regions), key=lambda r: streaks[r])

        final_region = current_region
        if target_region != current_region:
            has_spot_current = self.traces[current_region][timestep]
            has_spot_target = self.traces[target_region][timestep]
            wait_time_current = self.time_to_next_spot[current_region][timestep] * self.env.gap_seconds

            if not has_spot_current and has_spot_target and wait_time_current > self.restart_overhead:
                final_region = target_region
        
        if final_region != current_region:
            self.env.switch_region(final_region)
        
        # --- Cluster Type Selection for `final_region` ---
        progress_needed = max(0, self.task_duration - progress_done)
        steps_needed_od = math.ceil(progress_needed / self.env.gap_seconds) if progress_needed > 0 else 0
        time_needed_od = steps_needed_od * self.env.gap_seconds
        
        has_spot_in_final = self.traces[final_region][timestep]

        # 1. Evaluate SPOT option
        if has_spot_in_final:
            time_if_preempted = current_time + self.env.gap_seconds
            finish_time_if_preempted = time_if_preempted + self.restart_overhead + time_needed_od
            if finish_time_if_preempted <= self.deadline:
                return ClusterType.SPOT
        
        # 2. Evaluate NONE (wait) option
        wait_time = self.time_to_next_spot[final_region][timestep] * self.env.gap_seconds
        if current_time + wait_time < self.deadline:
            time_after_wait = current_time + wait_time
            finish_time_if_wait = time_after_wait + self.restart_overhead + time_needed_od
            if finish_time_if_wait <= self.deadline:
                return ClusterType.NONE

        # 3. Fallback to ON_DEMAND
        return ClusterType.ON_DEMAND