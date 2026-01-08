import json
from argparse import Namespace
import math
import os

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "jit_od_switching"  # REQUIRED: unique identifier

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

        spec_dir = os.path.dirname(spec_path)
        if not spec_dir:
            spec_dir = '.'
        
        self.traces = []
        trace_files = config.get("trace_files", [])
        for trace_file in trace_files:
            full_trace_path = os.path.join(spec_dir, trace_file)
            try:
                with open(full_trace_path, 'r') as f:
                    trace_data = [line.strip() == '1' for line in f if line.strip()]
                    self.traces.append(trace_data)
            except FileNotFoundError:
                self.traces.append([])

        self.stability_scores = []
        for trace in self.traces:
            if len(trace) < 2:
                self.stability_scores.append(0.0)
                continue
            
            available_at_t_count = sum(trace[:-1])
            stayed_available_count = sum(1 for i in range(len(trace) - 1) if trace[i] and trace[i+1])

            if available_at_t_count == 0:
                self.stability_scores.append(0.0)
            else:
                score = stayed_available_count / available_at_t_count
                self.stability_scores.append(score)

        self._cached_work_done = 0.0
        self._cached_task_done_len = 0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if len(self.task_done_time) > self._cached_task_done_len:
            new_segments = self.task_done_time[self._cached_task_done_len:]
            self._cached_work_done += sum(new_segments)
            self._cached_task_done_len = len(self.task_done_time)
        
        work_done = self._cached_work_done
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        num_steps_needed = math.ceil(work_remaining / gap)
        time_needed_od = num_steps_needed * gap

        slack = self.deadline - (current_time + time_needed_od)

        overhead_steps = math.ceil(self.restart_overhead / gap)
        time_cost_of_failure = gap + (overhead_steps * gap)

        if slack < time_cost_of_failure:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            current_step_index = int(current_time // gap)
            current_region = self.env.get_current_region()

            available_regions = []
            for r_idx in range(self.env.get_num_regions()):
                if r_idx == current_region:
                    continue
                
                if current_step_index < len(self.traces[r_idx]) and self.traces[r_idx][current_step_index]:
                    available_regions.append(r_idx)

            if not available_regions:
                return ClusterType.NONE
            else:
                best_region = max(available_regions, key=lambda r: self.stability_scores[r])
                self.env.switch_region(best_region)
                return ClusterType.SPOT