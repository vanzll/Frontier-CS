import json
import os
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "heuristic_scheduler"

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

        # --- Strategy Parameters ---
        self.LOOKAHEAD_HOURS = 8.0
        self.WAIT_BUFFER_HOURS = 10.0
        self.SWITCH_THRESHOLD = 1
        
        # --- Internal State ---
        self.spot_availability = []
        self.lookahead_steps = None
        self.wait_buffer_seconds = self.WAIT_BUFFER_HOURS * 3600.0
        self.work_done = 0.0
        self.last_task_done_len = 0

        spec_dir = os.path.dirname(spec_path)
        if not spec_dir:
            spec_dir = '.'

        for trace_file in config["trace_files"]:
            full_path = os.path.join(spec_dir, trace_file)
            region_trace = []
            try:
                with open(full_path, 'r') as trace_f:
                    for line in trace_f:
                        region_trace.append(line.strip() == '1')
                self.spot_availability.append(region_trace)
            except IOError:
                self.spot_availability.append([])
        
        if self.spot_availability:
            max_len = 0
            for t in self.spot_availability:
                if len(t) > max_len:
                    max_len = len(t)

            for i in range(len(self.spot_availability)):
                if len(self.spot_availability[i]) < max_len:
                    self.spot_availability[i].extend([False] * (max_len - len(self.spot_availability[i])))

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if self.lookahead_steps is None and hasattr(self, 'env'):
            if self.env.gap_seconds > 0:
                self.lookahead_steps = int(self.LOOKAHEAD_HOURS * 3600 / self.env.gap_seconds)
            else:
                self.lookahead_steps = 0
        
        if len(self.task_done_time) > self.last_task_done_len:
            self.work_done += sum(self.task_done_time[self.last_task_done_len:])
            self.last_task_done_len = len(self.task_done_time)

        work_remaining = self.task_duration - self.work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        elapsed_seconds = self.env.elapsed_seconds
        time_to_deadline = self.deadline - elapsed_seconds
        
        must_run_od_threshold = work_remaining + self.restart_overhead
        if time_to_deadline <= must_run_od_threshold:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        current_region = self.env.get_current_region()
        current_timestep_idx = int(elapsed_seconds // self.env.gap_seconds) if self.env.gap_seconds > 0 else 0

        if not self.spot_availability or not self.spot_availability[0]:
            return ClusterType.ON_DEMAND

        max_trace_len = len(self.spot_availability[0])
        lookahead_end = min(current_timestep_idx + 1 + self.lookahead_steps, max_trace_len)

        current_region_future_spots = 0
        if current_timestep_idx + 1 < max_trace_len:
            current_region_future_spots = sum(
                self.spot_availability[current_region][t] for t in range(current_timestep_idx + 1, lookahead_end)
            )

        best_switch_region = -1
        max_score = -1
        
        num_regions = self.env.get_num_regions()
        for r in range(num_regions):
            if r < len(self.spot_availability) and current_timestep_idx < len(self.spot_availability[r]) and self.spot_availability[r][current_timestep_idx]:
                future_spots = 0
                if current_timestep_idx + 1 < max_trace_len:
                    future_spots = sum(
                       self.spot_availability[r][t] for t in range(current_timestep_idx + 1, lookahead_end)
                    )
                score = 1 + future_spots
                if score > max_score:
                    max_score = score
                    best_switch_region = r

        potential_current_score = self.SWITCH_THRESHOLD + current_region_future_spots
        if best_switch_region != -1 and max_score > potential_current_score:
            self.env.switch_region(best_switch_region)
            return ClusterType.SPOT

        if time_to_deadline > work_remaining + self.wait_buffer_seconds:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND