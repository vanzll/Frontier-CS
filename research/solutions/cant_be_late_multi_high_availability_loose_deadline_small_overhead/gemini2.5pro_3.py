import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "CBL_Scheduler_v1"

    LOOKAHEAD_WINDOW_HOURS = 24.0
    SWITCH_SCORE_GAIN_THRESHOLD = 2.0
    SLACK_THRESHOLD_HOURS = 3.0

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

        self.availability = []
        for trace_file in config["trace_files"]:
            with open(trace_file) as f:
                region_trace = [int(float(line.strip())) for line in f]
                self.availability.append(region_trace)
        
        if not self.availability:
            self.num_regions = 0
            return self

        self.num_regions = len(self.availability)
        num_timesteps = len(self.availability[0])
        gap = self.env.gap_seconds
        if gap == 0:
            return self

        lookahead_window_steps = int(self.LOOKAHEAD_WINDOW_HOURS * 3600 / gap)
        
        self.spot_counts = [[0] * num_timesteps for _ in range(self.num_regions)]
        for r in range(self.num_regions):
            trace = self.availability[r]
            
            prefix_sum = [0] * (num_timesteps + 1)
            current_sum = 0
            for i in range(num_timesteps):
                current_sum += trace[i]
                prefix_sum[i+1] = current_sum

            for t in range(num_timesteps):
                end_idx = min(t + lookahead_window_steps, num_timesteps)
                self.spot_counts[r][t] = prefix_sum[end_idx] - prefix_sum[t]

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        work_done = sum(self.task_done_time)
        if work_done >= self.task_duration:
            return ClusterType.NONE

        if self.num_regions == 0:
            return ClusterType.ON_DEMAND

        w_rem = self.task_duration - work_done
        t_rem = self.deadline - self.env.elapsed_seconds
        
        num_steps_needed = math.ceil(w_rem / self.env.gap_seconds) if w_rem > 0 else 0
        time_needed_od_ideal = num_steps_needed * self.env.gap_seconds

        current_t_idx = int(self.env.elapsed_seconds // self.env.gap_seconds)
        if current_t_idx >= len(self.availability[0]):
             return ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        
        scores = [self.spot_counts[r][current_t_idx] for r in range(self.num_regions)]
        current_score = scores[current_region]

        best_alt_region = -1
        max_alt_score = -1.0
        for r_idx in range(self.num_regions):
            if self.availability[r_idx][current_t_idx]:
                score = scores[r_idx]
                if score > max_alt_score:
                    max_alt_score = score
                    best_alt_region = r_idx

        should_switch = False
        if best_alt_region != -1 and best_alt_region != current_region:
            switch_gain_threshold_steps = self.SWITCH_SCORE_GAIN_THRESHOLD * 3600 / self.env.gap_seconds

            if not has_spot:
                should_switch = True
            elif max_alt_score > current_score + switch_gain_threshold_steps:
                should_switch = True
        
        if should_switch:
            time_needed_after_switch = time_needed_od_ideal + self.restart_overhead
            if t_rem > time_needed_after_switch:
                self.env.switch_region(best_alt_region)
                return ClusterType.SPOT

        needs_restart_for_od = (last_cluster_type == ClusterType.SPOT)
        time_needed_with_od = time_needed_od_ideal + (self.restart_overhead if needs_restart_for_od else 0)

        if t_rem <= time_needed_with_od:
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            slack = t_rem - time_needed_with_od
            slack_threshold_seconds = self.SLACK_THRESHOLD_HOURS * 3600
            if slack < slack_threshold_seconds:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE