import json
from argparse import Namespace
import numpy as np

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

    RUNWAY_LOOKAHEAD_STEPS = 48
    WAIT_HORIZON_STEPS = 24
    SAFETY_MARGIN_SECONDS = 1.0

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

        traces = []
        if "trace_files" in config and config["trace_files"]:
            for trace_file in config["trace_files"]:
                try:
                    trace = np.loadtxt(trace_file, dtype=bool)
                    traces.append(trace)
                except (IOError, ValueError):
                    pass 

        if traces:
            max_len = 0
            for t in traces:
                if len(t.shape) > 0:
                    max_len = max(max_len, t.shape[0])

            padded_traces = []
            for t in traces:
                if len(t.shape) == 0 or t.shape[0] < max_len:
                    pad_width = max_len - (t.shape[0] if len(t.shape) > 0 else 0)
                    padded_trace = np.pad(t, (0, pad_width), 'constant', constant_values=False)
                    padded_traces.append(padded_trace)
                else:
                    padded_traces.append(t)
            if padded_traces:
                self.spot_availability = np.array(padded_traces)
            else:
                self.spot_availability = np.empty((0, 0), dtype=bool)
        else:
            self.spot_availability = np.empty((0, 0), dtype=bool)

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        if work_remaining <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        min_time_to_finish_on_demand = work_remaining + self.restart_overhead

        if time_to_deadline <= min_time_to_finish_on_demand + self.SAFETY_MARGIN_SECONDS:
            return ClusterType.ON_DEMAND

        if self.spot_availability.size == 0 or self.env.get_num_regions() != self.spot_availability.shape[0]:
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        current_step = int(self.env.elapsed_seconds // self.env.gap_seconds)
        trace_len = self.spot_availability.shape[1]

        if current_step >= trace_len:
            return ClusterType.ON_DEMAND

        if self.spot_availability[current_region, current_step]:
            return ClusterType.SPOT

        num_regions = self.env.get_num_regions()
        best_other_region = -1
        max_runway = 0
        
        lookahead_limit = min(trace_len, current_step + self.RUNWAY_LOOKAHEAD_STEPS)

        for j in range(num_regions):
            if self.spot_availability[j, current_step]:
                runway = 0
                for k in range(current_step, lookahead_limit):
                    if self.spot_availability[j, k]:
                        runway += 1
                    else:
                        break
                if runway > max_runway:
                    max_runway = runway
                    best_other_region = j
        
        if best_other_region != -1:
            self.env.switch_region(best_other_region)
            return ClusterType.SPOT

        wait_horizon_limit = min(trace_len, current_step + 1 + self.WAIT_HORIZON_STEPS)
        next_spot_step = -1
        for k in range(current_step + 1, wait_horizon_limit):
            if np.any(self.spot_availability[:, k]):
                next_spot_step = k
                break
        
        if next_spot_step != -1:
            wait_steps = next_spot_step - current_step
            wait_time_seconds = wait_steps * self.env.gap_seconds
            
            slack_time = time_to_deadline - min_time_to_finish_on_demand
            if slack_time > wait_time_seconds:
                return ClusterType.NONE

        return ClusterType.ON_DEMAND