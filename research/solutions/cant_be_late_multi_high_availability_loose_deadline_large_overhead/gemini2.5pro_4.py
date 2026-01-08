import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that balances cost and deadline risk.
    """

    NAME = "my_strategy"

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

        self.traces = []
        if "trace_files" in config:
            for trace_file in config["trace_files"]:
                with open(trace_file, 'r') as f:
                    trace = [bool(int(line.strip())) for line in f]
                    self.traces.append(trace)
        
        self.lookahead_window_hours = 4.0
        self.safety_margin_hours = 6.0
        
        self._initialized_step_params = False
        return self

    def _initialize_step_params(self):
        """
        One-time initialization of parameters that depend on the environment.
        """
        self.gap_seconds = self.env.gap_seconds
        self.cost_in_steps = self.restart_overhead / self.gap_seconds
        self.w_steps = int(self.lookahead_window_hours * 3600 / self.gap_seconds)
        self.safety_margin_seconds = self.safety_margin_hours * 3600
        
        if self.traces:
            self.max_trace_len = len(self.traces[0])
            self.num_traces = len(self.traces)
        else:
            self.max_trace_len = 0
            self.num_traces = 0
            
        self._initialized_step_params = True
        
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self._initialized_step_params:
            self._initialize_step_params()

        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time
        
        time_needed_for_od_fallback = remaining_work + self.restart_overhead
        if time_left <= time_needed_for_od_fallback:
            return ClusterType.ON_DEMAND

        current_idx = int(current_time / self.gap_seconds)
        
        has_spot_in_target_region = False
        
        if self.num_traces > 0 and current_idx < self.max_trace_len:
            current_region = self.env.get_current_region()
            
            if self.num_traces <= current_region:
                # Fallback if current region index is out of trace bounds
                has_spot_in_target_region = False
            else:
                effective_w_steps = min(self.w_steps, self.max_trace_len - current_idx)

                future_spot_avail = [
                    sum(trace[current_idx : current_idx + effective_w_steps])
                    for trace in self.traces
                ]

                best_region_idx = max(range(self.num_traces), key=future_spot_avail.__getitem__)
                
                gain_in_steps = future_spot_avail[best_region_idx] - future_spot_avail[current_region]
                
                target_region = current_region
                if gain_in_steps > self.cost_in_steps and best_region_idx != current_region:
                    target_region = best_region_idx
                    self.env.switch_region(target_region)
                
                has_spot_in_target_region = self.traces[target_region][current_idx]

        if has_spot_in_target_region:
            return ClusterType.SPOT
        else:
            soft_deadline = self.deadline - self.safety_margin_seconds
            time_left_for_soft = soft_deadline - current_time

            if time_left_for_soft <= time_needed_for_od_fallback:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE