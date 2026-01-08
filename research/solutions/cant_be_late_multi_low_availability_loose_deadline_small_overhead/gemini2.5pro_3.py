import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    Your multi-region scheduling strategy.
    """
    NAME = "HeuristicSchedulerV1"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config and pre-compute
        spot availability predictions.
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

        self._precompute(config)

        return self

    def _precompute(self, config: dict):
        """
        Loads traces and pre-computes availability predictions.
        """
        self.is_panic_mode = False
        
        PREDICTION_WINDOW_SECONDS = 3600.0
        
        self.traces = []
        if "trace_files" in config:
            trace_paths = config["trace_files"]
            for trace_path in trace_paths:
                try:
                    with open(trace_path) as f:
                        trace = [int(line.strip()) for line in f.readlines()]
                        self.traces.append(trace)
                except (IOError, ValueError):
                    self.traces.append([])

        self.num_regions = len(self.traces)
        if self.num_regions > 0 and self.traces[0]:
            self.trace_length = len(self.traces[0])
        else:
            self.trace_length = 0

        self.overall_avg_availability = [
            sum(t) / len(t) if t else 0.0 for t in self.traces
        ]

        if self.trace_length == 0 or not hasattr(self, 'env') or self.env.gap_seconds <= 0:
            self.predicted_availability = [[] for _ in range(self.num_regions)]
            return

        self.prediction_window_steps = max(1, int(PREDICTION_WINDOW_SECONDS / self.env.gap_seconds))
        self.predicted_availability = []

        for r in range(self.num_regions):
            trace = self.traces[r]
            if not trace:
                self.predicted_availability.append([0.0] * self.trace_length)
                continue

            preds = [0.0] * self.trace_length
            cumsum = [0] * (self.trace_length + 1)
            for i in range(self.trace_length):
                cumsum[i + 1] = cumsum[i] + trace[i]

            for t in range(self.trace_length):
                start = t
                end = min(t + self.prediction_window_steps, self.trace_length)
                window_sum = cumsum[end] - cumsum[start]
                window_len = end - start
                if window_len > 0:
                    preds[t] = window_sum / window_len
                else:
                    preds[t] = self.overall_avg_availability[r]
            
            self.predicted_availability.append(preds)

    def _get_predicted_availability(self, region: int, timestep: int) -> float:
        """
        Returns the pre-computed predicted spot availability for a given
        region and timestep.
        """
        if not (0 <= region < self.num_regions):
            return 0.0
        
        if not hasattr(self, 'predicted_availability') or not self.predicted_availability or not self.predicted_availability[region]:
             return self.overall_avg_availability[region] if hasattr(self, 'overall_avg_availability') and 0 <= region < len(self.overall_avg_availability) else 0.0

        if timestep >= self.trace_length:
            return self.overall_avg_availability[region]
        
        return self.predicted_availability[region][timestep]

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        SAFETY_RESTARTS = 2.0
        SWITCH_MARGIN = 0.2
        SWITCH_THRESHOLD = 0.3
        WAIT_SLACK_RESTARTS = 5.0
        WAIT_THRESHOLD = 0.5

        time_now = self.env.elapsed_seconds
        time_left = self.deadline - time_now
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done

        if work_left <= 0:
            return ClusterType.NONE

        time_needed_for_ondemand = work_left + SAFETY_RESTARTS * self.restart_overhead
        if time_left <= time_needed_for_ondemand:
            self.is_panic_mode = True

        if self.is_panic_mode:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            return ClusterType.SPOT

        current_region = self.env.get_current_region()
        current_timestep = int(time_now / self.env.gap_seconds) if self.env.gap_seconds > 0 else 0

        best_other_region = -1
        max_pred_avail = -1.0
        
        current_pred_avail = self._get_predicted_availability(current_region, current_timestep)

        for r in range(self.num_regions):
            if r == current_region:
                continue
            
            pred = self._get_predicted_availability(r, current_timestep)
            if pred > max_pred_avail:
                max_pred_avail = pred
                best_other_region = r

        if (best_other_region != -1 and
                max_pred_avail > current_pred_avail + SWITCH_MARGIN and
                max_pred_avail > SWITCH_THRESHOLD):
            self.env.switch_region(best_other_region)
            return ClusterType.ON_DEMAND

        slack_time = time_left - work_left
        wait_slack_threshold = WAIT_SLACK_RESTARTS * self.restart_overhead
        
        pred_avail_next_step = self._get_predicted_availability(current_region, current_timestep + 1)
        
        if (slack_time > wait_slack_threshold and
                pred_avail_next_step > WAIT_THRESHOLD):
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND