import json
from argparse import Namespace
import math

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

        self.spot_availability = []
        for trace_file in config["trace_files"]:
            with open(trace_file) as f:
                trace_data = [c == '1' for c in f.read().strip()]
                self.spot_availability.append(trace_data)

        if not self.spot_availability:
            self.num_regions = 0
            self.max_ticks = 0
            self.steps_lost_on_switch = None
            return self

        self.num_regions = len(self.spot_availability)
        self.max_ticks = len(self.spot_availability[0]) if self.num_regions > 0 else 0

        self.forward_streaks = [[0] * (self.max_ticks + 1) for _ in range(self.num_regions)]
        for r in range(self.num_regions):
            for t in range(self.max_ticks - 1, -1, -1):
                if self.spot_availability[r][t]:
                    self.forward_streaks[r][t] = self.forward_streaks[r][t + 1] + 1

        self.next_any_spot = [self.max_ticks] * (self.max_ticks + 1)
        for t in range(self.max_ticks - 1, -1, -1):
            if any(self.spot_availability[r][t] for r in range(self.num_regions)):
                self.next_any_spot[t] = t
            else:
                self.next_any_spot[t] = self.next_any_spot[t + 1]

        self.steps_lost_on_switch = None

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if self.num_regions == 0:
            remaining_work_check = self.task_duration - sum(self.task_done_time)
            return ClusterType.ON_DEMAND if remaining_work_check > 0 else ClusterType.NONE

        if self.steps_lost_on_switch is None:
            self.steps_lost_on_switch = math.ceil(self.restart_overhead / self.env.gap_seconds)

        current_time = self.env.elapsed_seconds
        current_tick = int(current_time / self.env.gap_seconds)
        
        if current_tick >= self.max_ticks:
            remaining_work_check = self.task_duration - sum(self.task_done_time)
            return ClusterType.ON_DEMAND if remaining_work_check > 0 else ClusterType.NONE

        current_work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - current_work_done

        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - current_time

        od_time_needed = remaining_work
        if last_cluster_type != ClusterType.ON_DEMAND:
            od_time_needed += self.restart_overhead
        
        if od_time_needed >= time_to_deadline:
            return ClusterType.ON_DEMAND

        current_region = self.env.get_current_region()
        
        if has_spot:
            current_streak = self.forward_streaks[current_region][current_tick]
            
            best_other_region = None
            max_other_streak = -1
            for r in range(self.num_regions):
                if r == current_region:
                    continue
                if self.spot_availability[r][current_tick]:
                    if self.forward_streaks[r][current_tick] > max_other_streak:
                        max_other_streak = self.forward_streaks[r][current_tick]
                        best_other_region = r

            if best_other_region is not None and max_other_streak > current_streak + self.steps_lost_on_switch:
                self.env.switch_region(best_other_region)
                return ClusterType.SPOT
            else:
                return ClusterType.SPOT
        else:
            spot_options = []
            for r in range(self.num_regions):
                 if self.spot_availability[r][current_tick]:
                     spot_options.append(r)
            
            if spot_options:
                best_spot_region = max(spot_options, key=lambda r: self.forward_streaks[r][current_tick])
                self.env.switch_region(best_spot_region)
                return ClusterType.SPOT

        next_spot_tick = self.next_any_spot[current_tick]
        
        if next_spot_tick >= self.max_ticks:
            return ClusterType.ON_DEMAND

        time_to_wait_for_spot = (next_spot_tick - current_tick) * self.env.gap_seconds
        
        slack = time_to_deadline - (remaining_work + self.restart_overhead)
        
        if slack > time_to_wait_for_spot:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND