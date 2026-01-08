import math
from typing import List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.spot_price = 0.97
        self.on_demand_price = 3.06
        self.initial_task_duration = None
        self.estimated_spot_availability = 0.5
        self.critical_threshold = 0.2
        self.conservative_mode = False
        self.spot_unavailable_streak = 0
        self.max_spot_wait = 4

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _calculate_remaining(self) -> Tuple[float, float]:
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - self.env.elapsed_seconds
        return remaining_work, remaining_time

    def _should_switch_to_ondemand(self, remaining_work: float, remaining_time: float, has_spot: bool) -> bool:
        if remaining_work <= 0:
            return False

        time_per_work_unit = self.env.gap_seconds
        if has_spot and not self.conservative_mode:
            effective_rate = 0.7
            estimated_time = remaining_work / effective_rate
        else:
            estimated_time = remaining_work

        safety_margin = 2 * self.restart_overhead

        if remaining_time - estimated_time < safety_margin:
            return True

        if remaining_time < remaining_work * 1.2:
            return True

        if self.conservative_mode and remaining_time < remaining_work * 1.5:
            return True

        if self.spot_unavailable_streak > self.max_spot_wait:
            return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.initial_task_duration is None:
            self.initial_task_duration = self.task_duration

        remaining_work, remaining_time = self._calculate_remaining()

        if remaining_work <= 0:
            return ClusterType.NONE

        work_done_ratio = 1 - (remaining_work / self.initial_task_duration)
        time_used_ratio = self.env.elapsed_seconds / self.deadline

        if work_done_ratio < time_used_ratio - self.critical_threshold:
            self.conservative_mode = True

        if not has_spot:
            self.spot_unavailable_streak += 1
        else:
            self.spot_unavailable_streak = 0

        if self._should_switch_to_ondemand(remaining_work, remaining_time, has_spot):
            if remaining_time > 0 and remaining_work > 0:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

        if has_spot:
            if self.env.cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            else:
                overhead_buffer = self.restart_overhead * 1.5
                if remaining_time > remaining_work + overhead_buffer:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        else:
            if remaining_time > remaining_work + self.restart_overhead * 3:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND