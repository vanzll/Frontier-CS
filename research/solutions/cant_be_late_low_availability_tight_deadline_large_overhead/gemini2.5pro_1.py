from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "risk_aware_dynamic_buffer"

    def __init__(self, args):
        super().__init__(args)
        self.INITIAL_SAFETY_FACTOR = 5.0
        self.FINAL_SAFETY_FACTOR = 2.0
        self._work_done_cache = 0.0
        self._task_done_time_len_cache = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_work_done(self) -> float:
        if len(self.task_done_time) != self._task_done_time_len_cache:
            self._work_done_cache = sum(end - start for start, end in self.task_done_time)
            self._task_done_time_len_cache = len(self.task_done_time)
        return self._work_done_cache

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done = self._get_work_done()
        work_remaining = self.task_duration - work_done

        if work_remaining <= 1e-6:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_to_deadline = self.deadline - current_time

        if time_to_deadline <= 0:
            return ClusterType.NONE

        if has_spot:
            is_already_on_spot = (last_cluster_type == ClusterType.SPOT)
            switch_cost_to_spot = 0.0 if is_already_on_spot else self.restart_overhead
            time_needed_with_spot = work_remaining + switch_cost_to_spot

            if time_to_deadline > time_needed_with_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            is_already_on_od = (last_cluster_type == ClusterType.ON_DEMAND)
            switch_cost_to_od = 0.0 if is_already_on_od else self.restart_overhead
            time_needed_with_od = work_remaining + switch_cost_to_od

            progress = work_done / self.task_duration if self.task_duration > 0 else 1.0
            safety_factor = (self.INITIAL_SAFETY_FACTOR * (1.0 - progress) +
                             self.FINAL_SAFETY_FACTOR * progress)
            safety_buffer = safety_factor * self.restart_overhead

            if time_to_deadline <= time_needed_with_od + safety_buffer:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)