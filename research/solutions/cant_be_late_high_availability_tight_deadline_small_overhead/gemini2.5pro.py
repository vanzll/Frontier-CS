import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self.initialized = False
        self.pace = 0.0
        self.comfort_progress_buffer = 0.0
        self.comfort_buffer_multiplier = 10.0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self.initialized:
            if self.deadline > 0 and self.task_duration < self.deadline:
                self.pace = self.task_duration / self.deadline
            else:
                self.pace = 1.0
            self.comfort_progress_buffer = self.comfort_buffer_multiplier * self.restart_overhead
            self.initialized = True

        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        if time_to_deadline <= work_remaining + self.restart_overhead:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        work_required_by_now = self.env.elapsed_seconds * self.pace
        progress_buffer = work_done - work_required_by_now

        if progress_buffer > self.comfort_progress_buffer:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)