import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.initial_slack = 0.0
        self.last_work_done_total = 0.0
        self.preemptions = 0

    def solve(self, spec_path: str) -> "Solution":
        if self.deadline > self.task_duration:
            self.initial_slack = self.deadline - self.task_duration
        else:
            self.initial_slack = 0.0
        
        self.last_work_done_total = 0.0
        self.preemptions = 0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done_total = sum(end - start for start, end in self.task_done_time)

        if (last_cluster_type == ClusterType.SPOT and
                self.last_work_done_total < self.task_duration - 1e-9 and
                abs(work_done_total - self.last_work_done_total) < 1e-9):
            self.preemptions += 1
        
        self.last_work_done_total = work_done_total

        work_remaining = self.task_duration - work_done_total
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        slack = time_to_deadline - work_remaining

        if slack <= self.restart_overhead:
            return ClusterType.ON_DEMAND

        od_frac_base = 0.15
        od_frac_penalty = 0.0075
        od_frac_max = 0.45
        
        none_frac_base = 0.50
        none_frac_penalty = 0.01
        none_frac_max = 0.85

        od_slack_frac = min(od_frac_max, od_frac_base + od_frac_penalty * self.preemptions)
        none_slack_frac = min(none_frac_max, none_frac_base + none_frac_penalty * self.preemptions)
        
        od_slack_threshold = od_slack_frac * self.initial_slack
        none_slack_threshold = none_slack_frac * self.initial_slack

        if slack < od_slack_threshold:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            return ClusterType.SPOT
        else:
            if slack < none_slack_threshold:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        args, _ = parser.parse_known_args()
        return cls(args)