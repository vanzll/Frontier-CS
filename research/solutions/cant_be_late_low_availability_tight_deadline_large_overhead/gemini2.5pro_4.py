import math
from argparse import ArgumentParser
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self.was_spot_available = True
        self.downtime_start_time = -1.0
        self.downtime_periods = []
        self.downtime_prior_avg = 3600.0
        self.downtime_prior_count = 1
        self.safety_factor = 1.2
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not has_spot and self.was_spot_available:
            self.downtime_start_time = self.env.elapsed_seconds
        elif has_spot and not self.was_spot_available:
            if self.downtime_start_time > -1.0:
                duration = self.env.elapsed_seconds - self.downtime_start_time
                self.downtime_periods.append(duration)
                self.downtime_start_time = -1.0
        self.was_spot_available = has_spot

        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0.0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        slack = time_to_deadline - work_remaining

        total_downtime_sum = sum(self.downtime_periods) + self.downtime_prior_avg * self.downtime_prior_count
        num_periods = len(self.downtime_periods) + self.downtime_prior_count
        avg_downtime = total_downtime_sum / num_periods if num_periods > 0 else self.downtime_prior_avg

        current_downtime = 0.0
        if self.downtime_start_time > -1.0:
            current_downtime = self.env.elapsed_seconds - self.downtime_start_time
        
        effective_downtime_est = max(avg_downtime, current_downtime)

        slack_threshold = self.safety_factor * (self.restart_overhead + effective_downtime_est)

        if has_spot:
            if slack > slack_threshold:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            if slack > slack_threshold:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: ArgumentParser):
        args, _ = parser.parse_known_args()
        return cls(args)