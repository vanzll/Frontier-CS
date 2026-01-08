import os
import sys
import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_deadline_aware"

    def __init__(self, args):
        super().__init__(args)
        self.spot_availability_history = []
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.conservative_threshold = 0.3
        self.aggressive_threshold = 0.7
        self.restart_penalty_factor = 1.5
        self.min_spot_streak = 3

    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    config_lines = f.readlines()
                    for line in config_lines:
                        if "conservative_threshold" in line:
                            self.conservative_threshold = float(
                                line.split("=")[1].strip())
                        elif "aggressive_threshold" in line:
                            self.aggressive_threshold = float(
                                line.split("=")[1].strip())
            except:
                pass
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)

        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        total_needed = self.task_duration
        done = sum(self.task_done_time)
        remaining_work = total_needed - done
        remaining_time = deadline - elapsed

        if remaining_work <= 0:
            return ClusterType.NONE

        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        critical_ratio = remaining_work / max(remaining_time, 0.1)

        avg_spot_availability = sum(self.spot_availability_history) / max(
            len(self.spot_availability_history), 1)

        recent_availability = sum(self.spot_availability_history[-5:]) / min(
            len(self.spot_availability_history), 5) if self.spot_availability_history else 0

        if last_cluster_type == ClusterType.SPOT and not has_spot:
            spot_streak = 0
        else:
            spot_streak = 0
            for avail in reversed(self.spot_availability_history):
                if avail:
                    spot_streak += 1
                else:
                    break

        expected_spot_work = gap
        if spot_streak < self.min_spot_streak and last_cluster_type != ClusterType.SPOT:
            expected_spot_work = gap * (spot_streak / self.min_spot_streak)

        effective_spot_cost = self.spot_price
        if last_cluster_type != ClusterType.SPOT and has_spot:
            effective_spot_cost += (self.restart_overhead * self.ondemand_price /
                                    gap) * self.restart_penalty_factor

        cost_ratio = effective_spot_cost / self.ondemand_price

        time_based_decision = self._time_based_decision(
            critical_ratio, remaining_time, gap)

        availability_based_decision = self._availability_based_decision(
            has_spot, avg_spot_availability, recent_availability, spot_streak)

        if time_based_decision == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if time_based_decision == ClusterType.NONE and availability_based_decision == ClusterType.SPOT:
            if has_spot and remaining_time > 2 * self.restart_overhead and spot_streak >= 2:
                return ClusterType.SPOT
            return ClusterType.NONE

        if availability_based_decision == ClusterType.SPOT:
            if has_spot:
                required_work_rate = remaining_work / remaining_time
                if required_work_rate > 1.0:
                    if expected_spot_work / gap >= 0.8:
                        return ClusterType.SPOT
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
            return ClusterType.NONE

        if time_based_decision == ClusterType.NONE:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    def _time_based_decision(self, critical_ratio, remaining_time, gap):
        if critical_ratio > self.aggressive_threshold:
            return ClusterType.ON_DEMAND
        elif critical_ratio < self.conservative_threshold:
            if remaining_time > 5 * gap:
                return ClusterType.NONE
            return ClusterType.SPOT
        else:
            return ClusterType.SPOT

    def _availability_based_decision(self, has_spot, avg_availability, recent_availability, spot_streak):
        if not has_spot:
            return ClusterType.NONE

        if recent_availability < 0.2 and spot_streak < 2:
            return ClusterType.NONE

        if avg_availability > 0.6 and recent_availability > 0.4:
            return ClusterType.SPOT

        if spot_streak >= self.min_spot_streak:
            return ClusterType.SPOT

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        parser.add_argument('--conservative_threshold', type=float,
                            default=0.3, help='Critical ratio threshold for conservative mode')
        parser.add_argument('--aggressive_threshold', type=float,
                            default=0.7, help='Critical ratio threshold for aggressive mode')
        args, _ = parser.parse_known_args()
        return cls(args)