import argparse
import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_threshold"

    def solve(self, spec_path: str) -> "Solution":
        self.remaining_overhead = 0.0
        self.spot_usage_count = 0
        self.od_usage_count = 0
        self.in_spot = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        if self.remaining_overhead > 0:
            self.remaining_overhead -= gap
            if self.remaining_overhead > 0:
                return ClusterType.NONE
        
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done
        time_left = self.deadline - current_time
        
        if work_left <= 0:
            return ClusterType.NONE
        
        if self.remaining_overhead > 0:
            return ClusterType.NONE
        
        urgent = False
        safety_margin = self.restart_overhead * 2.5
        
        if time_left - work_left < safety_margin:
            urgent = True
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.remaining_overhead = self.restart_overhead
            self.in_spot = False
            if urgent:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        if urgent:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            spot_ratio = self.spot_usage_count / (self.spot_usage_count + self.od_usage_count + 1)
            
            time_needed_estimate = work_left
            if time_left - time_needed_estimate > self.restart_overhead * 4:
                self.in_spot = True
                self.spot_usage_count += 1
                return ClusterType.SPOT
            elif time_left - time_needed_estimate > self.restart_overhead * 2:
                if spot_ratio < 0.7:
                    self.in_spot = True
                    self.spot_usage_count += 1
                    return ClusterType.SPOT
                else:
                    self.in_spot = False
                    self.od_usage_count += 1
                    return ClusterType.ON_DEMAND
            else:
                self.in_spot = False
                self.od_usage_count += 1
                return ClusterType.ON_DEMAND
        else:
            self.in_spot = False
            self.od_usage_count += 1
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)