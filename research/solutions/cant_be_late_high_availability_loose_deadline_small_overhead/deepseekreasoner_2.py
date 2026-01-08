import os
import json
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import math

class Solution(Strategy):
    NAME = "adaptive_threshold"

    def __init__(self, args):
        super().__init__(args)
        self.spec = None
        self.spot_price = None
        self.od_price = None
        self.cost_ratio = None
        self.min_spot_frac = 0.43
        self.max_spot_frac = 0.78
        self.base_slack_factor = 1.3

    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                self.spec = json.load(f)
        return self

    def _calculate_dynamic_threshold(self, elapsed: float, remaining_work: float, 
                                     time_to_deadline: float, spot_available: bool) -> float:
        """Calculate adaptive threshold for switching to on-demand."""
        if time_to_deadline <= 0:
            return 0
        
        work_ratio = remaining_work / self.task_duration
        time_ratio = time_to_deadline / (self.deadline - elapsed)
        
        urgency = remaining_work / max(time_to_deadline, 1e-6)
        
        if urgency > 1.2:
            threshold = 0.2
        elif urgency > 1.0:
            threshold = 0.4
        elif urgency > 0.8:
            threshold = 0.6
        elif urgency > 0.6:
            threshold = 0.75
        else:
            threshold = 0.9
        
        if not spot_available:
            threshold *= 0.7
        
        if work_ratio > 0.8 and time_ratio < 0.5:
            threshold *= 0.6
        
        return max(0.1, min(0.95, threshold))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        restart_overhead = self.restart_overhead
        
        total_work_done = sum(end - start for start, end in self.task_done_time)
        remaining_work = max(0, self.task_duration - total_work_done)
        time_to_deadline = max(0, deadline - elapsed)
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        time_needed_on_od = remaining_work
        time_needed_on_spot = remaining_work + restart_overhead
        
        urgent = time_to_deadline < time_needed_on_od * 1.1
        
        if urgent:
            return ClusterType.ON_DEMAND if has_spot else ClusterType.NONE
        
        threshold = self._calculate_dynamic_threshold(
            elapsed, remaining_work, time_to_deadline, has_spot
        )
        
        spot_time_frac = 0.6
        can_use_spot = (
            has_spot and 
            time_to_deadline > time_needed_on_spot * threshold and
            not (last_cluster_type == ClusterType.SPOT and 
                 self.env.cluster_type == ClusterType.NONE)
        )
        
        if can_use_spot:
            if last_cluster_type != ClusterType.SPOT:
                time_with_overhead = remaining_work + restart_overhead
                if time_to_deadline > time_with_overhead * 1.15:
                    return ClusterType.SPOT
            else:
                return ClusterType.SPOT
        
        if time_to_deadline < time_needed_on_od * 1.25:
            return ClusterType.ON_DEMAND if has_spot else ClusterType.NONE
        
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)