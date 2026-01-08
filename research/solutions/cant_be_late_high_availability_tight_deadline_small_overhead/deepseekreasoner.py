from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import pickle
import os
import math

class Solution(Strategy):
    NAME = "adaptive_safety_margin"

    def __init__(self, args):
        super().__init__(args)
        self.config = {}
        self.spot_price = 0.97
        self.ondemand_price = 3.06
        self.max_safety_factor = 1.2

    def solve(self, spec_path: str) -> "Solution":
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'rb') as f:
                    self.config = pickle.load(f)
            except:
                pass
        return self

    def _calculate_safety_margin(self, remaining_work, remaining_time, reliability_estimate):
        if remaining_work <= 0:
            return float('inf')
        
        if remaining_time <= 0:
            return 0
        
        min_margin = remaining_work / remaining_time
        base_reliability = 0.5
        reliability = max(0.01, reliability_estimate)
        
        safety_factor = 1.0 + (1.0 - reliability) * (self.max_safety_factor - 1.0)
        target_margin = min_margin * safety_factor
        
        return min(target_margin, 2.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.task_duration <= sum(self.task_done_time):
            return ClusterType.NONE
        
        elapsed = self.env.elapsed_seconds
        remaining_work = self.task_duration - sum(self.task_done_time)
        remaining_time = self.deadline - elapsed
        
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND
        
        work_done = sum(self.task_done_time[-100:]) if self.task_done_time else 0
        recent_steps = min(100, len(self.task_done_time))
        reliability_estimate = work_done / (recent_steps * self.env.gap_seconds) if recent_steps > 0 else 0.6
        
        required_rate = self._calculate_safety_margin(remaining_work, remaining_time, reliability_estimate)
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            if remaining_time < remaining_work + self.restart_overhead:
                return ClusterType.ON_DEMAND
        
        if required_rate >= 0.95:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            time_for_spot = remaining_work + (self.restart_overhead if last_cluster_type != ClusterType.SPOT else 0)
            if remaining_time > time_for_spot * 1.1:
                return ClusterType.SPOT
        
        if remaining_time < remaining_work * 1.05:
            return ClusterType.ON_DEMAND
        
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)