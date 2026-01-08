import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_slack_aware"

    def __init__(self, args):
        super().__init__(args)
        self.remaining_work = None
        self.spot_price = 0.97
        self.od_price = 3.06
        self.price_ratio = self.od_price / self.spot_price
        self.safety_factor = 1.2
        self.min_spot_attempt_time = 1800
        self.spot_unavailable_counter = 0
        self.current_restart_overhead = 0
        self.last_spot_start = 0
        self.consecutive_spot_failures = 0
        self.spot_availability_history = []

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        if self.remaining_work is None:
            self.remaining_work = self.task_duration
        
        if len(self.task_done_time) > 0:
            work_done = sum(self.task_done_time)
            self.remaining_work = self.task_duration - work_done
        
        time_to_deadline = self.deadline - elapsed
        required_rate = self.remaining_work / time_to_deadline if time_to_deadline > 0 else float('inf')
        
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        spot_availability_rate = (
            sum(self.spot_availability_history) / len(self.spot_availability_history)
            if self.spot_availability_history else 0
        )
        
        if self.current_restart_overhead > 0:
            self.current_restart_overhead = max(0, self.current_restart_overhead - gap)
        
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.current_restart_overhead = self.restart_overhead
            self.consecutive_spot_failures += 1
            self.spot_unavailable_counter += 1
        elif last_cluster_type == ClusterType.SPOT and has_spot:
            self.consecutive_spot_failures = 0
            self.spot_unavailable_counter = max(0, self.spot_unavailable_counter - 1)
        elif last_cluster_type == ClusterType.ON_DEMAND:
            self.consecutive_spot_failures = 0
            self.spot_unavailable_counter = max(0, self.spot_unavailable_counter - 2)
        
        if self.current_restart_overhead > 0:
            return ClusterType.NONE
        
        if time_to_deadline <= 0 or self.remaining_work <= 0:
            return ClusterType.NONE
        
        time_needed = self.remaining_work
        safety_margin = min(self.restart_overhead * 3, time_to_deadline * 0.1)
        
        urgency_factor = time_needed / (time_to_deadline - safety_margin) if time_to_deadline > safety_margin else float('inf')
        
        if urgency_factor > 1.0:
            if has_spot and self.consecutive_spot_failures < 3:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        if not has_spot:
            if self.spot_unavailable_counter > 10 and time_to_deadline < time_needed * 2:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
        
        if elapsed - self.last_spot_start < self.min_spot_attempt_time:
            return ClusterType.SPOT
        
        spot_risk_factor = max(1.0, self.consecutive_spot_failures * 0.5)
        
        expected_spot_time = time_needed / (spot_availability_rate + 0.01)
        expected_spot_time_with_overhead = expected_spot_time + (self.restart_overhead * spot_risk_factor)
        
        if expected_spot_time_with_overhead * self.spot_price * self.safety_factor < time_needed * self.od_price:
            if time_to_deadline > expected_spot_time_with_overhead * 1.5:
                self.last_spot_start = elapsed
                return ClusterType.SPOT
        
        if time_to_deadline < time_needed + self.restart_overhead * 2:
            return ClusterType.ON_DEMAND
        
        if spot_availability_rate > 0.6 and self.consecutive_spot_failures < 2:
            self.last_spot_start = elapsed
            return ClusterType.SPOT
        
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)