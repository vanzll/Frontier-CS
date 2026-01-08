import sys
import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_threshold"

    def __init__(self, args):
        super().__init__(args)
        self.safety_margin = None
        self.spot_ratio = None
        self.use_spot = True
        self.critical_time = None
        self.last_spot_available = True

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done
        time_left = self.deadline - elapsed
        gap = self.env.gap_seconds

        if self.safety_margin is None:
            self.safety_margin = max(2.0 * 3600, 3 * self.restart_overhead)

        if work_left <= 0:
            return ClusterType.NONE

        if not has_spot:
            self.last_spot_available = False
        else:
            self.last_spot_available = True

        required_rate = work_left / max(time_left, 0.1)
        safe_rate = work_left / max(time_left - self.safety_margin, 0.1)

        if self.critical_time is None:
            self.critical_time = False

        if time_left <= work_left + self.restart_overhead + 3600:
            self.critical_time = True

        if self.critical_time:
            if has_spot and time_left > work_left + self.restart_overhead + 1800:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        if required_rate > 0.95:
            return ClusterType.ON_DEMAND

        if safe_rate > 0.85:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            elif last_cluster_type == ClusterType.ON_DEMAND:
                if time_left > work_left + 3 * self.restart_overhead:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:
                return ClusterType.SPOT
        else:
            if time_left > work_left + self.safety_margin:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)