import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self.initial_p_spot = 0.60
        self.ema_alpha = 0.005
        self.coasting_buffer = 3600.0
        self.safety_margin = None
        self.p_spot_estimate = self.initial_p_spot
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_obs = 1.0 if has_spot else 0.0
        self.p_spot_estimate = (1 - self.ema_alpha) * self.p_spot_estimate + self.ema_alpha * current_obs

        work_rem = self.task_duration - self.work_done
        time_rem = self.deadline - self.env.elapsed_seconds
        gap = self.env.gap_seconds
        
        if self.safety_margin is None:
            self.safety_margin = gap

        if work_rem <= 0:
            return ClusterType.NONE

        if time_rem <= 0:
            return ClusterType.ON_DEMAND

        p_spot = self.p_spot_estimate
        overhead = self.restart_overhead

        p_adjusted = p_spot - (1.0 - p_spot) * overhead / gap
        p_adjusted = max(0.001, p_adjusted)

        time_needed_od = work_rem
        time_needed_cheapest = work_rem / p_adjusted

        if time_needed_od >= time_rem - self.safety_margin:
            return ClusterType.ON_DEMAND
        
        if time_rem - time_needed_cheapest > self.coasting_buffer:
            return ClusterType.NONE
            
        if has_spot:
            return ClusterType.SPOT
        else:
            if time_needed_cheapest >= time_rem - self.safety_margin:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)