import numpy as np
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import math
import json

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        try:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
                self._config = spec
        except:
            self._config = {}
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Time constants
        spot_price = 0.97
        od_price = 3.06
        restart_overhead = 180  # seconds (0.05 hours)
        task_duration = 48 * 3600  # 48 hours in seconds
        deadline = 70 * 3600  # 70 hours in seconds
        
        # Compute progress
        completed = sum(end - start for start, end in self.task_done_time)
        remaining = self.task_duration - completed
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Safety margin: time needed if we run entirely on-demand from now
        time_needed_od = remaining
        safety_margin = max(2 * restart_overhead, 3600)  # 1 hour minimum
        
        # If we can't finish even with on-demand from now, go full on-demand
        if time_needed_od > time_left - restart_overhead:
            return ClusterType.ON_DEMAND
        
        # If we're very close to finishing, use on-demand to guarantee completion
        if remaining <= 2 * self.env.gap_seconds:
            return ClusterType.ON_DEMAND
        
        # If spot is not available, either wait or use on-demand based on urgency
        if not has_spot:
            # Compute how much time we can afford to wait
            od_time_needed = remaining
            time_slack = time_left - od_time_needed - safety_margin
            
            # If we have slack, wait for spot
            if time_slack > restart_overhead:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND
        
        # Adaptive strategy based on time pressure
        progress_rate = self.env.gap_seconds
        steps_needed = remaining / progress_rate
        steps_left = time_left / self.env.gap_seconds
        
        # Compute risk factor based on time pressure
        time_pressure = 1.0 - (time_left / (deadline * 0.5))
        time_pressure = max(0.0, min(1.0, time_pressure))
        
        # Dynamic threshold for switching to on-demand
        # As time pressure increases, threshold decreases
        threshold = 0.3 + 0.5 * (1.0 - time_pressure)  # 0.3 to 0.8
        
        # Estimate probability of success with spot
        # This is a heuristic based on remaining time and work
        if steps_needed < steps_left * threshold:
            # We have enough time to risk using spot
            return ClusterType.SPOT
        else:
            # Too risky, use on-demand
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)