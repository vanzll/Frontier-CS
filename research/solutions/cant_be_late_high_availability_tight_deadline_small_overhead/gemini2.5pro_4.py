import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy's parameters and state.
        """
        # A conservative estimate of spot instance efficiency, accounting for
        # both unavailability and preemption overhead. An efficiency of 0.58
        # means we expect to get 0.58 seconds of work done for every 1 second
        # of wall-clock time on a spot-first strategy.
        self.SPOT_EFFICIENCY_ESTIMATE = 0.58

        # A safety buffer (in seconds) for our spot-first time estimate. If
        # the estimated time to completion + this margin exceeds the available
        # time, we switch to On-Demand. Set to 3x restart overhead.
        self.SAFETY_MARGIN = 540.0

        # A threshold (in seconds) for our time buffer. If the slack for a
        # spot-first strategy drops below this, we stop waiting for spot
        # and use On-Demand to make progress. Set to 30 minutes.
        self.WAIT_THRESHOLD = 1800.0

        # State flag to lock into On-Demand mode when deadline is close.
        self._final_on_demand_mode = False
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        if self._final_on_demand_mode:
            return ClusterType.ON_DEMAND

        work_done = sum(end - start for start, end in self.task_done_time)
        remaining_work = self.task_duration - work_done

        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # --- Point of No Return Checks ---
        # These checks determine if we must switch to On-Demand permanently.

        # 1. Hard Physical Limit Check: Not enough time left even for OD.
        # We add one restart_overhead as a buffer against a recent preemption.
        min_time_needed_for_od = remaining_work + self.restart_overhead
        if time_to_deadline <= min_time_needed_for_od:
            self._final_on_demand_mode = True
            return ClusterType.ON_DEMAND

        # 2. Soft Probabilistic Limit Check: Spot-first strategy is too risky.
        estimated_time_needed_with_spot = remaining_work / self.SPOT_EFFICIENCY_ESTIMATE
        if time_to_deadline <= estimated_time_needed_with_spot + self.SAFETY_MARGIN:
            self._final_on_demand_mode = True
            return ClusterType.ON_DEMAND

        # --- Default Strategy: Spot-Priority ---
        if has_spot:
            return ClusterType.SPOT
        else:
            # Spot is unavailable. Decide whether to wait or use On-Demand.
            current_spot_strategy_slack = time_to_deadline - estimated_time_needed_with_spot
            
            if current_spot_strategy_slack < self.WAIT_THRESHOLD:
                # Slack is low, use On-Demand to make progress.
                return ClusterType.ON_DEMAND
            else:
                # Slack is sufficient, wait for Spot.
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)