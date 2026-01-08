from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "safe_wait_od_switch_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._od_lock = False
        self._initialized = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _reset_if_new_run(self):
        # Heuristic reset at the start of each new trace/run
        if not self._initialized or (self.env.elapsed_seconds <= self.env.gap_seconds * 1.5 and len(self.task_done_time) == 0):
            self._od_lock = False
            self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._reset_if_new_run()

        # Compute remaining work and time
        done = 0.0
        if self.task_done_time:
            # Sum of completed work segments
            done = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - done)
        if remaining_work <= 1e-9:
            self._od_lock = False
            return ClusterType.NONE

        time_remaining = max(0.0, self.deadline - self.env.elapsed_seconds)

        # If no time left, must use on-demand
        if time_remaining <= 0.0:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Safety margin to account for discretization and overhead uncertainties
        gap = float(self.env.gap_seconds)
        overhead = float(self.restart_overhead)
        safety_delta = gap + overhead  # conservative 1-step + overhead buffer

        # Overhead if starting OD now
        od_start_overhead_now = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead

        # Slack if we switch to OD immediately
        slack_now = time_remaining - (remaining_work + od_start_overhead_now)

        # If previously locked to on-demand, keep it
        if self._od_lock:
            return ClusterType.ON_DEMAND

        # If we are close to deadline, switch to OD to guarantee finish
        if slack_now <= safety_delta:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # We have enough slack to prefer Spot if available
        if has_spot:
            # Optional hysteresis: if we are currently on OD (but not locked) only switch back to SPOT
            # when we have ample slack to avoid thrashing.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack_now > 3 * safety_delta:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot not available: decide to wait or use OD
        # Can we safely wait one step and still be able to switch to OD afterwards?
        # After waiting one gap (no progress), we will need overhead to start OD.
        slack_after_wait = (time_remaining - gap) - (remaining_work + overhead)
        if slack_after_wait > safety_delta:
            return ClusterType.NONE

        # Otherwise, use OD to maintain deadline feasibility
        self._od_lock = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)