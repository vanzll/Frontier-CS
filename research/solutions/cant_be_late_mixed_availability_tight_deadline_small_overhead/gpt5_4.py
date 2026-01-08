from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "deadline_guard_spot_saver"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._overhead_remaining_seconds = 0.0
        self._epsilon = 1e-9
        self._safety_seconds = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work_seconds(self) -> float:
        done = sum(self.task_done_time) if self.task_done_time else 0.0
        return max(0.0, self.task_duration - done)

    def _time_left_seconds(self) -> float:
        return max(0.0, self.deadline - self.env.elapsed_seconds)

    def _update_overhead_tracking(self, last_cluster_type: ClusterType, has_spot: bool):
        # Consume overhead if we ran compute last step
        if last_cluster_type in (ClusterType.SPOT, ClusterType.ON_DEMAND):
            self._overhead_remaining_seconds = max(
                0.0, self._overhead_remaining_seconds - self.env.gap_seconds
            )

        # If we were on SPOT and it becomes unavailable now, a new restart overhead is due
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self._overhead_remaining_seconds = float(self.restart_overhead)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update model of pending restart overhead based on last step and current availability
        self._update_overhead_tracking(last_cluster_type, has_spot)

        remaining = self._remaining_work_seconds()
        if remaining <= self._epsilon:
            return ClusterType.NONE

        time_left = self._time_left_seconds()
        gap = self.env.gap_seconds

        # If there is not even enough time to finish with immediate OD (including any pending overhead),
        # we must run OD now to minimize lateness risk.
        if time_left < remaining + self._overhead_remaining_seconds - self._epsilon:
            return ClusterType.ON_DEMAND

        if has_spot:
            # Safe-to-use-spot check:
            # After spending one step (gap) possibly with zero net progress,
            # we must still be able to finish on OD with at most one fresh restart overhead.
            can_afford_one_more_step_on_spot = (
                time_left - gap
                >= remaining + float(self.restart_overhead) + self._safety_seconds - self._epsilon
            )
            if can_afford_one_more_step_on_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND
        else:
            # Spot not available: either wait (NONE) if safe, or switch to OD.
            # If we wait one step, overhead to resume remains whatever is pending now.
            can_wait_one_step = (
                time_left - gap
                >= remaining + self._overhead_remaining_seconds + self._safety_seconds - self._epsilon
            )
            if can_wait_one_step:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        parser.add_argument("--safety-seconds", type=float, default=0.0)
        args, _ = parser.parse_known_args()
        obj = cls(args)
        if hasattr(args, "safety_seconds") and args.safety_seconds is not None:
            obj._safety_seconds = float(args.safety_seconds)
        return obj