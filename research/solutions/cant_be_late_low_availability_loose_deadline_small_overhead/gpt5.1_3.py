from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def solve(self, spec_path: str) -> "Solution":
        self.lock_on_demand = False
        self._prev_elapsed = -1.0
        return self

    def _remaining_work(self) -> float:
        done = 0.0
        try:
            if getattr(self, "task_done_time", None):
                done = float(sum(self.task_done_time))
        except TypeError:
            try:
                done = float(self.task_done_time)
            except Exception:
                done = 0.0
        remaining = float(self.task_duration) - done
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _maybe_reset_episode_state(self):
        current_elapsed = float(self.env.elapsed_seconds)
        if not hasattr(self, "_prev_elapsed"):
            self._prev_elapsed = current_elapsed
            if not hasattr(self, "lock_on_demand"):
                self.lock_on_demand = False
            return
        if current_elapsed < self._prev_elapsed:
            # New episode detected
            self.lock_on_demand = False
        self._prev_elapsed = current_elapsed

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._maybe_reset_episode_state()

        remaining_work = self._remaining_work()
        if remaining_work <= 0.0:
            # Job done or no work: don't spend money.
            self.lock_on_demand = False
            return ClusterType.NONE

        time_left = float(self.deadline) - float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)
        overhead = float(self.restart_overhead)

        if self.lock_on_demand:
            return ClusterType.ON_DEMAND

        # Check if we can afford to delay on-demand start by one more step
        # even in the worst case of zero progress during this step.
        # If not, lock into on-demand now.
        if time_left <= remaining_work + overhead + gap:
            self.lock_on_demand = True
            return ClusterType.ON_DEMAND

        # Safe to keep gambling on spot for this step.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and still plenty of slack: wait (cost 0) instead of using expensive OD.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)