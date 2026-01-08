from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_procrastinate_spot_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._commit_to_od = False
        self._safety_margin_seconds = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work(self) -> float:
        try:
            done = sum(self.task_done_time) if self.task_done_time else 0.0
        except Exception:
            done = 0.0
        remaining = self.task_duration - done
        return remaining if remaining > 0 else 0.0

    def _should_commit_to_od(self, last_cluster_type: ClusterType) -> bool:
        # Initialize safety margin on first call when env is available
        if self._safety_margin_seconds is None:
            gap = getattr(self.env, "gap_seconds", 60.0) or 60.0
            # One full step margin to handle discretization and immediate decisions near boundaries
            self._safety_margin_seconds = float(gap)

        remaining = self._remaining_work()
        if remaining <= 0:
            return False

        time_left = self.deadline - self.env.elapsed_seconds
        if time_left <= 0:
            return True

        # If we are already on OD, switching overhead is zero to continue OD
        overhead_needed = 0.0 if last_cluster_type == ClusterType.ON_DEMAND or self._commit_to_od else float(self.restart_overhead)

        need_time = remaining + overhead_needed + self._safety_margin_seconds
        return time_left <= need_time

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If task already done, idle
        if self._remaining_work() <= 0:
            return ClusterType.NONE

        # Decide whether to commit to On-Demand to guarantee finish
        if not self._commit_to_od and self._should_commit_to_od(last_cluster_type):
            self._commit_to_od = True

        # Once committed, always use On-Demand
        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        # Prefer Spot when available; otherwise wait (NONE) until last safe moment
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)