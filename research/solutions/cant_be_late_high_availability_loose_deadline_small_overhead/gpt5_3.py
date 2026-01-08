from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "lazy_on_demand_fallback_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._commit_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work(self) -> float:
        done = 0.0
        try:
            if self.task_done_time:
                done = float(sum(self.task_done_time))
        except Exception:
            done = 0.0
        remain = float(self.task_duration) - done
        if remain < 0.0:
            remain = 0.0
        return remain

    def _should_commit_to_od(self, time_remaining: float, remaining_work: float) -> bool:
        # Use a small safety margin to account for step discretization.
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        margin = max(0.0, min(gap, 60.0))  # at most 60s margin
        needed = remaining_work + float(self.restart_overhead)
        return time_remaining <= (needed + margin)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already committed to on-demand, stick with it to avoid extra overheads and risk.
        if self._commit_to_od:
            return ClusterType.ON_DEMAND

        # Compute remaining work and time.
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = float("inf")

        remaining_work = self._remaining_work()
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_remaining = deadline - elapsed

        # If we are close to the deadline such that only OD can guarantee completion, commit now.
        if self._should_commit_to_od(time_remaining, remaining_work):
            self._commit_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, prefer spot when available; if not, wait to maximize spot usage.
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)