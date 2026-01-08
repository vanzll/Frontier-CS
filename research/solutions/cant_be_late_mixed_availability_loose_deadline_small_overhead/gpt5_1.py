from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "deadline_guard_v2"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.committed_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _remaining_work(self):
        try:
            done = sum(self.task_done_time) if self.task_done_time else 0.0
        except Exception:
            done = 0.0
        rem = max(self.task_duration - done, 0.0)
        return rem

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If already committed to OD, stick with it.
        if self.committed_to_od:
            return ClusterType.ON_DEMAND

        gap = getattr(self.env, "gap_seconds", 60.0) or 60.0
        safety_buffer = max(gap, 30.0)

        remaining_work = self._remaining_work()
        if remaining_work <= 0:
            return ClusterType.NONE

        elapsed = getattr(self.env, "elapsed_seconds", 0.0) or 0.0
        deadline = getattr(self, "deadline", elapsed) or elapsed
        time_remaining = deadline - elapsed

        if time_remaining <= 0:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        # Overhead to start OD now (only if we're not already on OD)
        overhead_to_start_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else getattr(self, "restart_overhead", 0.0) or 0.0

        # If we must switch to OD now to guarantee finishing in time, do it.
        if time_remaining <= remaining_work + overhead_to_start_od + safety_buffer:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        # Prefer spot if available and we still have enough runway
        if has_spot:
            return ClusterType.SPOT

        # Spot not available; check if we can safely wait
        overhead_if_start_later = getattr(self, "restart_overhead", 0.0) or 0.0
        wait_budget = time_remaining - (remaining_work + overhead_if_start_later + safety_buffer)

        if wait_budget > 0:
            return ClusterType.NONE

        # No more slack to wait; commit to OD
        self.committed_to_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)