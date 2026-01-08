from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_safe_wait_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self.committed_to_od = False
        self._sum_done = 0.0
        self._last_done_len = 0
        self._safety_margin = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_work_done(self):
        l = len(self.task_done_time)
        if l > self._last_done_len:
            # Incrementally sum newly added segments
            self._sum_done += sum(self.task_done_time[self._last_done_len:])
            self._last_done_len = l

    def _remaining_work(self) -> float:
        self._update_work_done()
        rem = self.task_duration - self._sum_done
        return rem if rem > 0.0 else 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.committed_to_od:
            return ClusterType.ON_DEMAND

        # Remaining time and work in seconds
        time_remaining = self.deadline - self.env.elapsed_seconds
        if time_remaining < 0.0:
            time_remaining = 0.0
        work_remaining = self._remaining_work()
        gap = float(self.env.gap_seconds)
        if gap <= 0.0:
            gap = 1.0  # Fallback to avoid division-by-zero or logic issues

        if has_spot:
            # Use spot when available until we must commit to OD.
            return ClusterType.SPOT

        # Spot unavailable
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Already on OD; continue and finalize.
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        # Decide to wait (NONE) or commit to OD.
        # It's safe to wait one step if after waiting we could still start OD and finish:
        # (time_remaining - gap - safety_margin) >= (work_remaining + restart_overhead)
        # Use restart_overhead for starting OD from SPOT/NONE.
        need_overhead = self.restart_overhead
        can_wait = (time_remaining - gap - self._safety_margin) >= (work_remaining + need_overhead)

        if can_wait:
            return ClusterType.NONE

        # Commit to On-Demand to guarantee completion
        self.committed_to_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)