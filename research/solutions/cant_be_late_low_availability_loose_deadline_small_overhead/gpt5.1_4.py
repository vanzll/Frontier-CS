from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic_v1"

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self._cached_total_slack = None
        self._last_done_time_sum = 0.0
        self._last_done_list_len = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_done_time(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            self._last_done_list_len = 0
            self._last_done_time_sum = 0.0
            return 0.0
        n = len(segments)
        if n == self._last_done_list_len:
            return self._last_done_time_sum
        total = self._last_done_time_sum
        for i in range(self._last_done_list_len, n):
            total += segments[i]
        self._last_done_list_len = n
        self._last_done_time_sum = total
        return total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        deadline = getattr(self, "deadline", 0.0)
        task_duration = getattr(self, "task_duration", 0.0)
        restart_overhead = getattr(self, "restart_overhead", 0.0) or 0.0

        done_time = self._get_done_time()
        remaining_work = max(0.0, task_duration - done_time)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_remaining = max(0.0, deadline - elapsed)

        if time_remaining <= 0.0:
            return ClusterType.ON_DEMAND

        slack = time_remaining - remaining_work

        if self._cached_total_slack is None:
            total_slack = max(0.0, deadline - task_duration)
            self._cached_total_slack = total_slack
        else:
            total_slack = self._cached_total_slack

        idle_margin = restart_overhead + 2.0 * gap

        if total_slack > 0.0:
            target_slack = max(idle_margin * 2.0, 0.2 * total_slack)
        else:
            target_slack = idle_margin * 2.0

        if slack <= idle_margin:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        if slack > target_slack:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)