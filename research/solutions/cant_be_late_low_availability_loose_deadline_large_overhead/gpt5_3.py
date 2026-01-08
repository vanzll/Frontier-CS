from typing import Any
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_threshold_v3"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._committed_to_od = False
        self._last_elapsed = -1.0
        self._done_sum = 0.0
        self._done_list_len = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _reset_run_state_if_needed(self):
        # Detect new trace by elapsed time reset
        if self._last_elapsed > self.env.elapsed_seconds:
            self._committed_to_od = False
            self._done_sum = 0.0
            self._done_list_len = 0

    def _update_progress_sum(self):
        # Incrementally sum task_done_time list to avoid O(n) per step
        try:
            lst = self.task_done_time
            n = len(lst)
            if n > self._done_list_len:
                for i in range(self._done_list_len, n):
                    self._done_sum += float(lst[i])
                self._done_list_len = n
        except Exception:
            # Fallback if unexpected structure
            try:
                self._done_sum = float(self.task_duration) - float(self.env.remaining_task_seconds)  # type: ignore
            except Exception:
                pass

    def _remaining_work(self) -> float:
        return max(0.0, float(self.task_duration) - float(self._done_sum))

    def _time_left(self) -> float:
        return max(0.0, float(self.deadline) - float(self.env.elapsed_seconds))

    def _margin_seconds(self) -> float:
        # Safety margin to account for discretization and unseen in-flight progress.
        # Use at least two steps or 5 minutes, whichever is larger.
        step = float(self.env.gap_seconds)
        return max(2.0 * step, 300.0)

    def _overhead_if_switch_to_od_now(self) -> float:
        # If already committed to OD or currently on OD, no additional overhead to continue.
        if self._committed_to_od or self.env.cluster_type == ClusterType.ON_DEMAND:
            return 0.0
        return float(self.restart_overhead)

    def _should_commit_to_od_now(self, work_left: float, time_left: float) -> bool:
        margin = self._margin_seconds()
        overhead = self._overhead_if_switch_to_od_now()
        # Commit if waiting any longer risks missing the deadline even with OD from now.
        return time_left <= (work_left + overhead + margin)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._reset_run_state_if_needed()
        self._update_progress_sum()
        self._last_elapsed = float(self.env.elapsed_seconds)

        # If already done, no need to run anything.
        work_left = self._remaining_work()
        if work_left <= 0.0:
            return ClusterType.NONE

        time_left = self._time_left()

        # If already committed to on-demand, keep running OD until completion.
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Decide whether to commit to OD now
        if self._should_commit_to_od_now(work_left, time_left):
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Not committed yet: prefer SPOT when available; otherwise wait (NONE)
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and not urgent yet -> wait
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)