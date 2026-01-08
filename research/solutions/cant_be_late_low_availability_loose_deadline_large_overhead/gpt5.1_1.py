from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        self._runtime_initialized = False
        self._force_on_demand = False
        self._progress_done = 0.0
        self._last_task_done_len = 0
        self._buffer_time = None
        self._slack_hard = None
        self._slack_soft = None
        return self

    def _initialize_runtime(self):
        if self._runtime_initialized:
            return
        # Buffer covers one restart overhead and one time step.
        buffer_time = float(getattr(self, "restart_overhead", 0.0)) + float(
            getattr(self.env, "gap_seconds", 0.0)
        )
        if buffer_time <= 0:
            # Fallback if env attributes are missing/unexpected.
            buffer_time = 1.0
        self._buffer_time = buffer_time
        self._slack_hard = buffer_time
        self._slack_soft = 5.0 * buffer_time
        self._runtime_initialized = True

    def _update_progress(self):
        lst = getattr(self, "task_done_time", None)
        if not lst:
            return
        n = len(lst)
        i = self._last_task_done_len
        if i >= n:
            return
        total_add = 0.0
        for seg in lst[i:n]:
            dur = 0.0
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    try:
                        s = float(seg[0])
                        e = float(seg[1])
                    except (TypeError, ValueError):
                        continue
                    if e > s:
                        dur = e - s
            else:
                try:
                    val = float(seg)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    dur = val
            if dur > 0:
                total_add += dur
        if total_add > 0:
            self._progress_done += total_add
            if self._progress_done < 0:
                self._progress_done = 0.0
        self._last_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_runtime()
        self._update_progress()

        time_left = float(self.deadline) - float(self.env.elapsed_seconds)
        if time_left <= 0:
            return ClusterType.ON_DEMAND

        buffer_time = self._buffer_time

        # Hard safety: ensure we can always complete the full task on OD
        # even if we had made zero progress so far.
        if not self._force_on_demand:
            if time_left <= float(self.task_duration) + buffer_time:
                self._force_on_demand = True

        # Additional (earlier) switch based on estimated slack using observed progress.
        if not self._force_on_demand:
            remaining_est = float(self.task_duration) - float(self._progress_done)
            if remaining_est < 0.0:
                remaining_est = 0.0
            slack_est = time_left - remaining_est
            if slack_est <= self._slack_hard:
                self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit behavior: decide how aggressively to chase Spot.
        remaining_est = float(self.task_duration) - float(self._progress_done)
        if remaining_est < 0.0:
            remaining_est = 0.0
        slack_est = time_left - remaining_est

        if slack_est > self._slack_soft:
            # High slack: only use Spot, idle when unavailable.
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE
        else:
            # Moderate slack: use Spot when available, otherwise fall back to OD.
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)