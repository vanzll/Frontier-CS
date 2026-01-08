from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_slack_threshold_v1"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state
        self._done_sum = 0.0
        self._last_task_done_len = 0
        self._force_on_demand = False
        self._initialized = True
        return self

    def _ensure_initialized(self):
        # Fallback in case solve() was not called
        if not hasattr(self, "_initialized"):
            self._done_sum = 0.0
            self._last_task_done_len = 0
            self._force_on_demand = False
            self._initialized = True

    def _update_progress(self):
        # Incrementally track total completed work
        task_segments = getattr(self, "task_done_time", None)
        if not task_segments:
            return
        n = len(task_segments)
        if n > self._last_task_done_len:
            # Sum only new segments
            self._done_sum += sum(task_segments[self._last_task_done_len : n])
            self._last_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()
        self._update_progress()

        # Remaining work
        total_duration = getattr(self, "task_duration", 0.0) or 0.0
        done = self._done_sum
        remaining = total_duration - done
        if remaining <= 0.0:
            # Job is finished; no need to run anything
            return ClusterType.NONE

        # If we've already committed to on-demand, always use it
        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Time until deadline and slack without future overhead
        now = getattr(self.env, "elapsed_seconds", 0.0) or 0.0
        deadline = getattr(self, "deadline", 0.0) or 0.0
        available_time = deadline - now

        # Actual slack ignoring future restart overheads
        slack_no_overhead = available_time - remaining

        # Gap and restart overhead (seconds)
        gap = getattr(self.env, "gap_seconds", 0.0) or 0.0
        restart_overhead = getattr(self, "restart_overhead", 0.0) or 0.0

        # Safe-commit threshold: ensure enough time to pay for one restart_overhead
        # plus at most one full gap of idle/overhead before we can react.
        commit_threshold = restart_overhead + gap

        # If slack is small enough that we must secure completion, commit to on-demand.
        if slack_no_overhead <= commit_threshold:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Otherwise, exploit spot whenever available, wait (NONE) otherwise.
        if has_spot:
            return ClusterType.SPOT

        # No spot available and we have ample slack: wait.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)