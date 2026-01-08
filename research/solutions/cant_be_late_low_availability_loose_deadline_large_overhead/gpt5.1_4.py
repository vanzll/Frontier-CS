from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.args = args
        # Episode-specific state
        self.force_on_demand = False
        self.commit_margin = None
        self.hybrid_margin = None
        self._progress_seconds = 0.0
        self._cached_task_done_len = 0
        self._last_elapsed = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    # --- Internal helpers ---

    def _ensure_episode_state(self):
        """Reset internal state when a new episode starts."""
        current_t = getattr(getattr(self, "env", None), "elapsed_seconds", 0.0)
        if self._last_elapsed is None or current_t < self._last_elapsed:
            # New episode (or first call)
            self.force_on_demand = False
            self.commit_margin = None
            self.hybrid_margin = None
            self._progress_seconds = 0.0
            self._cached_task_done_len = 0
        self._last_elapsed = current_t

    def _update_progress(self) -> float:
        """Incrementally compute lower-bound estimate of completed work (seconds)."""
        td = getattr(self, "task_done_time", None)
        if td is None:
            return self._progress_seconds

        try:
            n = len(td)
        except TypeError:
            # Not a list-like; give up and keep current estimate (safe lower bound)
            return self._progress_seconds

        start_idx = self._cached_task_done_len
        if n <= start_idx:
            return self._progress_seconds

        for i in range(start_idx, n):
            seg = td[i]
            delta = 0.0
            try:
                if isinstance(seg, (int, float)):
                    delta = float(seg)
                elif isinstance(seg, dict):
                    if "duration" in seg:
                        delta = float(seg["duration"])
                    elif "start" in seg and "end" in seg:
                        delta = float(seg["end"]) - float(seg["start"])
                else:
                    # Treat as (start, end) or similar sequence
                    if hasattr(seg, "__len__") and len(seg) >= 2:
                        a = seg[0]
                        b = seg[1]
                        delta = float(b) - float(a)
            except Exception:
                delta = 0.0

            if delta > 0:
                self._progress_seconds += delta

        self._cached_task_done_len = n
        return self._progress_seconds

    def _ensure_margins_initialized(self):
        """Initialize commit and hybrid margins based on task/deadline parameters."""
        if self.commit_margin is not None and self.hybrid_margin is not None:
            return

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        base_slack = max(deadline - duration, 0.0)

        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        gap_seconds = float(getattr(getattr(self, "env", None), "gap_seconds", 0.0) or 0.0)

        if base_slack <= 0.0:
            # Essentially no slack: be very conservative, almost always on-demand.
            self.commit_margin = 0.0
            self.hybrid_margin = 0.0
            return

        # Base margin: 10% of slack
        margin = 0.1 * base_slack

        # Ensure margin comfortably exceeds restart overhead and time-step granularity
        if restart_overhead > 0.0:
            margin = max(margin, 3.0 * restart_overhead)
        if gap_seconds > 0.0:
            margin = max(margin, 5.0 * gap_seconds + 2.0 * restart_overhead)

        # At least 30 minutes
        margin = max(margin, 0.5 * 3600.0)

        # Do not commit too early: margin at most half the slack and at most 4 hours
        margin = min(margin, 0.5 * base_slack)
        margin = min(margin, 4.0 * 3600.0)

        self.commit_margin = margin

        # Hybrid region where we start using OD during spot gaps, but still prefer spot
        hybrid = self.commit_margin * 3.0
        hybrid = min(hybrid, 0.8 * base_slack)
        # Ensure hybrid > commit by at least one step / overhead / 10 minutes
        hybrid = max(
            hybrid,
            self.commit_margin + max(gap_seconds, restart_overhead, 600.0),
        )
        self.hybrid_margin = hybrid

    # --- Main decision function ---

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_episode_state()
        self._ensure_margins_initialized()

        # Update progress estimate
        progress = self._update_progress()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        remaining_work = max(task_duration - progress, 0.0)

        # If task is already finished, do nothing
        if remaining_work <= 0.0:
            self.force_on_demand = False
            return ClusterType.NONE

        elapsed = float(getattr(getattr(self, "env", None), "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = deadline - elapsed

        # If we've run out of time, just use on-demand (we may already be too late)
        if time_left <= 0.0:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        slack = time_left - remaining_work

        # If slack is non-positive, we must rush with on-demand.
        if slack <= 0.0:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Commit-to-OD region: ensure hard deadline safety.
        if self.force_on_demand or slack <= self.commit_margin:
            self.force_on_demand = True
            return ClusterType.ON_DEMAND

        # Hybrid region: use spot when available, otherwise on-demand to maintain progress.
        if slack <= self.hybrid_margin:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # High-slack region: aggressively use spot; wait (NONE) when spot unavailable.
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE