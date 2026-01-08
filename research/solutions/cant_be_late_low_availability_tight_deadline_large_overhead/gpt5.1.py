from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cb_late_safe_slack"

    def __init__(self, args=None):
        super().__init__(args)
        self._initialized = False
        self._force_on_demand = False
        # For incremental progress computation
        self._prefix_n = 0
        self._prefix_total = 0.0
        self._commit_slack_threshold = 0.0
        self._dt = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_runtime(self):
        self._initialized = True
        self._force_on_demand = False
        self._prefix_n = 0
        self._prefix_total = 0.0

        # Derive initial slack and time step.
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        total_slack = max(deadline - task_duration, 0.0)
        dt = getattr(self.env, "gap_seconds", 60.0)
        self._dt = float(dt) if dt is not None else 60.0

        # Commit to on-demand when slack gets below a threshold.
        # Use half of total slack, but at least a few steps worth.
        base_thr = 0.5 * total_slack
        min_thr = 3.0 * self._dt  # buffer of at least ~3 zero-progress steps
        thr = max(base_thr, min_thr)
        if total_slack > 0.0 and thr > total_slack:
            thr = total_slack
        self._commit_slack_threshold = thr

    def _segment_duration(self, seg) -> float:
        if seg is None:
            return 0.0

        # Numeric segment interpreted as duration
        if isinstance(seg, (int, float)):
            try:
                v = float(seg)
            except Exception:
                return 0.0
            return max(v, 0.0)

        # Dict with start/end
        if isinstance(seg, dict):
            key_pairs = [
                ("start", "end"),
                ("begin", "end"),
                ("t0", "t1"),
                ("s", "e"),
            ]
            for k0, k1 in key_pairs:
                if k0 in seg and k1 in seg:
                    a, b = seg[k0], seg[k1]
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        return max(float(b) - float(a), 0.0)

        # Objects with duration-like attribute
        for attr in ("duration", "dur", "delta", "seconds", "sec", "len", "length"):
            if hasattr(seg, attr):
                val = getattr(seg, attr)
                try:
                    v = float(val() if callable(val) else val)
                    if v >= 0.0:
                        return v
                except Exception:
                    pass

        # Objects with start/end attributes
        for a0, a1 in (("start", "end"), ("begin", "end"), ("t0", "t1")):
            if hasattr(seg, a0) and hasattr(seg, a1):
                v0 = getattr(seg, a0)
                v1 = getattr(seg, a1)
                if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                    return max(float(v1) - float(v0), 0.0)

        # Tuple/list of (start, end)
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            v0, v1 = seg[0], seg[1]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                return max(float(v1) - float(v0), 0.0)

        return 0.0

    def _compute_progress_seconds(self) -> float:
        segments = getattr(self, "task_done_time", None)
        if not segments:
            # Reset if list cleared
            self._prefix_n = 0
            self._prefix_total = 0.0
            return 0.0

        n = len(segments)
        if n <= 0:
            self._prefix_n = 0
            self._prefix_total = 0.0
            return 0.0

        # Ensure prefix_n is valid relative to current list size.
        if self._prefix_n > n - 1:
            self._prefix_n = 0
            self._prefix_total = 0.0

        last_index = n - 1

        # Any segments from current prefix_n up to (but not including) last_index
        # are now "fixed" and can be added once.
        for i in range(self._prefix_n, last_index):
            seg = segments[i]
            self._prefix_total += self._segment_duration(seg)

        self._prefix_n = last_index

        # Last segment may be in-progress; recompute each call.
        last_seg = segments[last_index]
        last_dur = self._segment_duration(last_seg)

        total = self._prefix_total + last_dur

        # Clamp progress to not exceed elapsed wall time to avoid overestimation
        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        try:
            elapsed_f = float(elapsed)
        except Exception:
            elapsed_f = 0.0

        if total > elapsed_f:
            total = elapsed_f

        return max(total, 0.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialize_runtime()

        progress = self._compute_progress_seconds()
        try:
            remaining_work = float(self.task_duration) - progress
        except Exception:
            remaining_work = 0.0
        if remaining_work <= 0.0:
            # Task completed; no need to run more instances
            return ClusterType.NONE

        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        try:
            elapsed_f = float(elapsed)
        except Exception:
            elapsed_f = 0.0

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = elapsed_f

        time_left = deadline - elapsed_f
        if time_left <= 0.0:
            # Deadline reached or passed; nothing sensible to do
            return ClusterType.NONE

        slack = time_left - remaining_work

        # Once slack is small or negative, permanently switch to on-demand.
        if not self._force_on_demand:
            if slack <= self._commit_slack_threshold or slack <= 0.0:
                self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Pre-commit phase: opportunistically use spot when available.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)