from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cb_late_balanced_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self.lock_on_demand = False
        self._work_done = 0.0
        self._processed_segments = 0

    def solve(self, spec_path: str) -> "Solution":
        # No offline preprocessing needed; spec_path ignored.
        return self

    def _segment_duration(self, seg) -> float:
        if seg is None:
            return 0.0

        # Objects with start/end attributes (e.g., namedtuple, dataclass)
        if hasattr(seg, "start") and hasattr(seg, "end"):
            try:
                start = float(seg.start)
                end = float(seg.end)
                if end > start:
                    return end - start
            except (TypeError, ValueError):
                return 0.0

        # Tuples/lists: try (start, end), else duration in first element.
        if isinstance(seg, (tuple, list)):
            if len(seg) >= 2:
                try:
                    start = float(seg[0])
                    end = float(seg[1])
                    if end > start:
                        return end - start
                except (TypeError, ValueError):
                    pass
            try:
                val = float(seg[0])
                if val > 0.0:
                    return val
            except (TypeError, ValueError, IndexError):
                return 0.0
            return 0.0

        # Dicts: try keys 'start'/'end' or 'duration'
        if isinstance(seg, dict):
            if "start" in seg and "end" in seg:
                try:
                    start = float(seg["start"])
                    end = float(seg["end"])
                    if end > start:
                        return end - start
                except (TypeError, ValueError):
                    return 0.0
            if "duration" in seg:
                try:
                    val = float(seg["duration"])
                    if val > 0.0:
                        return val
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

        # Fallback: treat as scalar duration
        try:
            val = float(seg)
            if val > 0.0:
                return val
        except (TypeError, ValueError):
            return 0.0
        return 0.0

    def _clamp_work_done(self):
        env = getattr(self, "env", None)
        if env is not None:
            elapsed = getattr(env, "elapsed_seconds", None)
            if elapsed is not None and self._work_done > elapsed:
                self._work_done = float(elapsed)
        task_duration = getattr(self, "task_duration", None)
        if task_duration is not None and self._work_done > task_duration:
            self._work_done = float(task_duration)

    def _update_work_done_cache(self):
        segments = getattr(self, "task_done_time", None)
        if not segments:
            return

        try:
            n = len(segments)
        except TypeError:
            # Non-sequence single segment
            if self._processed_segments == 0:
                self._work_done += self._segment_duration(segments)
                self._processed_segments = 1
                self._clamp_work_done()
            return

        if self._processed_segments > n:
            # List shrank unexpectedly; recompute from scratch.
            self._work_done = 0.0
            self._processed_segments = 0

        if self._processed_segments < n:
            for idx in range(self._processed_segments, n):
                self._work_done += self._segment_duration(segments[idx])
            self._processed_segments = n
            self._clamp_work_done()

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress.
        self._update_work_done_cache()

        task_duration = getattr(self, "task_duration", 0.0)
        if self._work_done >= task_duration:
            # Job already completed; avoid extra cost.
            return ClusterType.NONE

        env = getattr(self, "env", None)
        if env is None:
            # Fallback: if environment not ready, be idle or use spot if allowed.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

        elapsed = getattr(env, "elapsed_seconds", 0.0)
        gap = getattr(env, "gap_seconds", 0.0)
        deadline = getattr(self, "deadline", float("inf"))
        restart_overhead = getattr(self, "restart_overhead", 0.0)

        time_left = deadline - elapsed
        if time_left <= 0.0:
            # Already at or past deadline; minimize cost.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

        remaining = task_duration - self._work_done
        if remaining < 0.0:
            remaining = 0.0

        slack = time_left - remaining

        if slack < 0.0:
            # Impossible to finish regardless; run cheaply.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

        # If we've already committed to on-demand, stay there.
        if self.lock_on_demand:
            return ClusterType.ON_DEMAND

        total_slack = deadline - task_duration
        if total_slack <= 0.0:
            # No slack: safest is to stay on-demand the whole time.
            self.lock_on_demand = True
            return ClusterType.ON_DEMAND

        # Margin before we permanently switch to on-demand.
        # Use half of total slack, but at least several restart overheads.
        base_margin = 0.5 * total_slack
        overhead_margin = 3.0 * restart_overhead
        time_margin = 2.0 * gap
        switch_slack = max(base_margin, overhead_margin, time_margin)

        if slack <= switch_slack:
            # We are close to the deadline; consider switching to OD if it doesn't
            # immediately push us past the deadline due to restart overhead.
            if slack >= restart_overhead:
                self.lock_on_demand = True
                return ClusterType.ON_DEMAND
            # If slack is smaller than restart_overhead, switching clusters now
            # could make us worse off; continue current behavior instead.
            # Prefer to keep current stable cluster if possible.
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.SPOT and has_spot:
                return ClusterType.SPOT
            # Otherwise fall through to cheap choice.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

        # We are in the "safe to gamble" region.
        if has_spot:
            return ClusterType.SPOT

        # Spot currently unavailable: decide between waiting (NONE) and switching to OD.
        slack_after_wait = slack - gap
        if slack_after_wait > switch_slack:
            # We can afford to wait for spot to return.
            return ClusterType.NONE

        # Not safe to keep waiting: switch to OD and lock.
        if slack >= restart_overhead:
            self.lock_on_demand = True
            return ClusterType.ON_DEMAND

        # As above, if slack is too small for a safe restart, avoid switching.
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND
        if last_cluster_type == ClusterType.SPOT and has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)