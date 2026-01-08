from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "deadline_safe_fallback"

    def solve(self, spec_path: str) -> "Solution":
        # Initialize internal state
        self._work_done = 0.0
        self._last_segment_index = 0
        self._lock_od = False
        return self

    def _segment_duration(self, seg) -> float:
        """Best-effort extraction of a segment duration in seconds."""
        try:
            # Simple numeric duration
            if isinstance(seg, (int, float)):
                return float(seg)

            # Dict with common keys
            if isinstance(seg, dict):
                if "duration" in seg:
                    return float(seg["duration"])
                if "end" in seg and "start" in seg:
                    return float(seg["end"]) - float(seg["start"])

            # Tuple/list: (start, end) or (duration,)
            if isinstance(seg, (list, tuple)):
                if len(seg) >= 2:
                    return float(seg[1]) - float(seg[0])
                elif len(seg) == 1:
                    return float(seg[0])

            # Generic attributes
            if hasattr(seg, "duration"):
                return float(seg.duration)
            if hasattr(seg, "end") and hasattr(seg, "start"):
                return float(seg.end - seg.start)
        except Exception:
            pass
        return 0.0

    def _update_progress(self) -> float:
        """Incrementally update cached work done from self.task_done_time."""
        # Ensure attributes exist if solve() wasn't called
        if not hasattr(self, "_work_done"):
            self._work_done = 0.0
            self._last_segment_index = 0
            self._lock_od = False

        segments = getattr(self, "task_done_time", None)
        if not segments:
            return self._work_done

        n = len(segments)
        # Process only new segments
        for i in range(self._last_segment_index, n):
            dur = self._segment_duration(segments[i])
            if dur > 0:
                self._work_done += dur
        self._last_segment_index = n
        return self._work_done

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Ensure state initialized even if solve() wasn't called
        if not hasattr(self, "_work_done"):
            self._work_done = 0.0
            self._last_segment_index = 0
            self._lock_od = False

        # Update progress from environment
        work_done = self._update_progress()
        task_duration = getattr(self, "task_duration", 0.0)
        deadline = getattr(self, "deadline", float("inf"))
        restart_overhead = getattr(self, "restart_overhead", 0.0)

        # If task is already complete, avoid extra cost
        if work_done >= task_duration:
            return ClusterType.NONE

        elapsed = getattr(self.env, "elapsed_seconds", 0.0)
        gap = getattr(self.env, "gap_seconds", 0.0)
        time_left = deadline - elapsed

        if time_left <= 0:
            # Already at/past deadline; nothing sensible to do
            return ClusterType.NONE

        remaining_work = max(task_duration - work_done, 0.0)
        slack = max(deadline - task_duration, 0.0)

        # Safety margin: account for restart overhead + discretization.
        desired_margin = restart_overhead + 2.0 * gap
        if slack > 0:
            margin = min(desired_margin, 0.9 * slack)
        else:
            margin = desired_margin

        # Once we decide to lock into on-demand, never go back to spot
        if getattr(self, "_lock_od", False):
            return ClusterType.ON_DEMAND

        # Check if we must switch to on-demand now to safely finish by deadline
        if time_left <= remaining_work + margin:
            self._lock_od = True
            return ClusterType.ON_DEMAND

        # Safe zone: use spot when available, otherwise pause
        if has_spot:
            return ClusterType.SPOT

        # No spot, plenty of slack left: wait to save on-demand cost
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)