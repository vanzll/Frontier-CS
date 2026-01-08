import math
from typing import Any, Iterable

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_safe_threshold_v1"

    def solve(self, spec_path: str) -> "Solution":
        # Optional initialization; can read spec_path if needed.
        # Initialize internal state.
        if not hasattr(self, "_od_lock"):
            self._od_lock = False
        return self

    def _sum_done_seconds(self, segments: Iterable[Any]) -> float:
        done = 0.0
        if not segments:
            return 0.0
        for seg in segments:
            try:
                # Common case: float/int seconds
                if isinstance(seg, (int, float)):
                    done += float(seg)
                    continue
                # Dict with duration or start/end
                if isinstance(seg, dict):
                    if "duration" in seg:
                        done += float(seg["duration"])
                        continue
                    if "start" in seg and "end" in seg:
                        done += float(seg["end"] - seg["start"])
                        continue
                # Tuple/list: assume (start, end)
                if isinstance(seg, (tuple, list)) and len(seg) >= 2:
                    s0, s1 = seg[0], seg[1]
                    if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
                        done += float(s1 - s0)
                        continue
                # Object with attribute duration
                dur = getattr(seg, "duration", None)
                if isinstance(dur, (int, float)):
                    done += float(dur)
                    continue
            except Exception:
                # Ignore malformed entries
                pass
        return max(0.0, done)

    def _remaining_work(self) -> float:
        try:
            total = float(getattr(self, "task_duration", 0.0))
        except Exception:
            total = 0.0
        done = self._sum_done_seconds(getattr(self, "task_done_time", []))
        done = min(done, total)
        remaining = max(0.0, total - done)
        return remaining

    def _reaction_buffer(self) -> float:
        # Reserve time to detect spot loss and spin up OD:
        # - 2 * gap for detection + decision at next step
        # - plus a small constant cushion (60s)
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        cushion = 60.0
        return max(2.0 * gap + cushion, 120.0)  # At least 2 minutes

    def _threshold(self) -> float:
        # Time needed from now to safely finish on OD, accounting for restart overhead and reaction buffer.
        remaining = self._remaining_work()
        overhead = float(getattr(self, "restart_overhead", 0.0))
        buffer_sec = self._reaction_buffer()
        return remaining + overhead + buffer_sec

    def _time_left(self) -> float:
        deadline = float(getattr(self, "deadline", 0.0))
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        return max(0.0, deadline - elapsed)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize lock if needed
        if not hasattr(self, "_od_lock"):
            self._od_lock = False

        # If we've committed to on-demand, stay on it.
        if self._od_lock:
            return ClusterType.ON_DEMAND

        remaining = self._remaining_work()
        if remaining <= 0.0:
            return ClusterType.NONE

        time_left = self._time_left()
        threshold = self._threshold()

        # If we don't have enough slack beyond the threshold, commit to on-demand now.
        if time_left <= threshold:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Otherwise, we still have sufficient slack:
        # - Prefer SPOT if available
        # - If SPOT not available, wait (NONE) to save cost
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)