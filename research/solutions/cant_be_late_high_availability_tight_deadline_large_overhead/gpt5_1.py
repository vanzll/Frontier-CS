import math
from typing import Any, List, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # Fallbacks for environments without the package
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, *args, **kwargs):
            self.env = type("Env", (), {})()
            self.env.elapsed_seconds = 0.0
            self.env.gap_seconds = 60.0
            self.env.cluster_type = ClusterType.NONE
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


class Solution(Strategy):
    NAME = "jit_od_slack_guard_v3"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._od_lock = False
        self._initialized = False
        self._commit_slack = None  # seconds
        self._args = args

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _work_done_seconds(self) -> float:
        total = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = 0.0
        segs = getattr(self, "task_done_time", None)
        if segs is None:
            # Fallback to any alternative attributes if present
            for name in (
                "work_seconds_done",
                "compute_seconds_done",
                "progress_seconds",
                "elapsed_compute_seconds",
                "work_done",
            ):
                v = getattr(self, name, None)
                if isinstance(v, (int, float)):
                    done = float(v)
                    break
            return max(0.0, min(total, done))
        try:
            for seg in segs:
                if seg is None:
                    continue
                if isinstance(seg, (int, float)):
                    done += float(seg)
                    continue
                d = None
                # Try object with duration attribute or method
                if hasattr(seg, "duration"):
                    try:
                        val = seg.duration
                        d = val() if callable(val) else val
                    except Exception:
                        d = None
                # Try (start, end) style with attributes
                if d is None and hasattr(seg, "start") and hasattr(seg, "end"):
                    try:
                        d = float(seg.end) - float(seg.start)
                    except Exception:
                        d = None
                # Try tuple/list of two numbers
                if d is None and isinstance(seg, (tuple, list)) and len(seg) >= 2:
                    s0, s1 = seg[0], seg[1]
                    if isinstance(s0, (int, float)) and isinstance(s1, (int, float)):
                        d = float(s1) - float(s0)
                if d is not None:
                    done += max(0.0, float(d))
        except Exception:
            # If anything goes wrong, be conservative
            done = 0.0
        return max(0.0, min(total, done))

    def _remaining_work_seconds(self) -> float:
        total = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = self._work_done_seconds()
        return max(0.0, total - done)

    def _ensure_initialized(self):
        if self._initialized:
            return
        h = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 60.0) or 60.0)
        # Commit threshold: must retain enough slack to pay one restart overhead
        # plus a small buffer to account for step discretization.
        self._commit_slack = h + max(gap, 0.0)
        self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()

        # If already locked onto on-demand, never switch back.
        if self._od_lock:
            return ClusterType.ON_DEMAND

        # Compute remaining work and slack
        rem = self._remaining_work_seconds()
        time_left = float(getattr(self, "deadline", 0.0) or 0.0) - float(
            getattr(self.env, "elapsed_seconds", 0.0) or 0.0
        )
        time_left = max(0.0, time_left)
        slack = time_left - rem  # seconds

        # If slack is at or below the commit threshold, switch to (and lock) on-demand.
        if slack <= (self._commit_slack or 0.0):
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # Otherwise, prefer Spot when available; pause when not available to save cost.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable, we can afford to wait while slack > commit threshold.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)