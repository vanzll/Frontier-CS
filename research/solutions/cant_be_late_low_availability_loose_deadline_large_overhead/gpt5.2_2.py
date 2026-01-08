import json
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # Fallbacks for non-eval environments
    from enum import Enum

    class ClusterType(Enum):
        SPOT = 1
        ON_DEMAND = 2
        NONE = 3

    class Strategy:
        def __init__(self, *args, **kwargs):
            self.env = None


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._od_lock = False
        self._last_done: Optional[float] = None
        self._last_elapsed: Optional[float] = None

    def solve(self, spec_path: str) -> "Solution":
        # Optional: read spec for tuning if present
        try:
            with open(spec_path, "r") as f:
                spec = json.load(f)
            _ = spec
        except Exception:
            pass
        self._od_lock = False
        self._last_done = None
        self._last_elapsed = None
        return self

    @staticmethod
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and x == x  # exclude NaN

    def _get_done_work_seconds(self) -> Optional[float]:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return None

        if self._is_num(tdt):
            return float(tdt)

        if not isinstance(tdt, (list, tuple)):
            return None

        if len(tdt) == 0:
            return 0.0

        last = tdt[-1]

        # Case 1: list of numbers (either cumulative or per-segment)
        if self._is_num(last):
            nondec = True
            prev = -1e30
            s = 0.0
            cnt = 0
            for x in tdt:
                if not self._is_num(x):
                    nondec = False
                    break
                fx = float(x)
                if fx < prev - 1e-6:
                    nondec = False
                    break
                prev = fx
                s += fx
                cnt += 1
            if nondec:
                return float(last)
            return s if cnt > 0 else None

        # Case 2: list of segments (start, end)
        if isinstance(last, (list, tuple)) and len(last) >= 2 and self._is_num(last[0]) and self._is_num(last[1]):
            total = 0.0
            any_seg = False
            for seg in tdt:
                if (
                    isinstance(seg, (list, tuple))
                    and len(seg) >= 2
                    and self._is_num(seg[0])
                    and self._is_num(seg[1])
                ):
                    a = float(seg[0])
                    b = float(seg[1])
                    if b > a:
                        total += (b - a)
                    any_seg = True
            return total if any_seg else None

        return None

    def _safety_buffer_seconds(self) -> float:
        gap = 0.0
        try:
            gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        except Exception:
            gap = 0.0

        ro = 0.0
        try:
            ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        except Exception:
            ro = 0.0

        # Conservative buffer to handle step discretization and restart overhead uncertainty.
        return max(2.0 * gap, 0.5 * ro, 600.0)

    def _must_use_on_demand(self, time_remaining: float, remaining_work: float, last_cluster_type: ClusterType) -> bool:
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        od_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro
        buf = self._safety_buffer_seconds()
        return time_remaining <= (remaining_work + od_overhead + buf)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Basic guards
        try:
            elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        except Exception:
            elapsed = 0.0

        deadline = getattr(self, "deadline", None)
        task_duration = getattr(self, "task_duration", None)

        if deadline is None or task_duration is None or not self._is_num(deadline) or not self._is_num(task_duration):
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        deadline = float(deadline)
        task_duration = float(task_duration)

        time_remaining = deadline - elapsed
        if time_remaining <= 0.0:
            return ClusterType.NONE

        done = self._get_done_work_seconds()
        if done is None or not self._is_num(done):
            done = 0.0
        done = float(done)
        if done < 0.0:
            done = 0.0
        if done > task_duration:
            done = task_duration

        remaining_work = task_duration - done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Track observations (optional; may help debugging/extension)
        self._last_done = done
        self._last_elapsed = elapsed

        # If we ever decide we must run OD, lock it in to avoid oscillation.
        if self._od_lock:
            return ClusterType.ON_DEMAND

        if self._must_use_on_demand(time_remaining, remaining_work, last_cluster_type):
            self._od_lock = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # Spot not available and we are not yet in the "must OD" region: wait for free.
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)