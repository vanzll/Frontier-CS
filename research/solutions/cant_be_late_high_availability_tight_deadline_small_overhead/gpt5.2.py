import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._initialized = False
        self._last_elapsed = -1.0
        self._p_hat = 0.65
        self._alpha = 0.02
        self._last_has_spot: Optional[bool] = None
        self._consec_no_spot = 0
        self._switched_to_od = False

    def solve(self, spec_path: str) -> "Solution":
        self._reset_episode()
        self._initialized = True
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _reset_episode(self) -> None:
        self._last_elapsed = -1.0
        self._p_hat = 0.65
        self._alpha = 0.02
        self._last_has_spot = None
        self._consec_no_spot = 0
        self._switched_to_od = False

    def _get_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            try:
                return float(td)
            except Exception:
                return 0.0
        if isinstance(td, (list, tuple)):
            if not td:
                return 0.0
            first = td[0]
            try:
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    s = 0.0
                    for seg in td:
                        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                            continue
                        a = float(seg[0])
                        b = float(seg[1])
                        if b >= a:
                            s += (b - a)
                        else:
                            s += a
                    return max(0.0, s)
                vals = [float(x) for x in td]
                if len(vals) >= 2 and all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)):
                    return max(0.0, vals[-1])
                return max(0.0, sum(vals))
            except Exception:
                return 0.0
        return 0.0

    def _compute_buffer_seconds(self, gap: float) -> float:
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        p = min(0.98, max(0.05, float(self._p_hat)))

        base = 1800.0
        availability_term = (1.0 - p) * 7200.0
        overhead_term = min(1800.0, 6.0 * ro)
        streak_term = min(3600.0, 0.25 * float(self._consec_no_spot) * float(gap))

        buf = base + availability_term + overhead_term + streak_term
        buf = max(2700.0, min(10800.0, buf))
        return buf

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if self._last_elapsed >= 0.0 and (elapsed < self._last_elapsed - 1e-6 or (elapsed <= 1e-9 and self._last_elapsed > 1e-6)):
            self._reset_episode()
        self._last_elapsed = elapsed

        if self._last_has_spot is None:
            self._p_hat = 1.0 if has_spot else 0.0
        else:
            x = 1.0 if has_spot else 0.0
            self._p_hat = (1.0 - self._alpha) * self._p_hat + self._alpha * x

        if has_spot:
            self._consec_no_spot = 0
        else:
            self._consec_no_spot += 1
        self._last_has_spot = has_spot

        if self._switched_to_od:
            return ClusterType.ON_DEMAND

        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        work_left = max(0.0, task_duration - done)

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = max(0.0, deadline - elapsed)

        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        od_overhead_if_switch = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro
        min_time_if_od = od_overhead_if_switch + work_left
        slack_det = time_left - min_time_if_od

        if work_left <= 1e-6:
            return ClusterType.NONE

        if slack_det <= 0.0:
            self._switched_to_od = True
            return ClusterType.ON_DEMAND

        buffer = self._compute_buffer_seconds(gap)
        switch_guard = buffer + (1.25 * gap if gap > 0.0 else 0.0)

        if slack_det <= switch_guard:
            self._switched_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        if slack_det <= buffer + (gap if gap > 0.0 else 0.0):
            self._switched_to_od = True
            return ClusterType.ON_DEMAND

        return ClusterType.NONE