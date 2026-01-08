import math
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # pragma: no cover
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, args=None):
            self.env = type("Env", (), {"elapsed_seconds": 0, "gap_seconds": 300, "cluster_type": ClusterType.NONE})()
            self.task_duration = 0
            self.task_done_time = []
            self.deadline = 0
            self.restart_overhead = 0


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self._initialized = False
        self._steps = 0
        self._p_ema = 0.6
        self._alpha = 0.06
        self._unavail_streak = 0
        self._avail_streak = 0
        self._hard_buffer = 1800.0
        self._force_od = False
        self._od_streak = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _work_done_seconds(self) -> float:
        t = getattr(self, "task_done_time", None)
        if not t:
            return 0.0
        try:
            first = t[0]
        except Exception:
            return 0.0

        total = 0.0
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            for seg in t:
                try:
                    total += float(seg[1]) - float(seg[0])
                except Exception:
                    continue
            if total > 0:
                return total
            try:
                return float(t[-1][1]) if isinstance(t[-1], (tuple, list)) and len(t[-1]) >= 2 else float(t[-1])
            except Exception:
                return 0.0

        try:
            for x in t:
                total += float(x)
            if total >= 0:
                return total
        except Exception:
            pass

        try:
            return float(t[-1])
        except Exception:
            return 0.0

    def _wait_cap(self, p: float) -> float:
        # Between 10 minutes and 120 minutes, scaled by estimated availability.
        p = _clamp(p, 0.0, 1.0)
        s = _clamp((p - 0.10) / 0.60, 0.0, 1.0)
        return 600.0 + (7200.0 - 600.0) * s

    def _spot_stability_steps(self, p: float) -> int:
        if p >= 0.60:
            return 1
        if p >= 0.35:
            return 2
        return 3

    def _overhead_if_switch_to(self, last: ClusterType, nxt: ClusterType) -> float:
        if nxt == ClusterType.NONE:
            return 0.0
        if last == nxt and last != ClusterType.NONE:
            return 0.0
        return float(getattr(self, "restart_overhead", 0.0) or 0.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            gap = float(getattr(self.env, "gap_seconds", 300.0) or 300.0)
            ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
            self._hard_buffer = max(1200.0, 4.0 * gap, 10.0 * ro)
            self._initialized = True

        self._steps += 1
        gap = float(getattr(self.env, "gap_seconds", 300.0) or 300.0)
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        if has_spot:
            self._avail_streak += 1
            self._unavail_streak = 0
        else:
            self._unavail_streak += 1
            self._avail_streak = 0

        obs = 1.0 if has_spot else 0.0
        self._p_ema = (1.0 - self._alpha) * self._p_ema + self._alpha * obs
        p_pred = _clamp(self._p_ema, 0.01, 0.99)

        work_done = self._work_done_seconds()
        remaining_work = max(0.0, task_duration - work_done)
        time_left = deadline - elapsed

        if remaining_work <= 0.0:
            self._force_od = False
            self._od_streak = 0
            return ClusterType.NONE

        if time_left <= 0.0:
            return ClusterType.NONE

        od_overhead = self._overhead_if_switch_to(last_cluster_type, ClusterType.ON_DEMAND)
        slack_od = time_left - remaining_work - od_overhead

        if remaining_work + od_overhead > time_left:
            self._force_od = True
            self._od_streak += 1
            return ClusterType.ON_DEMAND

        if slack_od <= self._hard_buffer:
            self._force_od = True

        if self._force_od:
            self._od_streak += 1
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND:
            self._od_streak += 1
        else:
            self._od_streak = 0

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if remaining_work <= 3600.0:
                    return ClusterType.ON_DEMAND
                k = self._spot_stability_steps(p_pred)
                if self._avail_streak >= k and slack_od > 2.0 * self._hard_buffer:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        allowed_wait = max(0.0, slack_od - self._hard_buffer)
        allowed_wait = min(allowed_wait, self._wait_cap(p_pred))

        if self._unavail_streak * gap < allowed_wait:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)