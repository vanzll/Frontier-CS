import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._inited = False
        self._last_has_spot: Optional[bool] = None

        self._steps = 0
        self._spot_steps = 0
        self._observed_seconds = 0.0

        self._trans_avail_to_unavail = 0

        self._p_ewma = 0.65

        self._od_committed = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_work_done(task_done_time: Any) -> float:
        if not task_done_time:
            return 0.0
        total = 0.0
        try:
            for seg in task_done_time:
                if isinstance(seg, (int, float)):
                    total += float(seg)
                elif isinstance(seg, (tuple, list)) and len(seg) >= 2:
                    a, b = seg[0], seg[1]
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        total += float(b) - float(a)
                elif isinstance(seg, dict):
                    a = seg.get("start", None)
                    b = seg.get("end", None)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        total += float(b) - float(a)
        except Exception:
            return 0.0
        return max(0.0, total)

    def _update_stats(self, has_spot: bool) -> None:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if not self._inited:
            self._inited = True
            self._last_has_spot = has_spot

        if self._last_has_spot is True and has_spot is False:
            self._trans_avail_to_unavail += 1

        self._steps += 1
        if has_spot:
            self._spot_steps += 1
        self._observed_seconds += gap

        alpha = 0.03
        self._p_ewma = (1.0 - alpha) * self._p_ewma + alpha * (1.0 if has_spot else 0.0)

        self._last_has_spot = has_spot

    def _expected_avail_streak_seconds(self) -> float:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if self._trans_avail_to_unavail <= 0:
            return 1e9
        avg_steps = self._spot_steps / max(1.0, float(self._trans_avail_to_unavail))
        return max(gap, avg_steps * gap)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_stats(has_spot)

        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        work_done = self._safe_work_done(getattr(self, "task_done_time", None))
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        remaining = max(0.0, task_duration - work_done)
        time_left = max(0.0, deadline - now)

        if remaining <= 0.0:
            return ClusterType.NONE

        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        safety_margin = max(2.0 * ro + 2.0 * gap, 0.5 * 3600.0)

        if remaining + ro + safety_margin >= time_left:
            self._od_committed = True

        if self._od_committed:
            return ClusterType.ON_DEMAND

        p_expected = self._p_ewma - 0.06
        p_expected = min(0.95, max(0.35, p_expected))

        required_rate = remaining / max(1e-6, time_left)
        if required_rate >= 0.985:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        t0 = remaining / max(1e-6, p_expected)
        interrupt_rate = self._trans_avail_to_unavail / max(1.0, self._observed_seconds)
        overhead_est = interrupt_rate * t0 * ro
        t_wait_est = t0 + overhead_est

        feasible_wait = (t_wait_est + safety_margin) <= time_left

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                streak = self._expected_avail_streak_seconds()
                min_run = max(4.0 * ro, 1.5 * 3600.0)
                if (time_left - remaining) <= (2.0 * ro + safety_margin):
                    return ClusterType.ON_DEMAND
                if streak < min_run:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        if feasible_wait:
            return ClusterType.NONE

        if (time_left - remaining) <= (ro + safety_margin):
            self._od_committed = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)