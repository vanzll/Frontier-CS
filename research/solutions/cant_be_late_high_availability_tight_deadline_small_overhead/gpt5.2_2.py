import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._inited = False

        self._ema_avail = 0.65
        self._ema_drop = 0.02  # P(has_spot(t-1)=1 and has_spot(t)=0) per step
        self._last_has_spot: Optional[bool] = None
        self._steps = 0

        self._od_committed = False

    def solve(self, spec_path: str) -> "Solution":
        self._inited = True
        return self

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0
        if isinstance(tdt, (int, float)):
            v = float(tdt)
            return v if v > 0.0 else 0.0
        if not isinstance(tdt, list):
            return 0.0

        if not tdt:
            return 0.0

        # If it's a cumulative list of numbers, take the last.
        if all(isinstance(x, (int, float)) for x in tdt):
            try:
                if len(tdt) >= 2 and all(float(tdt[i]) >= float(tdt[i - 1]) for i in range(1, len(tdt))):
                    v = float(tdt[-1])
                    return v if v > 0.0 else 0.0
            except Exception:
                pass

        total = 0.0
        for x in tdt:
            if x is None:
                continue
            if isinstance(x, (int, float)):
                vx = float(x)
                if vx > 0.0:
                    total += vx
            elif isinstance(x, (tuple, list)) and len(x) >= 2:
                a, b = x[0], x[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    da = float(a)
                    db = float(b)
                    if db > da:
                        total += (db - da)

        return total if total > 0.0 else 0.0

    def _update_emas(self, has_spot: bool) -> None:
        self._steps += 1
        alpha = 0.03
        hs = 1.0 if has_spot else 0.0
        self._ema_avail = (1.0 - alpha) * self._ema_avail + alpha * hs

        drop = 0.0
        if self._last_has_spot is not None and self._last_has_spot and (not has_spot):
            drop = 1.0
        self._ema_drop = (1.0 - alpha) * self._ema_drop + alpha * drop

        self._last_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_emas(has_spot)

        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0)) if env is not None else 0.0
        gap = float(getattr(env, "gap_seconds", 60.0)) if env is not None else 60.0

        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        ro = float(getattr(self, "restart_overhead", 0.0))

        done = self._get_done_seconds()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-9:
            return ClusterType.NONE

        slack = remaining_time - remaining_work

        # Conservative buffers to avoid missing the hard deadline.
        base_margin = max(600.0, 2.0 * gap)  # 10 minutes or 2 steps
        min_buffer = max(2400.0, 12.0 * ro, 2.0 * gap)  # ~40 minutes or 12 restarts

        # Estimate future spot-loss events and their overhead impact if we keep using spot when available.
        steps_remaining = max(1.0, remaining_time / max(gap, 1e-6))
        q_drop = self._clamp(self._ema_drop, 0.0, 0.5)
        expected_drops = q_drop * steps_remaining
        safety_factor = 3.0
        drop_buffer = base_margin + safety_factor * (expected_drops + 1.0) * ro
        required_buffer = max(min_buffer, drop_buffer)

        # If we're behind schedule or too close to the deadline, commit to on-demand.
        if (not self._od_committed) and (slack <= required_buffer):
            self._od_committed = True

        if self._od_committed:
            return ClusterType.ON_DEMAND

        # Not committed: prefer SPOT whenever available (unless switching would consume the remaining buffer).
        if has_spot:
            extra_restart = ro if last_cluster_type != ClusterType.SPOT else 0.0
            if slack > (required_buffer + extra_restart):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # No spot available: wait (NONE) only if we can afford to spend a full step of slack.
        # Otherwise, use on-demand to keep progress and protect the deadline.
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if slack > (required_buffer + gap):
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)