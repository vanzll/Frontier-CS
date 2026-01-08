import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any):
        super().__init__(args)
        self._p_est = 0.2
        self._p_alpha = 0.05

        self._prev_has_spot: Optional[bool] = None
        self._on_streak_steps = 0
        self._off_streak_steps = 0
        self._ema_beta = 0.10
        self._ema_on_len_steps = 12.0
        self._ema_off_len_steps = 36.0

        self._entered_hybrid = False
        self._committed_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_done_work_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0

        if isinstance(tdt, (int, float)):
            return float(tdt)

        if not isinstance(tdt, (list, tuple)):
            return 0.0

        total = 0.0
        for x in tdt:
            if isinstance(x, (int, float)):
                total += float(x)
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                a, b = x[0], x[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if b >= a:
                        total += float(b) - float(a)
                    else:
                        total += float(b)
        return total

    def _update_spot_stats(self, has_spot: bool) -> None:
        self._p_est = (1.0 - self._p_alpha) * self._p_est + self._p_alpha * (1.0 if has_spot else 0.0)

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._on_streak_steps = 1 if has_spot else 0
            self._off_streak_steps = 0 if has_spot else 1
            return

        if has_spot == self._prev_has_spot:
            if has_spot:
                self._on_streak_steps += 1
            else:
                self._off_streak_steps += 1
            return

        beta = self._ema_beta
        if self._prev_has_spot:
            prev_len = max(1, self._on_streak_steps)
            self._ema_on_len_steps = (1.0 - beta) * self._ema_on_len_steps + beta * float(prev_len)
            self._on_streak_steps = 1
            self._off_streak_steps = 0
        else:
            prev_len = max(1, self._off_streak_steps)
            self._ema_off_len_steps = (1.0 - beta) * self._ema_off_len_steps + beta * float(prev_len)
            self._off_streak_steps = 1
            self._on_streak_steps = 0

        self._prev_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_spot_stats(has_spot)

        env = self.env
        gap = float(getattr(env, "gap_seconds", 60.0))
        elapsed = float(getattr(env, "elapsed_seconds", 0.0))

        done = self._get_done_work_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0))
        remaining = max(0.0, task_duration - done)

        if remaining <= 0.0:
            return ClusterType.NONE

        deadline = float(getattr(self, "deadline", 0.0))
        time_left = deadline - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        restart_overhead = float(getattr(self, "restart_overhead", 0.0))

        safety_commit = max(0.5 * 3600.0, 4.0 * gap, 3.0 * restart_overhead)
        extra_if_switch_to_od = restart_overhead if last_cluster_type != ClusterType.ON_DEMAND else 0.0
        if self._committed_od or time_left <= remaining + extra_if_switch_to_od + safety_commit:
            self._committed_od = True
            return ClusterType.ON_DEMAND

        p = max(0.05, min(0.95, float(self._p_est)))
        expected_wait_calendar = remaining / p
        safety_hybrid = max(1.5 * 3600.0, 6.0 * gap, 4.0 * restart_overhead)

        if not self._entered_hybrid and time_left <= expected_wait_calendar + safety_hybrid:
            self._entered_hybrid = True

        if not self._entered_hybrid:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE

        slack = time_left - remaining

        if not has_spot:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.SPOT:
            return ClusterType.SPOT

        confirm_steps = max(2, int(math.ceil((restart_overhead + 600.0) / max(gap, 1e-9))))
        if last_cluster_type == ClusterType.ON_DEMAND:
            if self._on_streak_steps < confirm_steps:
                return ClusterType.ON_DEMAND

            min_switch_slack = max(2.0 * gap, restart_overhead + gap, 15.0 * 60.0)
            if slack <= min_switch_slack:
                return ClusterType.ON_DEMAND

            exp_on_seconds = float(self._ema_on_len_steps) * gap
            min_expected_run = max(20.0 * 60.0, 5.0 * gap, 2.0 * restart_overhead)
            if exp_on_seconds < min_expected_run:
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)