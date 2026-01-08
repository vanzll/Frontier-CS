from __future__ import annotations

import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "debt_aware_spot_od_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._state_inited = False

    def solve(self, spec_path: str) -> "Solution":
        self._init_state()
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _init_state(self):
        self._state_inited = True

        self._prev_elapsed = 0.0
        self._prev_completed = 0.0

        self._ema_p = 0.6
        self._ema_spot_rate = 0.98
        self._ema_od_rate = 1.0

        self._od_debt = 0.0

        self._spot_streak = 0
        self._no_spot_streak = 0

        self._tdt_mode = None  # "sum", "last", "pairs"
        self._tdt_len = 0
        self._tdt_sum = 0.0

    def _lazy_init(self):
        if not getattr(self, "_state_inited", False):
            self._init_state()

    def _completed_work(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._tdt_mode = "sum"
            self._tdt_len = 0
            self._tdt_sum = 0.0
            return 0.0

        try:
            first = tdt[0]
        except Exception:
            return 0.0

        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)

        # Numeric list: either segment durations (sum) or cumulative totals (last)
        if isinstance(first, (int, float)):
            l = len(tdt)
            if self._tdt_mode not in (None, "sum", "last"):
                self._tdt_mode = None

            # Determine mode if unknown or if evidence changes
            if self._tdt_mode is None:
                # Default to sum early; switch to last if sum clearly exceeds task duration.
                self._tdt_mode = "sum"
                self._tdt_len = 0
                self._tdt_sum = 0.0

            if self._tdt_mode == "last":
                try:
                    return float(tdt[-1])
                except Exception:
                    return 0.0

            # sum mode (incremental)
            if l < self._tdt_len:
                self._tdt_len = 0
                self._tdt_sum = 0.0

            if l > self._tdt_len:
                try:
                    self._tdt_sum += float(sum(tdt[self._tdt_len : l]))
                    self._tdt_len = l
                except Exception:
                    # Fallback: full recompute
                    try:
                        self._tdt_sum = float(sum(tdt))
                        self._tdt_len = l
                    except Exception:
                        self._tdt_sum = 0.0
                        self._tdt_len = 0

            # If sum is way above task duration, treat list as cumulative totals.
            if task_dur > 0 and self._tdt_sum > task_dur * 1.2:
                self._tdt_mode = "last"
                try:
                    return float(tdt[-1])
                except Exception:
                    return 0.0

            return float(self._tdt_sum)

        # Pair list: [(start,end), ...]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            l = len(tdt)
            if self._tdt_mode != "pairs":
                self._tdt_mode = "pairs"
                self._tdt_len = 0
                self._tdt_sum = 0.0

            if l < self._tdt_len:
                self._tdt_len = 0
                self._tdt_sum = 0.0

            if l > self._tdt_len:
                inc = 0.0
                try:
                    for a, b in tdt[self._tdt_len : l]:
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b >= a:
                            inc += float(b - a)
                    self._tdt_sum += inc
                    self._tdt_len = l
                except Exception:
                    try:
                        self._tdt_sum = float(sum(float(b - a) for a, b in tdt if b >= a))
                        self._tdt_len = l
                    except Exception:
                        self._tdt_sum = 0.0
                        self._tdt_len = 0
            return float(self._tdt_sum)

        # Unknown structure: best-effort
        try:
            return float(tdt[-1])
        except Exception:
            return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 60.0

        dt = elapsed - self._prev_elapsed
        if not (dt > 0):
            dt = gap

        completed = self._completed_work()
        progress = completed - self._prev_completed
        if not (progress >= 0):
            progress = 0.0

        # Update availability streaks
        if has_spot:
            self._spot_streak += 1
            self._no_spot_streak = 0
        else:
            self._no_spot_streak += 1
            self._spot_streak = 0

        # Update EMAs
        alpha_p = 0.02
        alpha_r = 0.10

        self._ema_p = (1.0 - alpha_p) * self._ema_p + alpha_p * (1.0 if has_spot else 0.0)
        self._ema_p = min(0.99, max(0.01, self._ema_p))

        if last_cluster_type == ClusterType.SPOT:
            r = progress / dt if dt > 0 else 0.0
            r = min(1.05, max(0.0, r))
            self._ema_spot_rate = (1.0 - alpha_r) * self._ema_spot_rate + alpha_r * r
            self._ema_spot_rate = min(1.02, max(0.0, self._ema_spot_rate))
        elif last_cluster_type == ClusterType.ON_DEMAND:
            r = progress / dt if dt > 0 else 0.0
            r = min(1.05, max(0.0, r))
            self._ema_od_rate = (1.0 - alpha_r) * self._ema_od_rate + alpha_r * r
            self._ema_od_rate = min(1.02, max(0.5, self._ema_od_rate))

        self._prev_elapsed = elapsed
        self._prev_completed = completed

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        remaining_work = max(0.0, task_duration - completed)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND

        od_rate = max(0.85, min(1.02, self._ema_od_rate))
        spot_rate = max(0.0, min(1.02, self._ema_spot_rate))

        hard_slack = remaining_time - (remaining_work / od_rate)

        # Hard safety: ensure deadline met
        if remaining_time <= remaining_work + restart_overhead + 2.0 * dt:
            return ClusterType.ON_DEMAND
        if hard_slack <= max(2.0 * restart_overhead + 2.0 * dt, 1800.0):
            return ClusterType.ON_DEMAND

        # Conservative estimates
        p_adj = min(0.98, max(0.01, self._ema_p - 0.05))
        spot_rate_adj = max(0.0, spot_rate - 0.05)
        od_rate_adj = max(0.90, od_rate - 0.01)

        # Risk margin increases near deadline
        risk = 0.05
        if remaining_time < 12.0 * 3600.0:
            risk += 0.25 * (12.0 * 3600.0 - remaining_time) / (12.0 * 3600.0)
        risk = min(0.30, max(0.0, risk))

        required_rate = (remaining_work / remaining_time) * (1.0 + risk)

        base_rate = p_adj * spot_rate_adj
        den = (1.0 - p_adj) * od_rate_adj

        if required_rate <= base_rate + 1e-9:
            x_target = 0.0
        else:
            if den <= 1e-9:
                x_target = 1.0
            else:
                x_target = (required_rate - base_rate) / den
                x_target = min(1.0, max(0.0, x_target))

        if x_target < 0.002:
            x_target = 0.0
        elif x_target > 0.998:
            x_target = 1.0

        # Switch stability heuristic for OD -> SPOT
        if p_adj > 0.75:
            stable_steps = 1
        elif p_adj > 0.45:
            stable_steps = 2
        else:
            stable_steps = 3

        # Decision
        if has_spot:
            # Prefer spot when available, except when already on OD and spot seems too flaky
            if last_cluster_type == ClusterType.ON_DEMAND:
                if self._spot_streak >= stable_steps and spot_rate_adj >= 0.80 and (p_adj >= 0.35 or x_target < 0.6):
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND

            # If spot is performing very poorly and we need lots of OD, stick to OD
            if x_target >= 0.95 and p_adj <= 0.15 and spot_rate_adj < 0.75 and remaining_time < 8.0 * 3600.0:
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        # No spot: choose between OD and NONE using debt/quota to meet x_target
        if x_target <= 0.0:
            self._od_debt *= 0.5
            self._od_debt = max(-2.0 * dt, min(self._od_debt, 2.0 * 3600.0))
            return ClusterType.NONE

        if x_target >= 1.0:
            self._od_debt = 0.0
            return ClusterType.ON_DEMAND

        # Accumulate required OD time during no-spot periods
        self._od_debt += x_target * dt
        self._od_debt = max(-2.0 * dt, min(self._od_debt, 2.0 * 3600.0))

        # Hysteresis to avoid OD/NONE thrash
        if last_cluster_type == ClusterType.ON_DEMAND:
            if self._od_debt >= -0.2 * dt:
                self._od_debt -= dt
                self._od_debt = max(-2.0 * dt, min(self._od_debt, 2.0 * 3600.0))
                return ClusterType.ON_DEMAND
        elif last_cluster_type == ClusterType.NONE:
            if self._od_debt <= 0.6 * dt:
                return ClusterType.NONE

        if self._od_debt >= 0.6 * dt:
            self._od_debt -= dt
            self._od_debt = max(-2.0 * dt, min(self._od_debt, 2.0 * 3600.0))
            return ClusterType.ON_DEMAND

        return ClusterType.NONE