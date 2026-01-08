from __future__ import annotations

from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_wait_for_spot_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._od_lock = False

        # Spot availability stats (lightweight, used only for minor tuning).
        self._steps = 0
        self._spot_steps = 0
        self._ema_p = 0.2

        # Caches for computing completed work efficiently.
        self._td_cache_id = None
        self._td_cache_len = -1
        self._td_cache_last = None
        self._td_cache_sum = 0.0
        self._td_cache_max = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_spot_stats(self, has_spot: bool) -> None:
        self._steps += 1
        if has_spot:
            self._spot_steps += 1
        x = 1.0 if has_spot else 0.0
        self._ema_p = 0.92 * self._ema_p + 0.08 * x

    def _spot_p_est(self) -> float:
        # Conservative Beta prior; keep p from becoming 0 or 1.
        a, b = 1.0, 6.0
        p_raw = (self._spot_steps + a) / (self._steps + a + b) if self._steps > 0 else a / (a + b)
        p = 0.5 * self._ema_p + 0.5 * p_raw
        if p < 0.01:
            return 0.01
        if p > 0.99:
            return 0.99
        return p

    def _get_completed_work_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            return float(td)

        dur = float(getattr(self, "task_duration", 0.0) or 0.0)

        # List/tuple/array handling with caching (typical lengths are small, but this is safe).
        try:
            n = len(td)  # type: ignore[arg-type]
        except Exception:
            return 0.0

        td_id = id(td)
        last_item = None
        try:
            last_item = td[-1] if n > 0 else None  # type: ignore[index]
        except Exception:
            last_item = None

        if td_id == self._td_cache_id and n == self._td_cache_len and last_item == self._td_cache_last:
            return min(self._td_cache_sum, dur) if dur > 0 else self._td_cache_sum

        completed = 0.0
        maxv = 0.0

        if n == 0:
            completed = 0.0
            maxv = 0.0
        else:
            first = td[0]  # type: ignore[index]
            if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[0], (int, float)) and isinstance(first[1], (int, float)):
                s = 0.0
                for a, b in td:  # type: ignore[assignment]
                    da = float(a)
                    db = float(b)
                    if db > da:
                        s += (db - da)
                completed = s
                maxv = s
            else:
                # Numeric list: treat as segment durations by default.
                s = 0.0
                m = 0.0
                ok = True
                for x in td:  # type: ignore[assignment]
                    if not isinstance(x, (int, float)):
                        ok = False
                        break
                    fx = float(x)
                    s += fx
                    if fx > m:
                        m = fx
                if ok:
                    completed = s
                    maxv = m

                    # If this looks like cumulative timestamps, use max instead of sum.
                    if dur > 0 and maxv <= dur * 1.05 and completed > dur * 1.20:
                        completed = maxv
                else:
                    completed = 0.0
                    maxv = 0.0

        self._td_cache_id = td_id
        self._td_cache_len = n
        self._td_cache_last = last_item
        self._td_cache_sum = completed
        self._td_cache_max = maxv

        if dur > 0:
            if completed < 0.0:
                return 0.0
            return min(completed, dur)
        return max(0.0, completed)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_spot_stats(has_spot)

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = self._get_completed_work_seconds()
        remaining_work = task_duration - done
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        t = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        dt = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if dt <= 0.0:
            dt = 1.0

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        remaining_time = deadline - t
        if remaining_time <= 0.0:
            return ClusterType.NONE

        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        p = self._spot_p_est()

        # Buffer accounts for: discretization, at least one restart overhead, and some extra caution.
        # Keep it modest to preserve spot usage, but non-trivial to avoid deadline misses.
        buffer = overhead + 3.0 * dt + max(0.0, 300.0 * (1.0 - p))  # add up to 5 minutes when p is low

        slack = remaining_time - remaining_work  # how much "no-progress" time we can afford (ignoring future overhead)

        # Critical region: if we delay further, even continuous on-demand may not finish.
        latest_start_od = deadline - remaining_work - buffer
        if self._od_lock or (t + dt >= latest_start_od) or (slack <= 0.5 * buffer):
            self._od_lock = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # No spot: wait (NONE) if we still have enough slack to absorb it; otherwise switch to OD and lock.
        # Keep a small reserve for restart overhead/discretization.
        reserve = max(2.0 * overhead + 2.0 * dt, 0.25 * buffer)
        if slack > reserve + dt:
            return ClusterType.NONE

        self._od_lock = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)