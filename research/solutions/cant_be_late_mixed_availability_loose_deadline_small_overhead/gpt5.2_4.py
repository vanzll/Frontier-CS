import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass

        self._alpha = 0.12
        self._initialized = False

        self._prev_has_spot: Optional[bool] = None
        self._run_has_spot: Optional[bool] = None
        self._run_len_seconds: float = 0.0

        self._mean_up: float = 2.0 * 3600.0
        self._mean_down: float = 1.5 * 3600.0

        self._od_lock: bool = False
        self._od_lock_until_done: bool = False

    def solve(self, spec_path: str) -> "Solution":
        self._alpha = 0.12
        self._initialized = True

        self._prev_has_spot = None
        self._run_has_spot = None
        self._run_len_seconds = 0.0

        self._mean_up = 2.0 * 3600.0
        self._mean_down = 1.5 * 3600.0

        self._od_lock = False
        self._od_lock_until_done = False
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
            return float(getattr(self.env, "task_done_seconds", 0.0) or 0.0)

        if isinstance(tdt, (int, float)):
            return float(tdt)

        if not isinstance(tdt, (list, tuple)):
            return 0.0

        if len(tdt) == 0:
            return 0.0

        # Segments as (start, end)
        if isinstance(tdt[0], (list, tuple)) and len(tdt[0]) == 2:
            total = 0.0
            for seg in tdt:
                try:
                    a, b = float(seg[0]), float(seg[1])
                    if b > a:
                        total += (b - a)
                except Exception:
                    continue
            return total

        # Numeric list: could be segment durations or cumulative totals.
        vals = []
        for x in tdt:
            try:
                vals.append(float(x))
            except Exception:
                pass
        if not vals:
            return 0.0

        s = sum(v for v in vals if v > 0)
        last = vals[-1]
        try:
            task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        except Exception:
            task_dur = 0.0

        nondecreasing = True
        for i in range(1, len(vals)):
            if vals[i] + 1e-9 < vals[i - 1]:
                nondecreasing = False
                break

        # Heuristic: if looks cumulative, use last element.
        if nondecreasing and task_dur > 0:
            if 0.0 <= last <= task_dur * 1.01 and s > task_dur * 1.05:
                return max(0.0, last)

        return max(0.0, s)

    def _update_availability_stats(self, has_spot: bool) -> None:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 1.0

        if self._run_has_spot is None:
            self._run_has_spot = has_spot
            self._run_len_seconds = gap
            self._prev_has_spot = has_spot
            return

        if has_spot == self._run_has_spot:
            self._run_len_seconds += gap
        else:
            run_len = max(gap, self._run_len_seconds)
            a = self._alpha
            if self._run_has_spot:
                self._mean_up = (1.0 - a) * self._mean_up + a * run_len
            else:
                self._mean_down = (1.0 - a) * self._mean_down + a * run_len

            self._run_has_spot = has_spot
            self._run_len_seconds = gap

        self._prev_has_spot = has_spot

    def _estimate_p(self) -> float:
        up = max(1.0, self._mean_up)
        down = max(1.0, self._mean_down)
        p = up / (up + down)
        return self._clamp(p, 0.03, 0.97)

    def _expected_restart_overhead(self, remaining_work: float) -> float:
        mean_up = max(float(getattr(self.env, "gap_seconds", 1.0) or 1.0), self._mean_up)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        if ro <= 0 or remaining_work <= 0:
            return 0.0
        expected_interruptions = remaining_work / max(mean_up, 1.0)
        return expected_interruptions * ro

    def _switch_overhead(self, last_cluster_type: ClusterType, next_cluster_type: ClusterType) -> float:
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        if ro <= 0:
            return 0.0
        if next_cluster_type == ClusterType.NONE:
            return 0.0
        if last_cluster_type == next_cluster_type:
            return 0.0
        return ro

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_availability_stats(has_spot)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)

        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        done = self._get_done_seconds()

        remaining_work = max(0.0, task_dur - done)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        remaining_time = max(0.0, deadline - elapsed)
        slack = remaining_time - remaining_work

        initial_slack = max(1.0, deadline - task_dur) if (deadline > 0 and task_dur > 0) else max(1.0, slack + 1.0)
        urgency = self._clamp(1.0 - (slack / initial_slack), 0.0, 1.0)

        p_est = self._estimate_p()
        p_adj = self._clamp(p_est - (0.10 + 0.25 * urgency), 0.03, 0.95)

        expected_overhead = self._expected_restart_overhead(remaining_work)
        base_buffer = float(getattr(self, "restart_overhead", 0.0) or 0.0) + 2.0 * gap

        # Deterministic fail-safe: if we must run continuously from now, go on-demand.
        od_switch_oh = self._switch_overhead(last_cluster_type, ClusterType.ON_DEMAND)
        must_od_if = remaining_work + od_switch_oh + gap
        if remaining_time <= must_od_if + base_buffer:
            self._od_lock_until_done = True
            return ClusterType.ON_DEMAND

        if self._od_lock_until_done:
            return ClusterType.ON_DEMAND

        # Soft lock: once we decide OD under moderate/high urgency, avoid switching back.
        if self._od_lock and last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        risk_factor = 1.0 + 0.55 * urgency
        buffer = base_buffer

        # Expected total wall time if we primarily use spot and wait otherwise.
        expected_spot_wall = (remaining_work / max(p_adj, 0.03)) + expected_overhead

        if has_spot:
            # When spot is available, take it if expected completion is safe.
            spot_switch_oh = self._switch_overhead(last_cluster_type, ClusterType.SPOT)
            expected_total = (expected_spot_wall + spot_switch_oh) * risk_factor + buffer

            if expected_total <= remaining_time:
                return ClusterType.SPOT

            # Not enough room; go OD.
            if slack < 6.0 * 3600.0 or urgency > 0.55:
                self._od_lock = True
            return ClusterType.ON_DEMAND

        # No spot available: decide between waiting (NONE) and OD.
        down_remaining = 0.0
        if self._run_has_spot is False:
            down_remaining = max(0.0, self._mean_down - self._run_len_seconds)
        else:
            down_remaining = self._mean_down

        expected_total_wait = (down_remaining + expected_spot_wall) * risk_factor + buffer

        if expected_total_wait <= remaining_time and slack > (2.0 * gap + float(getattr(self, "restart_overhead", 0.0) or 0.0)):
            return ClusterType.NONE

        if slack < 6.0 * 3600.0 or urgency > 0.55:
            self._od_lock = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)