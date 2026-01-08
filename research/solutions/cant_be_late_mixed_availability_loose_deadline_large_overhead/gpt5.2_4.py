import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_ucb_paced"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args

        self._a = 7.0  # Beta prior (pseudo-counts): available
        self._b = 3.0  # Beta prior (pseudo-counts): unavailable
        self._od_acc = 0.0
        self._inited = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_sum(x: Any) -> float:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(sum(x))
        except Exception:
            try:
                return float(x[-1])
            except Exception:
                return 0.0

    def _estimate_p(self, gap: float) -> float:
        denom = self._a + self._b
        if denom <= 1e-9:
            return 0.5
        return max(0.0, min(1.0, self._a / denom))

    def _update_availability_model(self, has_spot: bool, gap: float) -> None:
        # Exponential forgetting to adapt to non-stationary traces.
        # Half-life ~ 6 hours.
        half_life = 6.0 * 3600.0
        if gap <= 0.0:
            decay = 1.0
        else:
            decay = 0.5 ** (gap / half_life)
        self._a *= decay
        self._b *= decay
        if has_spot:
            self._a += 1.0
        else:
            self._b += 1.0

    def _compute_unavail_od_fraction(self, W: float, T: float, p_est: float) -> float:
        # Fraction of *unavailable* time we should spend on OD so that expected
        # work completed by deadline matches remaining work.
        if T <= 1e-9:
            return 1.0
        if p_est >= 1.0 - 1e-9:
            return 0.0
        expected_spot_work = p_est * T
        deficit = W - expected_spot_work
        if deficit <= 0.0:
            return 0.0
        expected_unavail_time = (1.0 - p_est) * T
        if expected_unavail_time <= 1e-9:
            return 1.0
        frac = deficit / expected_unavail_time
        if frac < 0.0:
            return 0.0
        if frac > 1.0:
            return 1.0
        return frac

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        if not self._inited:
            self._inited = True
            self._od_acc = 0.0

        self._update_availability_model(has_spot, gap)
        p_est = self._estimate_p(gap)

        work_done = self._safe_sum(getattr(self, "task_done_time", None))
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        W = task_duration - work_done
        if W <= 0.0:
            return ClusterType.NONE

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        T = deadline - elapsed
        if T <= 0.0:
            return ClusterType.NONE

        slack = T - W

        # Hard safety: if we're close to the deadline relative to remaining work,
        # avoid idling; also bias to OD very close to the end to avoid last-minute issues.
        endgame_buffer = 3.0 * overhead + 2.0 * gap
        if slack <= endgame_buffer:
            if has_spot and slack > (overhead + gap):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # If spot is available, prefer it. Add slight hysteresis to avoid frequent switching
        # when we essentially need OD almost all the time.
        g = self._compute_unavail_od_fraction(W, T, p_est)
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and g >= 0.90:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot unavailable: pace OD usage over unavailable time.
        # Run OD for a fraction g of unavailable steps; otherwise idle.
        if g >= 0.999:
            return ClusterType.ON_DEMAND

        self._od_acc += g
        if self._od_acc >= 1.0:
            self._od_acc -= 1.0
            return ClusterType.ON_DEMAND
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)