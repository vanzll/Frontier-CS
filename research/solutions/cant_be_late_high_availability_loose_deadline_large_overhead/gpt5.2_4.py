import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


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

        self._p_hat = 0.7
        self._initialized = False
        self._steps = 0
        self._od_lock = False

        # Tunables (seconds)
        self._half_life_seconds = 2.0 * 3600.0  # EWMA half-life for spot availability
        self._base_wait_buffer = 0.5 * 3600.0   # base buffer to allow waiting on spot outages
        self._panic_extra = 1.25 * 3600.0       # extra panic buffer beyond overhead
        self._switch_back_slack = 8.0 * 3600.0  # slack needed to consider switching OD->SPOT
        self._switch_back_min_tleft = 10.0 * 3600.0
        self._switch_back_min_rem = 3.0 * 3600.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _work_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0
        if isinstance(tdt, (list, tuple)):
            s = 0.0
            for v in tdt:
                s += self._safe_float(v, 0.0)
            return s
        return self._safe_float(tdt, 0.0)

    def _update_p_hat(self, has_spot: bool) -> None:
        gap = self._safe_float(getattr(self.env, "gap_seconds", 300.0), 300.0)
        hl = max(60.0, float(self._half_life_seconds))
        alpha = 1.0 - math.exp(-gap / hl)
        obs = 1.0 if has_spot else 0.0
        self._p_hat = (1.0 - alpha) * self._p_hat + alpha * obs
        if self._p_hat < 0.02:
            self._p_hat = 0.02
        elif self._p_hat > 0.98:
            self._p_hat = 0.98

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialized = True
            self._p_hat = 0.7
            self._steps = 0
            self._od_lock = False

        self._steps += 1
        self._update_p_hat(has_spot)

        elapsed = self._safe_float(getattr(self.env, "elapsed_seconds", 0.0), 0.0)
        gap = self._safe_float(getattr(self.env, "gap_seconds", 300.0), 300.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        restart_overhead = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        time_left = max(0.0, deadline - elapsed)
        done = self._work_done_seconds()
        remaining = max(0.0, task_duration - done)

        if remaining <= 1e-6:
            return ClusterType.NONE

        if time_left <= 1e-9:
            return ClusterType.ON_DEMAND

        slack = time_left - remaining

        # Risk buffers scale up when availability estimate is low.
        risk_buffer = (1.0 - self._p_hat) * 3.0 * 3600.0  # up to 3 hours
        wait_buffer = restart_overhead + max(self._base_wait_buffer, 2.0 * gap) + risk_buffer
        panic_buffer = 2.0 * restart_overhead + self._panic_extra + 0.5 * risk_buffer

        # If we're very close to the deadline, lock to on-demand and never switch back.
        if time_left <= remaining + panic_buffer:
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # If we ever fall behind a conservative feasibility bound, lock to on-demand.
        # (Even if spot comes back, avoid additional risk and overhead from flapping.)
        if slack <= restart_overhead + 0.5 * max(gap, 600.0):
            self._od_lock = True
            return ClusterType.ON_DEMAND

        # If already locked, stay on-demand.
        if self._od_lock:
            return ClusterType.ON_DEMAND

        # Proactive scheduling: if expected spot time is not enough, avoid idling during outages.
        expected_spot_work = self._p_hat * time_left
        need_more_than_spot = expected_spot_work < (remaining + 2.0 * restart_overhead)

        # Decision logic:
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Switch back only if early enough, enough slack, and enough remaining work.
                if (
                    (time_left > self._switch_back_min_tleft)
                    and (remaining > self._switch_back_min_rem)
                    and (slack > (self._switch_back_slack + 0.5 * risk_buffer))
                    and (self._p_hat > 0.55)
                ):
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot available this step.
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if need_more_than_spot:
            return ClusterType.ON_DEMAND

        # If we can safely wait one more step and still finish with OD afterwards, pause.
        if (time_left - gap) >= (remaining + wait_buffer):
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)