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
        self._initialized = False
        self._last_elapsed = -1.0
        self._ema_p = 0.7
        self._consec_no = 0
        self._consec_yes = 0
        self._od_until = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _get_done_seconds(self) -> float:
        t = getattr(self, "task_done_time", None)
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        try:
            s = 0.0
            for v in t:
                try:
                    s += float(v)
                except Exception:
                    pass
            if s > 0.0:
                return s
        except Exception:
            pass
        try:
            return float(t[-1])
        except Exception:
            return 0.0

    def _reset_episode(self):
        self._initialized = True
        self._last_elapsed = 0.0
        self._ema_p = 0.7
        self._consec_no = 0
        self._consec_yes = 0
        self._od_until = 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        if env is None:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        elapsed = self._safe_float(getattr(env, "elapsed_seconds", 0.0), 0.0)
        gap = self._safe_float(getattr(env, "gap_seconds", 300.0), 300.0)

        if (not self._initialized) or (elapsed == 0.0 and self._last_elapsed > 0.0) or (elapsed < self._last_elapsed):
            self._reset_episode()

        self._last_elapsed = elapsed

        alpha = 0.08
        self._ema_p = (1.0 - alpha) * self._ema_p + alpha * (1.0 if has_spot else 0.0)
        self._ema_p = min(max(self._ema_p, 0.02), 0.98)

        if has_spot:
            self._consec_yes += 1
            self._consec_no = 0
        else:
            self._consec_no += 1
            self._consec_yes = 0

        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        restart = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        done = self._get_done_seconds()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # Conservative reserve to avoid deadline miss due to overhead and last-minute outages.
        reserve = max(3.0 * restart, 2.0 * gap, 3600.0)
        slack = remaining_time - remaining_work

        # Hard panic: must run reliably now.
        if remaining_time <= remaining_work + restart + gap:
            self._od_until = max(self._od_until, elapsed + max(2.0 * restart, 1800.0))
            return ClusterType.ON_DEMAND

        # Commit-to-OD zone: too little slack to gamble on outages/switching.
        if slack <= reserve:
            self._od_until = max(self._od_until, elapsed + max(2.0 * restart, 1800.0))
            return ClusterType.ON_DEMAND

        # If currently in OD sticky mode, keep OD until timer expires unless we're very far from deadline.
        if elapsed < self._od_until:
            return ClusterType.ON_DEMAND

        if has_spot:
            # If we were on-demand and near the end, avoid thrashing.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if remaining_work < max(4.0 * restart, 2.0 * gap):
                    return ClusterType.ON_DEMAND
                if self._ema_p < 0.55 and remaining_time < 8.0 * 3600.0:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot: decide whether to wait (NONE) or pay for OD.
        slack_for_idle = slack - reserve
        if slack_for_idle <= 0.0:
            self._od_until = max(self._od_until, elapsed + max(2.0 * restart, 1800.0))
            return ClusterType.ON_DEMAND

        p = min(max(self._ema_p, 0.05), 0.95)
        expected_return_steps = max(1, int(round(1.0 / p)))
        max_wait_steps = max(1, int(expected_return_steps * 1.5))

        max_by_slack = int(slack_for_idle / max(gap, 1e-9))
        max_wait_steps = min(max_wait_steps, max_by_slack)

        # Also cap absolute waiting during an outage to prevent rare long outages from blowing the deadline.
        max_wait_steps = min(max_wait_steps, 24)  # up to ~24 steps (e.g., 2 hours if 5-min steps)

        if self._consec_no <= max_wait_steps:
            return ClusterType.NONE

        self._od_until = max(self._od_until, elapsed + max(2.0 * restart, 1800.0))
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)