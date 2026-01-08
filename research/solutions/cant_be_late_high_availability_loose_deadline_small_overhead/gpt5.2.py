import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "lazy_slack_spot_v1"

    def __init__(self, args: Any):
        super().__init__(args)
        self._od_lock = False

        self._spot_up = 0
        self._spot_down = 0
        self._prev_has_spot: Optional[bool] = None

        self._done_mode: Optional[str] = None  # "cumulative", "sum", "segments", "scalar"
        self._done_sum_cache = 0.0
        self._done_len_cache = 0

        self._prior_p = 0.65
        self._prior_w = 50.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _estimate_p_low(self) -> float:
        n = self._spot_up + self._spot_down
        denom = n + self._prior_w
        p_hat = (self._spot_up + self._prior_p * self._prior_w) / denom
        var = p_hat * (1.0 - p_hat) / max(1.0, denom)
        se = math.sqrt(max(1e-12, var))
        p_low = p_hat - 1.0 * se
        if p_low < 0.01:
            p_low = 0.01
        elif p_low > 0.99:
            p_low = 0.99
        return p_low

    def _infer_done_mode(self, tdt: Any) -> None:
        if isinstance(tdt, (int, float)):
            self._done_mode = "scalar"
            self._done_sum_cache = float(tdt)
            self._done_len_cache = 1
            return

        if not isinstance(tdt, (list, tuple)):
            self._done_mode = "unknown"
            self._done_sum_cache = 0.0
            self._done_len_cache = 0
            return

        if len(tdt) == 0:
            self._done_mode = "sum"
            self._done_sum_cache = 0.0
            self._done_len_cache = 0
            return

        first = tdt[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            a, b = first
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                self._done_mode = "segments"
                self._done_sum_cache = 0.0
                self._done_len_cache = 0
                return

        if isinstance(first, (int, float)):
            self._done_mode = "sum"
            self._done_sum_cache = 0.0
            self._done_len_cache = 0
            return

        self._done_mode = "unknown"
        self._done_sum_cache = 0.0
        self._done_len_cache = 0

    def _get_done_seconds(self) -> float:
        tdt = self.task_done_time

        if self._done_mode is None:
            self._infer_done_mode(tdt)

        if self._done_mode == "scalar":
            try:
                return float(tdt)
            except Exception:
                return 0.0

        if not isinstance(tdt, (list, tuple)):
            try:
                return float(tdt)
            except Exception:
                return 0.0

        n = len(tdt)
        if n == 0:
            return 0.0

        if self._done_mode == "segments":
            if n > self._done_len_cache:
                s = self._done_sum_cache
                for i in range(self._done_len_cache, n):
                    seg = tdt[i]
                    if isinstance(seg, (list, tuple)) and len(seg) == 2:
                        a, b = seg
                        try:
                            s += float(b) - float(a)
                        except Exception:
                            pass
                self._done_sum_cache = s
                self._done_len_cache = n
            return max(0.0, self._done_sum_cache)

        if self._done_mode == "cumulative":
            try:
                return max(0.0, float(tdt[-1]))
            except Exception:
                return 0.0

        if self._done_mode == "sum":
            if n >= 5 and self._done_len_cache == 0:
                # Try to detect cumulative series vs per-step contributions.
                try:
                    sample = [float(x) for x in tdt[: min(n, 30)]]
                    last = float(tdt[min(n, 30) - 1])
                    avg = sum(sample) / len(sample)
                    if avg > 0 and last > 1.4 * avg and last <= float(self.task_duration) * 1.2:
                        self._done_mode = "cumulative"
                        return max(0.0, float(tdt[-1]))
                except Exception:
                    pass

            if n > self._done_len_cache:
                s = self._done_sum_cache
                for i in range(self._done_len_cache, n):
                    try:
                        s += float(tdt[i])
                    except Exception:
                        pass
                self._done_sum_cache = s
                self._done_len_cache = n
            return max(0.0, self._done_sum_cache)

        # unknown fallback
        try:
            if isinstance(tdt[-1], (int, float)):
                return max(0.0, float(tdt[-1]))
        except Exception:
            pass
        return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if has_spot:
            self._spot_up += 1
        else:
            self._spot_down += 1
        self._prev_has_spot = has_spot

        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0
        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0

        done = self._get_done_seconds()
        task_duration = float(self.task_duration)
        remaining = task_duration - done
        if remaining <= 0.0:
            return ClusterType.NONE

        deadline = float(self.deadline)
        time_left = deadline - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        restart_overhead = float(self.restart_overhead)

        # Slack: maximum time we can afford to not make progress (e.g., waiting on spot).
        slack = time_left - remaining

        # Reserve some slack for launch/restart overhead and discretization.
        reserve_wait = restart_overhead + 1.5 * gap
        reserve_lock = max(restart_overhead * 6.0 + 3.0 * gap, 6.0 * gap)

        if slack <= reserve_lock:
            self._od_lock = True

        if self._od_lock:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # No spot this step: wait if we still have enough slack, otherwise run on-demand.
        if slack > reserve_wait:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)