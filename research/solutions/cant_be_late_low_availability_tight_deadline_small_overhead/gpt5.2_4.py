import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_slack_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self._args = args
        self._reset_state()

    def _reset_state(self) -> None:
        self._prev_done = None  # type: Optional[float]
        self._avail_prev = None  # type: Optional[bool]
        self._avail_run_seconds = 0.0
        self._up_time = 0.0
        self._down_time = 0.0
        self._mean_up = 2.0 * 3600.0
        self._mean_down = 2.0 * 3600.0
        self._beta = 0.12
        self._od_hold_until = -1.0

    def solve(self, spec_path: str) -> "Solution":
        self._reset_state()
        return self

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _done_seconds(self) -> float:
        t = getattr(self, "task_done_time", None)
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, (list, tuple)):
            s = 0.0
            for v in t:
                try:
                    s += float(v)
                except Exception:
                    pass
            return s
        try:
            return float(sum(t))  # type: ignore[arg-type]
        except Exception:
            try:
                return float(t)
            except Exception:
                return 0.0

    def _update_availability_stats(self, has_spot: bool, gap_seconds: float) -> None:
        if self._avail_prev is None:
            self._avail_prev = has_spot
            self._avail_run_seconds = gap_seconds
        else:
            if has_spot == self._avail_prev:
                self._avail_run_seconds += gap_seconds
            else:
                run = max(self._avail_run_seconds, gap_seconds)
                b = self._beta
                if self._avail_prev:
                    self._mean_up = (1.0 - b) * self._mean_up + b * run
                else:
                    self._mean_down = (1.0 - b) * self._mean_down + b * run
                self._avail_prev = has_spot
                self._avail_run_seconds = gap_seconds

        if has_spot:
            self._up_time = self._avail_run_seconds
            self._down_time = 0.0
        else:
            self._down_time = self._avail_run_seconds
            self._up_time = 0.0

        self._mean_up = max(self._mean_up, gap_seconds)
        self._mean_down = max(self._mean_down, gap_seconds)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        if env is None:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        gap = self._safe_float(getattr(env, "gap_seconds", 60.0), 60.0)
        elapsed = self._safe_float(getattr(env, "elapsed_seconds", 0.0), 0.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        overhead = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        self._update_availability_stats(has_spot, gap)

        done = self._done_seconds()
        remaining = max(0.0, task_duration - done)
        time_left = deadline - elapsed
        slack = time_left - remaining

        if remaining <= 1e-9:
            self._prev_done = done
            return ClusterType.NONE

        if slack < -gap:
            self._prev_done = done
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        delta_done = None
        if self._prev_done is not None:
            delta_done = done - self._prev_done

        in_overhead = False
        if delta_done is not None:
            if last_cluster_type != ClusterType.NONE and delta_done < 0.25 * gap:
                in_overhead = True

        self._prev_done = done

        min_up_to_switch = max(2.0 * gap, 3.0 * overhead, 600.0)
        od_hold_seconds = max(min_up_to_switch, 900.0)
        reserve_slack = max(15.0 * overhead, 6.0 * gap, 2700.0)
        lock_margin = max(6.0 * overhead + 3.0 * gap, 1800.0)

        if in_overhead:
            if last_cluster_type == ClusterType.SPOT and not has_spot:
                if slack > reserve_slack:
                    return ClusterType.NONE
                if last_cluster_type != ClusterType.ON_DEMAND:
                    self._od_hold_until = elapsed + od_hold_seconds
                return ClusterType.ON_DEMAND
            return last_cluster_type

        if slack <= lock_margin:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if has_spot and slack < overhead:
                return ClusterType.SPOT
            self._od_hold_until = elapsed + od_hold_seconds
            return ClusterType.ON_DEMAND

        if not has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and elapsed < self._od_hold_until:
                return ClusterType.ON_DEMAND

            if slack > reserve_slack:
                if last_cluster_type == ClusterType.ON_DEMAND:
                    if slack > reserve_slack + 2.0 * overhead:
                        return ClusterType.NONE
                    return ClusterType.ON_DEMAND
                return ClusterType.NONE

            if last_cluster_type != ClusterType.ON_DEMAND:
                self._od_hold_until = elapsed + od_hold_seconds
            return ClusterType.ON_DEMAND

        # has_spot == True
        if last_cluster_type == ClusterType.SPOT:
            return ClusterType.SPOT

        if last_cluster_type == ClusterType.ON_DEMAND:
            if elapsed < self._od_hold_until:
                return ClusterType.ON_DEMAND
            if slack > reserve_slack and self._up_time >= min_up_to_switch and remaining > 2.0 * gap:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # last_cluster_type == NONE
        if slack <= reserve_slack * 0.75:
            return ClusterType.SPOT
        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)