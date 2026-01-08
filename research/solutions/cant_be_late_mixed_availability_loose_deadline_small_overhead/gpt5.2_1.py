import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_budget_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self._initialized = False

        self._prev_has_spot: Optional[bool] = None
        self._cur_run_len_s: float = 0.0

        self._p_hat: float = 0.6
        self._mean_on_s: float = 2.0 * 3600.0
        self._mean_off_s: float = 1.0 * 3600.0

        self._alpha_p: float = 0.02
        self._alpha_run: float = 0.10

        self._od_lock_until_s: float = 0.0
        self._last_elapsed_s: float = 0.0

        self._done_cache_s: float = 0.0
        self._done_cache_len: int = 0
        self._done_cache_last_id: int = 0
        self._done_cache_last_val: Optional[float] = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _is_numeric_list(self, x) -> bool:
        if not isinstance(x, list):
            return False
        for v in x:
            if not isinstance(v, (int, float)):
                return False
        return True

    def _compute_done_work_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._done_cache_s = 0.0
            self._done_cache_len = 0
            self._done_cache_last_id = 0
            self._done_cache_last_val = None
            return 0.0

        if self._is_numeric_list(tdt):
            n = len(tdt)
            last = float(tdt[-1]) if n else 0.0
            if n >= 3:
                s = float(sum(tdt))
                if s > 1.5 * last and last >= 0.0:
                    return max(0.0, last)
                return max(0.0, s)
            if n == 2:
                a, b = float(tdt[0]), float(tdt[1])
                if b >= a >= 0.0:
                    return max(0.0, b)
                return max(0.0, a + b)
            return max(0.0, last)

        try:
            cur_id = id(tdt)
            if cur_id != self._done_cache_last_id:
                self._done_cache_last_id = cur_id
                self._done_cache_len = 0
                self._done_cache_s = 0.0
                self._done_cache_last_val = None

            n = len(tdt)
            if n < self._done_cache_len:
                self._done_cache_len = 0
                self._done_cache_s = 0.0
                self._done_cache_last_val = None

            if n == self._done_cache_len and n > 0:
                last_item = tdt[-1]
                if isinstance(last_item, (int, float)):
                    lv = float(last_item)
                    if self._done_cache_last_val is not None and lv != self._done_cache_last_val:
                        self._done_cache_len = 0
                        self._done_cache_s = 0.0
                        self._done_cache_last_val = None

            if self._done_cache_len == 0:
                total = 0.0
                for item in tdt:
                    total += self._seg_to_seconds(item)
                self._done_cache_s = total
                self._done_cache_len = n
                self._done_cache_last_val = float(tdt[-1]) if n and isinstance(tdt[-1], (int, float)) else None
                return max(0.0, self._done_cache_s)

            if n > self._done_cache_len:
                total = self._done_cache_s
                for item in tdt[self._done_cache_len :]:
                    total += self._seg_to_seconds(item)
                self._done_cache_s = total
                self._done_cache_len = n
                self._done_cache_last_val = float(tdt[-1]) if n and isinstance(tdt[-1], (int, float)) else None

            return max(0.0, self._done_cache_s)
        except Exception:
            total = 0.0
            try:
                for item in tdt:
                    total += self._seg_to_seconds(item)
            except Exception:
                total = 0.0
            return max(0.0, total)

    def _seg_to_seconds(self, seg: Any) -> float:
        if seg is None:
            return 0.0
        if isinstance(seg, (int, float)):
            return max(0.0, float(seg))
        if isinstance(seg, (tuple, list)):
            if len(seg) == 2 and isinstance(seg[0], (int, float)) and isinstance(seg[1], (int, float)):
                return max(0.0, float(seg[1]) - float(seg[0]))
            total = 0.0
            for v in seg:
                if isinstance(v, (int, float)):
                    total += float(v)
            return max(0.0, total)
        if isinstance(seg, dict):
            if "duration" in seg and isinstance(seg["duration"], (int, float)):
                return max(0.0, float(seg["duration"]))
            if "done" in seg and isinstance(seg["done"], (int, float)):
                return max(0.0, float(seg["done"]))
            if "work" in seg and isinstance(seg["work"], (int, float)):
                return max(0.0, float(seg["work"]))
            if "start" in seg and "end" in seg and isinstance(seg["start"], (int, float)) and isinstance(seg["end"], (int, float)):
                return max(0.0, float(seg["end"]) - float(seg["start"]))
        return 0.0

    def _update_availability_stats(self, has_spot: bool, gap_s: float):
        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._cur_run_len_s = gap_s
            self._p_hat = 1.0 if has_spot else 0.0
            return

        self._p_hat = (1.0 - self._alpha_p) * self._p_hat + self._alpha_p * (1.0 if has_spot else 0.0)

        if has_spot == self._prev_has_spot:
            self._cur_run_len_s += gap_s
            return

        ended_len = self._cur_run_len_s
        if self._prev_has_spot:
            self._mean_on_s = (1.0 - self._alpha_run) * self._mean_on_s + self._alpha_run * ended_len
        else:
            self._mean_off_s = (1.0 - self._alpha_run) * self._mean_off_s + self._alpha_run * ended_len

        self._prev_has_spot = has_spot
        self._cur_run_len_s = gap_s

    def _reserve_seconds(self, remaining_work_s: float, gap_s: float) -> float:
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        base = max(900.0, 6.0 * ro, 2.0 * gap_s)
        dyn = 0.01 * max(0.0, remaining_work_s)
        return max(base, dyn)

    def _too_flappy_to_enter_spot(self, gap_s: float) -> bool:
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        mean_on = max(1.0, float(self._mean_on_s))
        ratio = ro / mean_on
        if ratio > 0.30 and self._p_hat < 0.55:
            return True
        if mean_on < max(4.0 * ro, 4.0 * gap_s) and self._p_hat < 0.35:
            return True
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        elapsed_s = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap_s = float(getattr(env, "gap_seconds", 300.0) or 300.0)

        if not self._initialized:
            self._initialized = True
            self._last_elapsed_s = elapsed_s

        self._update_availability_stats(has_spot, gap_s)

        task_duration_s = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline_s = float(getattr(self, "deadline", 0.0) or 0.0)
        done_s = self._compute_done_work_seconds()

        remaining_work_s = max(0.0, task_duration_s - done_s)
        remaining_time_s = max(0.0, deadline_s - elapsed_s)
        slack_s = remaining_time_s - remaining_work_s

        if remaining_work_s <= 0.0:
            return ClusterType.NONE

        if remaining_time_s <= 0.0:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        reserve_s = self._reserve_seconds(remaining_work_s, gap_s)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        hard_slack_s = max(3600.0, 2.0 * reserve_s)
        if slack_s <= hard_slack_s:
            self._od_lock_until_s = max(self._od_lock_until_s, elapsed_s + max(7200.0, 4.0 * self._mean_off_s, 8.0 * ro))
            return ClusterType.ON_DEMAND

        in_od_lock = elapsed_s < self._od_lock_until_s

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                if in_od_lock:
                    return ClusterType.ON_DEMAND
                if slack_s <= (reserve_s + ro + gap_s):
                    return ClusterType.ON_DEMAND
                if self._too_flappy_to_enter_spot(gap_s):
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
            return ClusterType.SPOT

        if last_cluster_type == ClusterType.ON_DEMAND:
            if in_od_lock:
                return ClusterType.ON_DEMAND
            if slack_s > (reserve_s + ro + 0.5 * self._mean_off_s) and self._p_hat > 0.65:
                self._od_lock_until_s = 0.0
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        if slack_s > reserve_s:
            return ClusterType.NONE

        lock_dur = max(3600.0, 2.0 * self._mean_off_s, 8.0 * ro)
        self._od_lock_until_s = max(self._od_lock_until_s, elapsed_s + lock_dur)
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)