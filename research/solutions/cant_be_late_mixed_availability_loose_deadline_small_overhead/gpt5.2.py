import math
from typing import Any

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._reset_state()

    def _reset_state(self):
        self._initialized = False

        self._p_ema = 0.6
        self._p_alpha = 0.02

        self._total_steps = 0
        self._spot_steps = 0
        self._nospot_steps = 0

        self._last_has_spot = None

        self._consec_avail_steps = 0
        self._consec_out_steps = 0

        self._avail_len_ema_sec = None
        self._out_len_ema_sec = None
        self._run_alpha = 0.15

        self._trans_avail_to_out = 0
        self._avail_time_sec = 0.0

        self._done_sum_sec = 0.0
        self._done_idx = 0

        self._od_lock_left = 0
        self._entered_urgent_mode = False

    def solve(self, spec_path: str) -> "Solution":
        self._reset_state()
        return self

    def _gap(self) -> float:
        g = getattr(getattr(self, "env", None), "gap_seconds", None)
        if g is None or not isinstance(g, (int, float)) or g <= 0:
            return 300.0
        return float(g)

    def _init_on_first_step(self):
        if self._initialized:
            return
        gap = self._gap()
        self._avail_len_ema_sec = 12.0 * gap
        self._out_len_ema_sec = 6.0 * gap
        self._initialized = True

    def _update_done_work(self) -> float:
        td = getattr(self, "task_done_time", None)
        if not isinstance(td, list):
            self._done_sum_sec = 0.0
            self._done_idx = 0
            return 0.0

        if self._done_idx > len(td):
            self._done_idx = 0
            self._done_sum_sec = 0.0

        for i in range(self._done_idx, len(td)):
            item = td[i]
            try:
                if isinstance(item, (int, float)):
                    self._done_sum_sec += float(item)
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    self._done_sum_sec += float(item[1]) - float(item[0])
                elif isinstance(item, dict) and ("start" in item and "end" in item):
                    self._done_sum_sec += float(item["end"]) - float(item["start"])
            except Exception:
                pass

        self._done_idx = len(td)
        if self._done_sum_sec < 0:
            self._done_sum_sec = 0.0
        return self._done_sum_sec

    def _update_stats(self, has_spot: bool, last_cluster_type: ClusterType):
        self._init_on_first_step()
        gap = self._gap()

        self._total_steps += 1
        if has_spot:
            self._spot_steps += 1
            self._nospot_steps += 0
        else:
            self._nospot_steps += 1

        self._p_ema = (1.0 - self._p_alpha) * self._p_ema + self._p_alpha * (1.0 if has_spot else 0.0)
        self._p_ema = min(0.999, max(0.001, self._p_ema))

        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            self._consec_avail_steps = 1 if has_spot else 0
            self._consec_out_steps = 0 if has_spot else 1
            return

        if self._last_has_spot:
            self._avail_time_sec += gap

        if has_spot == self._last_has_spot:
            if has_spot:
                self._consec_avail_steps += 1
                self._consec_out_steps = 0
            else:
                self._consec_out_steps += 1
                self._consec_avail_steps = 0
        else:
            if self._last_has_spot:
                run_len = self._consec_avail_steps * gap
                if run_len > 0:
                    self._avail_len_ema_sec = (1.0 - self._run_alpha) * self._avail_len_ema_sec + self._run_alpha * run_len
                self._trans_avail_to_out += 1
            else:
                run_len = self._consec_out_steps * gap
                if run_len > 0:
                    self._out_len_ema_sec = (1.0 - self._run_alpha) * self._out_len_ema_sec + self._run_alpha * run_len

            if has_spot:
                self._consec_avail_steps = 1
                self._consec_out_steps = 0
            else:
                self._consec_out_steps = 1
                self._consec_avail_steps = 0

        self._last_has_spot = has_spot

    def _estimated_churn_penalty_sec(self, remaining_work_sec: float) -> float:
        if self._avail_time_sec <= 1e-6 or self._trans_avail_to_out <= 0:
            return 0.0
        trans_per_sec = self._trans_avail_to_out / self._avail_time_sec
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        if overhead <= 0:
            return 0.0
        return remaining_work_sec * trans_per_sec * overhead

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_stats(has_spot, last_cluster_type)

        gap = self._gap()
        elapsed = float(getattr(getattr(self, "env", None), "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._update_done_work()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-9:
            self._od_lock_left = 0
            return ClusterType.NONE

        if remaining_time <= 1e-9:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        speed_req = remaining_work / max(remaining_time, 1e-9)
        slack = remaining_time - remaining_work

        safety = max(3.0 * gap, 2.0 * overhead, 900.0)

        churn_pen = self._estimated_churn_penalty_sec(remaining_work)
        slack_eff = slack - min(churn_pen, 0.5 * slack if slack > 0 else 0.0)

        urgent = (slack_eff <= max(safety, 12.0 * gap))
        if urgent:
            self._entered_urgent_mode = True
        urgent = urgent or self._entered_urgent_mode

        need_od_in_outages = (self._p_ema < (speed_req + 0.03)) or (slack_eff <= safety) or urgent

        if last_cluster_type == ClusterType.ON_DEMAND and self._od_lock_left > 0:
            self._od_lock_left -= 1
            return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                self._od_lock_left = 0
                return ClusterType.SPOT

            if last_cluster_type == ClusterType.ON_DEMAND:
                if urgent or slack_eff <= (safety + overhead):
                    self._od_lock_left = 0
                    return ClusterType.ON_DEMAND

                stable_avail_sec = self._consec_avail_steps * gap
                min_stable_sec = max(2.0 * gap, 2.0 * overhead, 600.0)

                if (self._avail_len_ema_sec is not None and self._avail_len_ema_sec < max(4.0 * overhead, 4.0 * gap, 600.0)):
                    self._od_lock_left = 0
                    return ClusterType.ON_DEMAND

                if stable_avail_sec >= min_stable_sec:
                    self._od_lock_left = 0
                    return ClusterType.SPOT

                return ClusterType.ON_DEMAND

            self._od_lock_left = 0
            if urgent and (self._avail_len_ema_sec is not None and self._avail_len_ema_sec < max(4.0 * overhead, 4.0 * gap, 600.0)):
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # has_spot == False
        if remaining_time <= remaining_work + safety:
            dwell = max(1, int(math.ceil(overhead / max(gap, 1e-9))) + 1)
            self._od_lock_left = dwell
            return ClusterType.ON_DEMAND

        if need_od_in_outages:
            dwell = max(1, int(math.ceil(overhead / max(gap, 1e-9))) + 1)
            self._od_lock_left = dwell
            return ClusterType.ON_DEMAND

        out_elapsed = self._consec_out_steps * gap
        out_ema = float(self._out_len_ema_sec if self._out_len_ema_sec is not None else 6.0 * gap)

        wait_budget = max(0.0, 0.5 * max(0.0, slack_eff - safety))
        wait_cap = min(wait_budget, 1.5 * out_ema + safety)

        if out_elapsed < wait_cap:
            self._od_lock_left = 0
            return ClusterType.NONE

        dwell = max(1, int(math.ceil(overhead / max(gap, 1e-9))) + 1)
        self._od_lock_left = dwell
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)