import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._reset_internal(full=True)

    def _reset_internal(self, full: bool = False) -> None:
        self._prev_elapsed = None
        self._committed_od = False

        self._p_avail_ema = 0.65
        self._alpha_p = 0.05

        self._avg_outage_ema = 20 * 60.0  # seconds
        self._avg_avail_ema = 60 * 60.0   # seconds
        self._alpha_run = 0.15

        self._run_state = None
        self._run_len = 0.0

        self._last_decision = None
        self._last_switch_elapsed = None

        if full:
            self._total_steps = 0
            self._avail_steps = 0
            self._outage_steps = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_sum_task_done(task_done_time: Any) -> float:
        if task_done_time is None:
            return 0.0
        if isinstance(task_done_time, (int, float)):
            return float(task_done_time)
        if isinstance(task_done_time, (list, tuple)):
            if not task_done_time:
                return 0.0
            try:
                s = 0.0
                for x in task_done_time:
                    if isinstance(x, (int, float)):
                        s += float(x)
                    elif isinstance(x, (list, tuple)) and x and isinstance(x[-1], (int, float)):
                        s += float(x[-1])
                    elif isinstance(x, dict):
                        for k in ("duration", "work", "done", "seconds"):
                            if k in x and isinstance(x[k], (int, float)):
                                s += float(x[k])
                                break
                return s
            except Exception:
                try:
                    v = task_done_time[-1]
                    if isinstance(v, (int, float)):
                        return float(v)
                except Exception:
                    pass
        return 0.0

    def _update_stats(self, has_spot: bool) -> None:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        self._total_steps += 1
        if has_spot:
            self._avail_steps += 1
        else:
            self._outage_steps += 1

        x = 1.0 if has_spot else 0.0
        self._p_avail_ema = (1.0 - self._alpha_p) * self._p_avail_ema + self._alpha_p * x

        if self._run_state is None:
            self._run_state = has_spot
            self._run_len = gap
            return

        if has_spot == self._run_state:
            self._run_len += gap
            return

        finished_len = self._run_len
        if self._run_state:
            self._avg_avail_ema = (1.0 - self._alpha_run) * self._avg_avail_ema + self._alpha_run * max(gap, finished_len)
        else:
            self._avg_outage_ema = (1.0 - self._alpha_run) * self._avg_outage_ema + self._alpha_run * max(gap, finished_len)

        self._run_state = has_spot
        self._run_len = gap

    def _compute_remaining_work(self) -> float:
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        done = self._safe_sum_task_done(getattr(self, "task_done_time", None))
        remaining = task_duration - done
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if self._prev_elapsed is None or elapsed < self._prev_elapsed or elapsed == 0.0:
            self._reset_internal(full=False)
        self._prev_elapsed = elapsed

        self._update_stats(has_spot)

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = deadline - elapsed
        if time_left < 0.0:
            time_left = 0.0

        remaining_work = self._compute_remaining_work()
        if remaining_work <= 0.0:
            self._last_decision = ClusterType.NONE
            return ClusterType.NONE

        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # Hard feasibility guard: must run continuously on OD.
        if time_left <= remaining_work + max(overhead, gap):
            self._committed_od = True

        # Commit-to-OD condition (avoid late tail risk).
        commit_buffer = max(8.0 * overhead, 2.0 * gap, 0.75 * self._avg_outage_ema)
        if time_left <= remaining_work + commit_buffer:
            self._committed_od = True

        slack = time_left - remaining_work
        if slack < 0.0:
            slack = 0.0

        # Hysteresis for switching back from OD to spot.
        switch_back_slack = max(6.0 * overhead, 0.5 * self._avg_outage_ema, 3.0 * gap)

        # When spot is unavailable, decide OD vs NONE based on slack vs expected remaining outage.
        if not has_spot:
            if self._committed_od:
                self._last_decision = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

            # If we are currently in an outage run, estimate remaining outage time.
            if self._run_state is False:
                remaining_outage_est = max(0.0, self._avg_outage_ema - self._run_len)
            else:
                remaining_outage_est = self._avg_outage_ema

            wait_buffer = max(6.0 * overhead, 2.0 * gap)
            if slack >= remaining_outage_est + wait_buffer:
                self._last_decision = ClusterType.NONE
                return ClusterType.NONE
            else:
                # If we're forced to pay for progress, prefer OD.
                self._last_decision = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

        # Spot is available.
        if self._committed_od:
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND and slack < switch_back_slack:
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        self._last_decision = ClusterType.SPOT
        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)