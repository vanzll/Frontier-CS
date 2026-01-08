import argparse
import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "lazy_slack_guard_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._alpha = 0.03
        self._p_hat = 0.2
        self._prev_has_spot: Optional[bool] = None

        self._avail_run_steps = 0
        self._unavail_run_steps = 0
        self._run_alpha = 0.2
        self._ewma_spot_runlen_s: Optional[float] = None
        self._ewma_unavail_runlen_s: Optional[float] = None

        self._od_started_due_to_urgency = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        # Common case: list of segments (start, end)
        try:
            last = tdt[-1]
        except Exception:
            return 0.0

        total = 0.0
        if isinstance(last, (tuple, list)) and len(last) == 2:
            for seg in tdt:
                if isinstance(seg, (tuple, list)) and len(seg) == 2:
                    a = self._safe_float(seg[0], 0.0)
                    b = self._safe_float(seg[1], 0.0)
                    if b > a:
                        total += (b - a)
                elif isinstance(seg, (int, float)):
                    v = float(seg)
                    if v > 0:
                        total += v
            td = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
            if td > 0:
                total = min(total, td)
            return max(0.0, total)

        # Numeric list: could be per-step increments or cumulative done values
        nums = []
        for v in tdt:
            if isinstance(v, (int, float)):
                nums.append(float(v))
        if not nums:
            return 0.0

        td = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        s = sum(nums)
        lastv = nums[-1]
        # Heuristic: if sum is huge relative to task_duration, list is cumulative samples
        if td > 0 and s > td * 2.0:
            done = lastv
        else:
            done = s

        if td > 0:
            done = min(done, td)
        return max(0.0, done)

    def _update_stats(self, has_spot: bool) -> None:
        gap = self._safe_float(getattr(getattr(self, "env", None), "gap_seconds", 0.0), 0.0)

        x = 1.0 if has_spot else 0.0
        self._p_hat = (1.0 - self._alpha) * self._p_hat + self._alpha * x

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._avail_run_steps = 1 if has_spot else 0
            self._unavail_run_steps = 0 if has_spot else 1
            return

        if has_spot:
            if self._prev_has_spot:
                self._avail_run_steps += 1
            else:
                # unavail run ended
                run_s = self._unavail_run_steps * gap
                if run_s > 0:
                    if self._ewma_unavail_runlen_s is None:
                        self._ewma_unavail_runlen_s = run_s
                    else:
                        self._ewma_unavail_runlen_s = (1.0 - self._run_alpha) * self._ewma_unavail_runlen_s + self._run_alpha * run_s
                self._unavail_run_steps = 0
                self._avail_run_steps = 1
        else:
            if not self._prev_has_spot:
                self._unavail_run_steps += 1
            else:
                # avail run ended
                run_s = self._avail_run_steps * gap
                if run_s > 0:
                    if self._ewma_spot_runlen_s is None:
                        self._ewma_spot_runlen_s = run_s
                    else:
                        self._ewma_spot_runlen_s = (1.0 - self._run_alpha) * self._ewma_spot_runlen_s + self._run_alpha * run_s
                self._avail_run_steps = 0
                self._unavail_run_steps = 1

        self._prev_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_stats(has_spot)

        gap = self._safe_float(getattr(getattr(self, "env", None), "gap_seconds", 300.0), 300.0)
        elapsed = self._safe_float(getattr(getattr(self, "env", None), "elapsed_seconds", 0.0), 0.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        restart = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        done = self._get_done_seconds()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-6:
            return ClusterType.NONE

        # Hard feasibility guard: if time is extremely tight, choose on-demand
        # (avoid relying on spot/no-op in this zone).
        tight_guard = restart + 2.0 * gap
        if remaining_time <= remaining_work + tight_guard:
            self._od_started_due_to_urgency = True
            return ClusterType.ON_DEMAND

        slack = remaining_time - remaining_work

        # Thresholds (seconds)
        idle_guard = max(2.0 * restart + 3.0 * gap, 12.0 * 60.0)          # need this much slack to safely idle
        spot_guard = max(3.0 * restart + 4.0 * gap, 18.0 * 60.0)          # need this much slack to safely (re)try spot

        # If we've entered urgency mode previously, avoid idling afterwards.
        if self._od_started_due_to_urgency and not has_spot:
            return ClusterType.ON_DEMAND

        if not has_spot:
            if slack > idle_guard:
                return ClusterType.NONE
            self._od_started_due_to_urgency = True
            return ClusterType.ON_DEMAND

        # has_spot == True
        if slack <= spot_guard:
            self._od_started_due_to_urgency = True
            return ClusterType.ON_DEMAND

        # If we're currently on on-demand, only switch to spot when it seems not too choppy
        # (prevents thrashing overhead from forcing extra OD later).
        if last_cluster_type == ClusterType.ON_DEMAND:
            runlen = self._ewma_spot_runlen_s
            if runlen is not None:
                # If typical spot streak is too short compared to restart overhead, avoid switching unless slack is huge.
                if runlen < max(6.0 * restart, 3.0 * gap) and slack < 6.0 * 3600.0:
                    return ClusterType.ON_DEMAND

        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)