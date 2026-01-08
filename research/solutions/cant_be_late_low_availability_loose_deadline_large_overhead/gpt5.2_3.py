import json
import math
import os
from collections import deque
from typing import Any, Deque, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cbll_slack_aware_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._initialized = False
        self._gap: Optional[float] = None

        self._hist: Optional[Deque[int]] = None
        self._hist_len: int = 0

        self._last_has_spot: Optional[bool] = None
        self._cur_run_len_steps: int = 0

        self._spot_on_runs_steps = []
        self._spot_off_runs_steps = []

        self._prior_p = 0.20  # conservative prior (low availability)
        self._min_p = 0.05

    def solve(self, spec_path: str) -> "Solution":
        # Optional: try to read any configuration if present.
        # Strategy should remain robust if spec_path is absent/unreadable.
        try:
            if spec_path and os.path.exists(spec_path):
                with open(spec_path, "r") as f:
                    _ = json.load(f)
        except Exception:
            pass
        return self

    @staticmethod
    def _sum_task_done(task_done_time: Any) -> float:
        if task_done_time is None:
            return 0.0
        if isinstance(task_done_time, (int, float)):
            return float(task_done_time)
        total = 0.0
        try:
            for x in task_done_time:
                if x is None:
                    continue
                if isinstance(x, (int, float)):
                    total += float(x)
                elif isinstance(x, (list, tuple)) and len(x) >= 2:
                    try:
                        total += float(x[1]) - float(x[0])
                    except Exception:
                        pass
                elif isinstance(x, dict):
                    # Best-effort handling for possible formats
                    if "duration" in x:
                        try:
                            total += float(x["duration"])
                        except Exception:
                            pass
                    elif "start" in x and "end" in x:
                        try:
                            total += float(x["end"]) - float(x["start"])
                        except Exception:
                            pass
        except Exception:
            return 0.0
        return max(0.0, total)

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            self._gap = float(getattr(self.env, "gap_seconds", 60.0))
        except Exception:
            self._gap = 60.0
        if not self._gap or self._gap <= 0:
            self._gap = 60.0

        # Use ~6 hours of history for availability estimate
        self._hist_len = int(max(10, round((6 * 3600) / self._gap)))
        self._hist = deque(maxlen=self._hist_len)
        self._initialized = True

    def _update_availability_stats(self, has_spot: bool):
        self._ensure_initialized()
        self._hist.append(1 if has_spot else 0)

        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            self._cur_run_len_steps = 1
            return

        if has_spot == self._last_has_spot:
            self._cur_run_len_steps += 1
            return

        # Run ended; record it
        if self._last_has_spot:
            self._spot_on_runs_steps.append(self._cur_run_len_steps)
        else:
            self._spot_off_runs_steps.append(self._cur_run_len_steps)

        self._last_has_spot = has_spot
        self._cur_run_len_steps = 1

    def _estimate_p(self) -> float:
        if not self._hist:
            return self._prior_p
        if len(self._hist) < max(5, self._hist_len // 10):
            # Blend prior with limited evidence
            obs = sum(self._hist) / max(1, len(self._hist))
            w = len(self._hist) / float(max(1, self._hist_len))
            return max(self._min_p, min(0.99, (1 - w) * self._prior_p + w * obs))
        return max(self._min_p, min(0.99, sum(self._hist) / float(len(self._hist))))

    def _mean_run_seconds(self, runs_steps, default_seconds: float) -> float:
        if not runs_steps:
            return default_seconds
        n = len(runs_steps)
        if n <= 0:
            return default_seconds
        # Use trimmed mean for robustness
        arr = sorted(runs_steps)
        lo = int(n * 0.1)
        hi = int(n * 0.9)
        if hi <= lo:
            mean_steps = sum(arr) / float(n)
        else:
            trimmed = arr[lo:hi]
            mean_steps = sum(trimmed) / float(len(trimmed))
        return max(self._gap, mean_steps * self._gap)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()
        self._update_availability_stats(has_spot)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        completed = self._sum_task_done(getattr(self, "task_done_time", None))
        remaining_work = max(0.0, task_duration - completed)
        remaining_time = max(0.0, deadline - elapsed)

        # If already done, stop paying.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # If no time left, try on-demand as last resort.
        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND

        slack = remaining_time - remaining_work
        urgency = remaining_work / max(1e-9, remaining_time)

        p_est = self._estimate_p()
        # Safety-adjusted availability estimate
        p_eff = max(self._min_p, min(0.99, p_est * 0.85))

        # Estimated average on/off durations
        mean_on = self._mean_run_seconds(self._spot_on_runs_steps, default_seconds=45 * 60.0)
        mean_off = self._mean_run_seconds(self._spot_off_runs_steps, default_seconds=90 * 60.0)

        gap = self._gap
        safety_margin = max(2.0 * restart_overhead, 2.0 * gap)

        # Hard safety: if we're tight on slack, use on-demand.
        if slack <= 0.0:
            return ClusterType.ON_DEMAND
        if slack <= safety_margin:
            return ClusterType.ON_DEMAND
        if remaining_time <= remaining_work + 3.0 * restart_overhead:
            return ClusterType.ON_DEMAND

        # When spot isn't available, decide whether to wait or pay for on-demand.
        if not has_spot:
            # If we need higher average progress than spot availability provides, run on-demand now.
            if urgency > p_eff:
                return ClusterType.ON_DEMAND

            # If we already committed to on-demand, keep it unless slack is very large.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > (mean_off + 2.0 * restart_overhead + 2.0 * gap) and urgency < max(self._min_p, p_eff * 0.8):
                    return ClusterType.NONE
                return ClusterType.ON_DEMAND

            # Otherwise wait if slack can cover expected downtime and restart.
            if slack > (1.2 * mean_off + restart_overhead + gap):
                return ClusterType.NONE

            return ClusterType.ON_DEMAND

        # Spot is available.
        # If we're extremely urgent, still prefer on-demand to reduce interruption risk.
        if urgency >= 0.97:
            return ClusterType.ON_DEMAND

        # If already on spot (or paused), use spot.
        if last_cluster_type != ClusterType.ON_DEMAND:
            return ClusterType.SPOT

        # Currently on on-demand; switch to spot only if likely beneficial and not too risky.
        # Require enough slack and a reasonable expected spot-on run length to amortize overhead.
        if slack > (2.5 * restart_overhead + 2.0 * gap) and mean_on > (2.0 * restart_overhead + 2.0 * gap) and urgency < 0.92:
            return ClusterType.SPOT

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)