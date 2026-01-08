import os
import json
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

        self._configured = False
        self._p_avail = 0.15
        self._p_alpha = 0.04

        self._prev_has_spot: Optional[bool] = None
        self._cur_spot_streak_steps = 0
        self._avg_spot_streak_steps = 6.0
        self._streak_alpha = 0.15

        self._min_p = 0.01
        self._conservatism = 0.85

        self._critical_slack_s: Optional[float] = None
        self._return_to_spot_slack_s: Optional[float] = None
        self._min_spot_run_s: Optional[float] = None

        self._ever_used_od = False

    def solve(self, spec_path: str) -> "Solution":
        cfg = {}
        try:
            if spec_path and os.path.exists(spec_path):
                with open(spec_path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    cfg = json.loads(txt)
        except Exception:
            cfg = {}

        if isinstance(cfg, dict):
            try:
                v = float(cfg.get("p_alpha", self._p_alpha))
                if 0.001 <= v <= 0.5:
                    self._p_alpha = v
            except Exception:
                pass
            try:
                v = float(cfg.get("conservatism", self._conservatism))
                if 0.5 <= v <= 1.0:
                    self._conservatism = v
            except Exception:
                pass
            try:
                v = float(cfg.get("min_p", self._min_p))
                if 0.001 <= v <= 0.2:
                    self._min_p = v
            except Exception:
                pass
        return self

    def _get_done_work_seconds(self) -> float:
        t = getattr(self, "task_done_time", None)
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        if isinstance(t, (list, tuple)):
            if not t:
                return 0.0
            try:
                last = float(t[-1])
            except Exception:
                last = 0.0
            try:
                s = float(sum(float(x) for x in t))
            except Exception:
                s = 0.0
            # If cumulative, sum is much larger than last.
            if last > 0 and s > last * 2.0:
                return max(0.0, last)
            # Otherwise assume segments.
            return max(0.0, s)
        return 0.0

    def _ensure_params(self) -> None:
        if self._configured:
            return
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        ro = float(getattr(self, "restart_overhead", 0.0))

        # Critical slack: when remaining slack is below this, commit to OD.
        # Bias toward safety (deadline penalty is catastrophic).
        self._critical_slack_s = max(3600.0, 2.5 * ro + 2.0 * gap)

        # If already on OD, only return to spot if we have ample slack.
        self._return_to_spot_slack_s = max(5400.0, 6.0 * ro + 4.0 * gap)

        # Minimum expected spot run length to justify switching from OD back to spot.
        self._min_spot_run_s = max(1800.0, 1.25 * ro)

        self._configured = True

    def _update_spot_stats(self, has_spot: bool) -> None:
        x = 1.0 if has_spot else 0.0
        self._p_avail = (1.0 - self._p_alpha) * self._p_avail + self._p_alpha * x

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._cur_spot_streak_steps = 1 if has_spot else 0
            return

        if has_spot:
            self._cur_spot_streak_steps += 1
        else:
            if self._prev_has_spot and self._cur_spot_streak_steps > 0:
                self._avg_spot_streak_steps = (
                    (1.0 - self._streak_alpha) * self._avg_spot_streak_steps
                    + self._streak_alpha * float(self._cur_spot_streak_steps)
                )
            self._cur_spot_streak_steps = 0

        self._prev_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_params()
        self._update_spot_stats(bool(has_spot))

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 60.0))
        deadline = float(getattr(self, "deadline", 0.0))
        task_duration = float(getattr(self, "task_duration", 0.0))
        restart_overhead = float(getattr(self, "restart_overhead", 0.0))

        done = self._get_done_work_seconds()
        remaining_work = max(0.0, task_duration - done)
        time_left = max(0.0, deadline - elapsed)
        slack_left = time_left - remaining_work

        if remaining_work <= 0.0:
            return ClusterType.NONE

        safety = restart_overhead + 2.0 * gap

        # Hard safety: if we're behind (or almost), use OD immediately.
        if time_left <= remaining_work + safety:
            self._ever_used_od = True
            return ClusterType.ON_DEMAND

        # Conservative estimate of spot availability for planning.
        p_eff = max(self._min_p, min(1.0, self._p_avail * self._conservatism))

        # If we were to only make progress when spot is available and otherwise pause,
        # expected completion time is remaining_work / p_eff. If that fits, we can pause
        # during outages; otherwise we must use OD during outages.
        spot_only_time = remaining_work / p_eff

        critical_slack = float(self._critical_slack_s or 3600.0)
        if slack_left <= critical_slack:
            self._ever_used_od = True
            # Even if spot is available, commit to OD to reduce interruption risk.
            return ClusterType.ON_DEMAND

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and self._ever_used_od:
                expected_streak_s = float(self._avg_spot_streak_steps) * gap
                if (slack_left >= float(self._return_to_spot_slack_s or 7200.0) and
                        expected_streak_s >= float(self._min_spot_run_s or 1800.0) and
                        p_eff >= 0.12):
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot available this step.
        if last_cluster_type == ClusterType.ON_DEMAND and self._ever_used_od:
            return ClusterType.ON_DEMAND

        # Decide between pausing and OD based on feasibility of "spot-only progress".
        if spot_only_time <= (time_left - safety) and slack_left > gap:
            return ClusterType.NONE

        self._ever_used_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)