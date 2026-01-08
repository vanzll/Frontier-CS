import math
import json
from collections import deque
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
        self.args = args

        self._total_steps = 0
        self._spot_avail_steps = 0

        self._recent_has_spot = deque(maxlen=256)

        self._last_has_spot: Optional[bool] = None
        self._cur_outage_steps = 0
        self._cur_spot_streak_steps = 0

        self._avg_outage_steps: Optional[float] = None
        self._avg_spot_streak_steps: Optional[float] = None

        self._od_run_len = 0
        self._spot_run_len = 0
        self._none_run_len = 0

        self._last_action: Optional[ClusterType] = None

    def solve(self, spec_path: str) -> "Solution":
        # Optional: allow JSON config overrides without relying on specific schema.
        try:
            with open(spec_path, "r") as f:
                _ = json.load(f)
        except Exception:
            pass
        return self

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _work_done_seconds(self) -> float:
        t = 0.0
        segs = getattr(self, "task_done_time", None)
        if not segs:
            return 0.0
        for seg in segs:
            if seg is None:
                continue
            if isinstance(seg, (tuple, list)) and len(seg) == 2:
                a = self._safe_float(seg[0], 0.0)
                b = self._safe_float(seg[1], 0.0)
                if b > a:
                    t += (b - a)
            else:
                t += self._safe_float(seg, 0.0)
        return max(0.0, t)

    def _ewma_update(self, prev: Optional[float], x: float, alpha: float = 0.2) -> float:
        if prev is None:
            return float(x)
        return float(prev) * (1.0 - alpha) + float(x) * alpha

    def _update_availability_stats(self, has_spot: bool) -> None:
        self._total_steps += 1
        if has_spot:
            self._spot_avail_steps += 1

        self._recent_has_spot.append(1 if has_spot else 0)

        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            self._cur_outage_steps = 0 if has_spot else 1
            self._cur_spot_streak_steps = 1 if has_spot else 0
            return

        if has_spot == self._last_has_spot:
            if has_spot:
                self._cur_spot_streak_steps += 1
            else:
                self._cur_outage_steps += 1
        else:
            # Transition: close previous streak.
            if self._last_has_spot:
                if self._cur_spot_streak_steps > 0:
                    self._avg_spot_streak_steps = self._ewma_update(
                        self._avg_spot_streak_steps, float(self._cur_spot_streak_steps)
                    )
                self._cur_spot_streak_steps = 1
                self._cur_outage_steps = 0
            else:
                if self._cur_outage_steps > 0:
                    self._avg_outage_steps = self._ewma_update(
                        self._avg_outage_steps, float(self._cur_outage_steps)
                    )
                self._cur_outage_steps = 1
                self._cur_spot_streak_steps = 0

            self._last_has_spot = has_spot

    def _update_run_length_counters(self, last_cluster_type: ClusterType) -> None:
        if last_cluster_type == ClusterType.ON_DEMAND:
            self._od_run_len += 1
        else:
            self._od_run_len = 0

        if last_cluster_type == ClusterType.SPOT:
            self._spot_run_len += 1
        else:
            self._spot_run_len = 0

        if last_cluster_type == ClusterType.NONE:
            self._none_run_len += 1
        else:
            self._none_run_len = 0

    def _recent_avail_rate(self) -> float:
        if not self._recent_has_spot:
            return 0.5
        return sum(self._recent_has_spot) / float(len(self._recent_has_spot))

    def _overall_avail_rate(self) -> float:
        # Smoothed estimator to avoid extremes.
        return (self._spot_avail_steps + 1.0) / (self._total_steps + 2.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not has_spot and last_cluster_type == ClusterType.SPOT:
            # Environment may pass SPOT as last even if spot just became unavailable.
            # We'll handle decisions below; do not return SPOT when has_spot is False.
            pass

        self._update_availability_stats(has_spot)
        self._update_run_length_counters(last_cluster_type)

        elapsed = self._safe_float(getattr(self.env, "elapsed_seconds", 0.0), 0.0)
        gap = self._safe_float(getattr(self.env, "gap_seconds", 300.0), 300.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        over = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        done = self._work_done_seconds()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-9:
            self._last_action = ClusterType.NONE
            return ClusterType.NONE

        if remaining_time <= 1e-9:
            self._last_action = ClusterType.NONE
            return ClusterType.NONE

        slack = remaining_time - remaining_work

        p_all = self._overall_avail_rate()
        p_recent = self._recent_avail_rate()
        p = 0.65 * p_recent + 0.35 * p_all

        over_steps = max(1, int(math.ceil(over / max(gap, 1e-9)))) if over > 0 else 1
        min_od_steps = max(2, over_steps)
        min_switch_spot_streak = max(3, 2 * over_steps)

        base_commit = 3600.0 + 2.0 * over + 4.0 * gap
        # Commit earlier when spot looks unreliable recently.
        unreliability = max(0.0, 0.55 - p)
        commit_buffer = base_commit + 7200.0 * (unreliability / 0.55)  # up to +2h

        if remaining_time <= remaining_work + commit_buffer:
            self._last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Estimate expected streak/outage lengths (in steps) for hysteresis.
        exp_spot_streak_steps = self._avg_spot_streak_steps
        if exp_spot_streak_steps is None:
            # Fall back to a conservative guess based on recent rate.
            exp_spot_streak_steps = max(1.0, 2.0 + 10.0 * p)
        exp_outage_steps = self._avg_outage_steps
        if exp_outage_steps is None:
            exp_outage_steps = max(1.0, 2.0 + 10.0 * (1.0 - p))

        # If spot is available now.
        if has_spot:
            # Avoid thrashing off on-demand unless it seems worthwhile.
            if last_cluster_type == ClusterType.ON_DEMAND and self._od_run_len < min_od_steps:
                self._last_action = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

            if last_cluster_type == ClusterType.ON_DEMAND:
                # Switch back only if we likely get a decent spot run to amortize restart overhead.
                if slack > (2.0 * over + 6.0 * gap) and exp_spot_streak_steps >= float(min_switch_spot_streak) and p >= 0.25:
                    self._last_action = ClusterType.SPOT
                    return ClusterType.SPOT
                self._last_action = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

            # Default: take spot when available.
            self._last_action = ClusterType.SPOT
            return ClusterType.SPOT

        # Spot not available now: decide to wait briefly vs on-demand.
        # If already on-demand, keep it for a minimum duration.
        if last_cluster_type == ClusterType.ON_DEMAND and self._od_run_len < min_od_steps:
            self._last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Waiting policy: small patience window, reduced if slack is low or outages tend to be long.
        base_patience = min(3600.0, max(over, gap) * (1.0 + 4.0 * p))  # 1x..5x base, capped at 1h
        # If outages usually long, don't wait too long; wait about 60% of typical outage.
        outage_based_cap = max(gap, 0.6 * exp_outage_steps * gap)
        patience = min(base_patience, outage_based_cap)

        # Don't spend slack too aggressively on waiting; reserve margin.
        reserve = max(2.0 * over, 6.0 * gap)
        wait_budget = max(0.0, slack - reserve)

        max_wait = min(patience, wait_budget)
        max_wait_steps = int(max_wait // gap) if gap > 0 else 0

        if max_wait_steps > 0 and self._cur_outage_steps <= max_wait_steps:
            self._last_action = ClusterType.NONE
            return ClusterType.NONE

        self._last_action = ClusterType.ON_DEMAND
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)