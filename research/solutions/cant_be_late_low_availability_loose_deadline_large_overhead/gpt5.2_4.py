import argparse
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "slack_guard_ema_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self._reset_all_state()

    def _reset_all_state(self) -> None:
        self._od_committed = False
        self._overhead_remaining_guess = 0.0

        self._done_cache = 0.0
        self._done_cache_len = 0

        self._last_elapsed = None

        self._prev_has_spot = None
        self._avail_run_steps = 0
        self._avail_uptime_ema_seconds = None  # EMA of consecutive has_spot=True run length (seconds)

    def _reset_episode_state(self) -> None:
        self._od_committed = False
        self._overhead_remaining_guess = 0.0

        self._done_cache = 0.0
        self._done_cache_len = 0

        self._prev_has_spot = None
        self._avail_run_steps = 0
        self._avail_uptime_ema_seconds = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _segment_to_seconds(seg: Any) -> float:
        try:
            if isinstance(seg, (int, float)):
                return float(seg)
            if isinstance(seg, dict):
                v = seg.get("duration", None)
                if isinstance(v, (int, float)):
                    return float(v)
                v = seg.get("seconds", None)
                if isinstance(v, (int, float)):
                    return float(v)
                return 0.0
            if hasattr(seg, "duration"):
                v = getattr(seg, "duration")
                if isinstance(v, (int, float)):
                    return float(v)
            if isinstance(seg, (tuple, list)) and len(seg) >= 2:
                a, b = seg[0], seg[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if b >= a:
                        return float(b - a)
                # Fallback: try last element as duration
                v = seg[-1]
                if isinstance(v, (int, float)):
                    return float(v)
        except Exception:
            return 0.0
        return 0.0

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            self._done_cache = 0.0
            self._done_cache_len = 0
            return 0.0

        try:
            n = len(tdt)
        except Exception:
            self._done_cache = 0.0
            self._done_cache_len = 0
            return 0.0

        if n < self._done_cache_len:
            self._done_cache = 0.0
            self._done_cache_len = 0

        if n > self._done_cache_len:
            add = 0.0
            for seg in tdt[self._done_cache_len : n]:
                add += self._segment_to_seconds(seg)
            self._done_cache += add
            self._done_cache_len = n

        if self._done_cache < 0:
            self._done_cache = 0.0
        return self._done_cache

    def _update_spot_uptime_ema(self, has_spot: bool, gap: float) -> None:
        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._avail_run_steps = 1 if has_spot else 0
            return

        if has_spot:
            if self._prev_has_spot:
                self._avail_run_steps += 1
            else:
                self._avail_run_steps = 1
        else:
            if self._prev_has_spot and self._avail_run_steps > 0:
                run_seconds = float(self._avail_run_steps) * float(gap)
                if self._avail_uptime_ema_seconds is None:
                    self._avail_uptime_ema_seconds = run_seconds
                else:
                    alpha = 0.20
                    self._avail_uptime_ema_seconds = alpha * run_seconds + (1.0 - alpha) * self._avail_uptime_ema_seconds
            self._avail_run_steps = 0

        self._prev_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 1.0))
        if gap <= 0:
            gap = 1.0

        if self._last_elapsed is None or elapsed < self._last_elapsed or elapsed == 0.0:
            if self._last_elapsed is not None and (elapsed < self._last_elapsed or elapsed == 0.0):
                self._reset_episode_state()
        self._last_elapsed = elapsed

        if self._overhead_remaining_guess > 0.0:
            self._overhead_remaining_guess = max(0.0, self._overhead_remaining_guess - gap)

        self._update_spot_uptime_ema(bool(has_spot), gap)

        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        oh = float(getattr(self, "restart_overhead", 0.0))

        rem_work = max(0.0, task_duration - done)
        time_left = deadline - elapsed
        slack = time_left - rem_work

        if rem_work <= 0.0:
            self._overhead_remaining_guess = 0.0
            return ClusterType.NONE

        if not has_spot and last_cluster_type == ClusterType.SPOT:
            pass

        mean_uptime = self._avail_uptime_ema_seconds
        if mean_uptime is None or mean_uptime <= 0.0:
            mean_uptime = 3600.0
        mean_uptime = max(mean_uptime, gap)

        risk_mult = 1.0 + min(4.0, 3600.0 / mean_uptime)
        launch_buffer = max(oh + 2.0 * gap, 2.0 * gap)
        spot_safe_slack = max((2.0 * oh) * risk_mult, oh + 2.0 * gap, 4.0 * gap, 600.0)
        wait_safe_slack = launch_buffer + gap

        if self._od_committed:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.SPOT and has_spot and slack < launch_buffer:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if slack <= spot_safe_slack:
            self._od_committed = True
            if last_cluster_type == ClusterType.SPOT and has_spot and slack < launch_buffer:
                decision = ClusterType.SPOT
            else:
                decision = ClusterType.ON_DEMAND
            if decision == ClusterType.NONE:
                self._overhead_remaining_guess = 0.0
            elif decision != last_cluster_type:
                self._overhead_remaining_guess = oh
            return decision

        if self._overhead_remaining_guess > 0.0:
            if last_cluster_type == ClusterType.SPOT:
                if has_spot:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        if time_left <= 0.0:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        if not has_spot:
            if slack > wait_safe_slack:
                decision = ClusterType.NONE
            else:
                decision = ClusterType.ON_DEMAND

            if decision == ClusterType.NONE:
                self._overhead_remaining_guess = 0.0
            elif decision != last_cluster_type:
                self._overhead_remaining_guess = oh
            return decision

        if last_cluster_type == ClusterType.SPOT:
            if slack > spot_safe_slack:
                decision = ClusterType.SPOT
            else:
                if slack > launch_buffer + gap:
                    decision = ClusterType.ON_DEMAND
                else:
                    decision = ClusterType.SPOT
        else:
            extra_for_switch = oh + gap
            if slack > spot_safe_slack + extra_for_switch:
                decision = ClusterType.SPOT
            else:
                decision = ClusterType.ON_DEMAND

        if decision == ClusterType.NONE:
            self._overhead_remaining_guess = 0.0
        elif decision != last_cluster_type:
            self._overhead_remaining_guess = oh
        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)