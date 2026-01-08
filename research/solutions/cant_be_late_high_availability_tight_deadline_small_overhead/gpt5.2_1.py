import math
from typing import Optional, List

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._reset_episode_state()

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _reset_episode_state(self) -> None:
        self._last_elapsed_seconds: Optional[float] = None

        self._prev_has_spot: Optional[bool] = None
        self._down_consec_steps: int = 0
        self._up_consec_steps: int = 0

        self._current_outage_wait_seconds: float = 0.0
        self._total_wait_seconds: float = 0.0

        self._downtime_lengths: List[float] = []
        self._uptime_lengths: List[float] = []

        self._initial_slack_seconds: Optional[float] = None

    def _maybe_reset_for_new_episode(self, elapsed_seconds: float) -> None:
        if self._last_elapsed_seconds is None:
            self._last_elapsed_seconds = elapsed_seconds
            return
        if elapsed_seconds < self._last_elapsed_seconds or (elapsed_seconds == 0 and self._last_elapsed_seconds > 0):
            self._reset_episode_state()
        self._last_elapsed_seconds = elapsed_seconds

    @staticmethod
    def _percentile(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        vs = sorted(values)
        n = len(vs)
        if n == 1:
            return float(vs[0])
        q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
        pos = q * (n - 1)
        lo = int(pos)
        hi = lo + 1
        if hi >= n:
            return float(vs[lo])
        frac = pos - lo
        return float(vs[lo] * (1.0 - frac) + vs[hi] * frac)

    def _done_work_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0
        try:
            arr = [float(x) for x in tdt if x is not None]
            if not arr:
                return 0.0
            # Heuristic: if it's monotone non-decreasing and last is within duration, treat as cumulative.
            is_monotone = True
            for i in range(1, len(arr)):
                if arr[i] + 1e-9 < arr[i - 1]:
                    is_monotone = False
                    break
            dur = float(getattr(self, "task_duration", 0.0) or 0.0)
            if is_monotone and arr[-1] <= dur * 1.10:
                done = arr[-1]
            else:
                done = sum(arr)
            if dur > 0.0:
                done = min(done, dur)
            return max(0.0, done)
        except Exception:
            return 0.0

    def _update_availability_stats(self, has_spot: bool, gap: float) -> None:
        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            if has_spot:
                self._up_consec_steps = 1
                self._down_consec_steps = 0
            else:
                self._down_consec_steps = 1
                self._up_consec_steps = 0
                self._current_outage_wait_seconds = 0.0
            return

        if has_spot:
            if not self._prev_has_spot:
                # Downtime ended.
                down_len = float(self._down_consec_steps) * gap
                if down_len > 0:
                    self._downtime_lengths.append(down_len)
                    if len(self._downtime_lengths) > 200:
                        self._downtime_lengths = self._downtime_lengths[-200:]
                self._down_consec_steps = 0
                self._up_consec_steps = 1
                self._current_outage_wait_seconds = 0.0
            else:
                self._up_consec_steps += 1
        else:
            if self._prev_has_spot:
                # Uptime ended.
                up_len = float(self._up_consec_steps) * gap
                if up_len > 0:
                    self._uptime_lengths.append(up_len)
                    if len(self._uptime_lengths) > 200:
                        self._uptime_lengths = self._uptime_lengths[-200:]
                self._up_consec_steps = 0
                self._down_consec_steps = 1
                self._current_outage_wait_seconds = 0.0
            else:
                self._down_consec_steps += 1

        self._prev_has_spot = has_spot

    def _compute_wait_cap_seconds(self, slack: float, reserve: float, gap: float) -> float:
        # Available slack beyond reserve that we can spend waiting.
        available = slack - reserve
        if available <= 0:
            return 0.0

        recent = self._downtime_lengths[-60:] if self._downtime_lengths else []
        q75 = self._percentile(recent, 0.75) if recent else None

        # Default cap (favor short waits); adapt to observed downtime.
        base_cap = 15 * 60.0
        if q75 is not None and q75 > 0:
            base_cap = max(8 * 60.0, min(45 * 60.0, q75))

        # Do not wait longer than a fraction of remaining slack.
        frac_cap = 0.20 * slack
        cap = min(base_cap, frac_cap, available)

        # Ensure at least one step if we decide to wait at all.
        cap = max(0.0, cap)
        if cap > 0.0:
            cap = max(cap, gap)
        return cap

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0.0:
            gap = 60.0

        self._maybe_reset_for_new_episode(elapsed)
        self._update_availability_stats(has_spot, gap)

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._done_work_seconds()
        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)
        slack = remaining_time - remaining_work

        if self._initial_slack_seconds is None:
            self._initial_slack_seconds = slack

        if remaining_work <= 0.0:
            return ClusterType.NONE

        if remaining_time <= 0.0:
            return ClusterType.ON_DEMAND

        critical_slack = max(45 * 60.0, 12.0 * restart_overhead, 6.0 * gap)
        reserve_slack = max(90 * 60.0, 20.0 * restart_overhead, 10.0 * gap)

        # If we're too close to the deadline, prefer reliability.
        if slack <= critical_slack:
            return ClusterType.ON_DEMAND

        if has_spot:
            # Avoid rapid OD->SPOT switching when spot just came back briefly.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if self._up_consec_steps < 2 and slack <= (reserve_slack + restart_overhead + 2.0 * gap):
                    return ClusterType.ON_DEMAND
                if slack <= (critical_slack + restart_overhead + gap):
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot this step.
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        if slack <= reserve_slack:
            return ClusterType.ON_DEMAND

        # Optionally wait (NONE) for a short time per outage, spending slack to avoid OD cost.
        cap = self._compute_wait_cap_seconds(slack=slack, reserve=reserve_slack, gap=gap)
        if cap > 0.0 and self._current_outage_wait_seconds < cap:
            self._current_outage_wait_seconds += gap
            self._total_wait_seconds += gap
            return ClusterType.NONE

        return ClusterType.ON_DEMAND