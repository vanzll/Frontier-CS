import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "slack_guard_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass

        self._last_elapsed: float = -1.0
        self._last_has_spot: Optional[bool] = None
        self._streak_len_steps: int = 0
        self._avg_up_steps: float = 24.0    # default: 24 steps (~2h if 5min steps)
        self._avg_down_steps: float = 6.0   # default: 6 steps (~30min if 5min steps)

        self._od_until: float = 0.0
        self._forced_od: bool = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _maybe_reset_episode(self) -> None:
        env = getattr(self, "env", None)
        if env is None:
            return
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        if self._last_elapsed >= 0.0 and elapsed + 1e-9 < self._last_elapsed:
            self._reset_state()
        elif elapsed <= 1e-9 and self._last_elapsed > 1e-9:
            self._reset_state()

    def _reset_state(self) -> None:
        self._last_elapsed = -1.0
        self._last_has_spot = None
        self._streak_len_steps = 0
        self._avg_up_steps = 24.0
        self._avg_down_steps = 6.0
        self._od_until = 0.0
        self._forced_od = False

    def _update_spot_streaks(self, has_spot: bool) -> None:
        alpha = 0.2  # EWMA update rate
        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            self._streak_len_steps = 1
            return

        if has_spot == self._last_has_spot:
            self._streak_len_steps += 1
            return

        # Transition
        if self._last_has_spot:
            # ended an up-streak
            self._avg_up_steps = (1.0 - alpha) * self._avg_up_steps + alpha * float(self._streak_len_steps)
        else:
            # ended a down-streak
            self._avg_down_steps = (1.0 - alpha) * self._avg_down_steps + alpha * float(self._streak_len_steps)

        self._last_has_spot = has_spot
        self._streak_len_steps = 1

    def _work_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            return float(td)
        try:
            n = len(td)
        except Exception:
            return 0.0
        if n == 0:
            return 0.0

        try:
            x0 = td[0]
        except Exception:
            return 0.0

        # List of (start, end) segments
        if isinstance(x0, (tuple, list)) and len(x0) >= 2:
            s = 0.0
            for seg in td:
                try:
                    a = float(seg[0])
                    b = float(seg[1])
                    if b > a:
                        s += (b - a)
                except Exception:
                    continue
            return max(0.0, s)

        # List of durations or cumulative
        vals = []
        for x in td:
            try:
                vals.append(float(x))
            except Exception:
                continue
        if not vals:
            return 0.0

        s = float(sum(vals))
        last = float(vals[-1])
        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)

        # Heuristic: if sum is wildly larger than task_duration but last is within task_duration, treat as cumulative.
        if task_dur > 0.0 and s > 1.5 * task_dur and last <= 1.2 * task_dur:
            return max(0.0, min(last, task_dur))

        # Otherwise treat as additive segments.
        if task_dur > 0.0:
            return max(0.0, min(s, task_dur))
        return max(0.0, s)

    def _compute_buffers(self) -> tuple[float, float, float, float]:
        env = getattr(self, "env", None)
        gap = float(getattr(env, "gap_seconds", 300.0) or 300.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # Safety slack reserved to absorb restarts and mild trace uncertainty.
        safety = max(1800.0, 12.0 * overhead, 3.0 * gap)

        # If slack is extremely low, lock into OD to avoid a single restart causing failure.
        lock_slack = max(900.0, 4.0 * overhead, 2.0 * gap)

        # Switch-to-spot threshold: only switch OD->SPOT if typical up runs are long enough.
        switch_up_thresh = max(1800.0, 6.0 * overhead + 2.0 * gap)

        # OD commit duration (after starting OD during an outage): longer when spot is unstable.
        avg_up_seconds = max(gap, self._avg_up_steps * gap)
        if avg_up_seconds < 1800.0:
            od_commit = 3600.0
        elif avg_up_seconds < 7200.0:
            od_commit = 1800.0
        else:
            od_commit = 600.0

        return gap, overhead, safety, lock_slack, switch_up_thresh, od_commit

    def _enter_od_mode(self, elapsed: float, rem_wall: float, od_commit: float) -> None:
        # Shorten commitment as the deadline approaches.
        if rem_wall <= 4.0 * 3600.0:
            od_commit = min(od_commit, 600.0)
        if rem_wall <= 2.0 * 3600.0:
            od_commit = min(od_commit, 300.0)

        self._od_until = max(self._od_until, elapsed + max(300.0, od_commit))

    def _should_stay_od_on_spot(self, rem_wall: float, slack: float, switch_up_thresh: float) -> bool:
        env = getattr(self, "env", None)
        gap = float(getattr(env, "gap_seconds", 300.0) or 300.0)
        avg_up_seconds = max(gap, self._avg_up_steps * gap)

        # Near deadline / low slack, avoid OD->SPOT churn.
        if rem_wall <= 2.0 * 3600.0 and slack <= 1800.0:
            return True

        # If spot is very unstable (short up runs), don't switch back to spot from OD.
        if avg_up_seconds < switch_up_thresh:
            return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._maybe_reset_episode()

        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)

        self._update_spot_streaks(bool(has_spot))

        done = self._work_done_seconds()
        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)
        rem_work = max(0.0, task_dur - done)

        if rem_work <= 1e-9:
            self._last_elapsed = elapsed
            return ClusterType.NONE

        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        rem_wall = max(0.0, deadline - elapsed)

        gap, overhead, safety, lock_slack, switch_up_thresh, od_commit = self._compute_buffers()

        slack = rem_wall - rem_work

        # Hard feasibility guard: if even continuous running is tight, force OD.
        # Also keep a small guard for one restart + one step granularity.
        infeasible_guard = overhead + gap
        if rem_work + infeasible_guard >= rem_wall:
            self._forced_od = True

        # If slack is critically low, lock into OD (even if spot is available).
        if slack <= lock_slack:
            self._forced_od = True

        if self._forced_od:
            self._last_elapsed = elapsed
            return ClusterType.ON_DEMAND

        # If we're currently committed to OD, keep OD even if spot is available.
        if elapsed < self._od_until:
            self._last_elapsed = elapsed
            return ClusterType.ON_DEMAND

        if has_spot:
            # If we're already on OD, optionally keep OD if spot seems unstable.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if self._should_stay_od_on_spot(rem_wall=rem_wall, slack=slack, switch_up_thresh=switch_up_thresh):
                    self._last_elapsed = elapsed
                    return ClusterType.ON_DEMAND
            self._last_elapsed = elapsed
            return ClusterType.SPOT

        # No spot available: use slack budget to pause for free; otherwise OD.
        if slack > safety:
            self._last_elapsed = elapsed
            return ClusterType.NONE

        self._enter_od_mode(elapsed=elapsed, rem_wall=rem_wall, od_commit=od_commit)
        self._last_elapsed = elapsed
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)