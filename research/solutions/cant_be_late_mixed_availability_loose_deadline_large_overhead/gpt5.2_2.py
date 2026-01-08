import math
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # Fallback stubs for non-eval environments
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, args=None):
            self.args = args
            self.env = type("Env", (), {"elapsed_seconds": 0.0, "gap_seconds": 60.0, "cluster_type": ClusterType.NONE})()
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._initialized = False
        self._last_elapsed: Optional[float] = None

        # Availability stats
        self._total_steps = 0
        self._spot_avail_steps = 0

        self._prev_has_spot: Optional[bool] = None
        self._avail_streak = 0
        self._consec_spot = 0
        self._consec_no_spot = 0

        # EMA of streak lengths (in steps)
        self._ema_true_run_steps: Optional[float] = None
        self._ema_false_run_steps: Optional[float] = None
        self._ema_alpha = 0.15

        # OD holding to avoid flapping switches
        self._od_hold_until: Optional[float] = None
        self._od_lock = False

        # Config (computed when env.gap_seconds known)
        self._stable_spot_seconds = 1800.0
        self._min_od_run_seconds = 1800.0
        self._warmup_seconds = 7200.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _reset_episode(self):
        self._last_elapsed = None
        self._total_steps = 0
        self._spot_avail_steps = 0
        self._prev_has_spot = None
        self._avail_streak = 0
        self._consec_spot = 0
        self._consec_no_spot = 0
        self._ema_true_run_steps = None
        self._ema_false_run_steps = None
        self._od_hold_until = None
        self._od_lock = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        gap = float(getattr(self.env, "gap_seconds", 60.0) or 60.0)
        over = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # Time-based hysteresis; tuned to overhead scale but not too large.
        self._stable_spot_seconds = max(900.0, 2.5 * over, 2.0 * gap)
        self._min_od_run_seconds = max(900.0, 2.0 * over, 2.0 * gap)
        self._warmup_seconds = max(3600.0, 6.0 * over, 6.0 * gap)

        self._initialized = True

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    def _get_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        try:
            if len(td) == 0:
                return 0.0
        except Exception:
            return 0.0

        # Common cases:
        # - list of cumulative done seconds (monotonic)
        # - list of segment tuples (start, end)
        # - list of per-segment durations
        try:
            last = td[-1]
        except Exception:
            return 0.0

        if isinstance(last, (int, float)):
            try:
                m = max(self._safe_float(v, 0.0) for v in td if isinstance(v, (int, float)))
                if m > 0:
                    return m
            except Exception:
                pass
            # If it's per-step done times (not monotonic), sum might be closer:
            try:
                s = sum(self._safe_float(v, 0.0) for v in td if isinstance(v, (int, float)))
                if s > 0:
                    return s
            except Exception:
                pass
            return self._safe_float(last, 0.0)

        if isinstance(last, (list, tuple)):
            total = 0.0
            any_seg = False
            for seg in td:
                if isinstance(seg, (int, float)):
                    total += self._safe_float(seg, 0.0)
                    any_seg = True
                elif isinstance(seg, (list, tuple)):
                    if len(seg) >= 2 and isinstance(seg[0], (int, float)) and isinstance(seg[1], (int, float)):
                        total += max(0.0, self._safe_float(seg[1], 0.0) - self._safe_float(seg[0], 0.0))
                        any_seg = True
                    elif len(seg) == 1 and isinstance(seg[0], (int, float)):
                        total += self._safe_float(seg[0], 0.0)
                        any_seg = True
            if any_seg and total > 0:
                return total

        # Fallback: try to interpret last element as done
        return self._safe_float(last, 0.0)

    def _update_availability_stats(self, has_spot: bool):
        self._total_steps += 1
        if has_spot:
            self._spot_avail_steps += 1

        if has_spot:
            self._consec_spot += 1
            self._consec_no_spot = 0
        else:
            self._consec_no_spot += 1
            self._consec_spot = 0

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._avail_streak = 1
            return

        if has_spot == self._prev_has_spot:
            self._avail_streak += 1
            return

        # Streak ended; update EMA of run lengths
        streak = float(self._avail_streak)
        if self._prev_has_spot:
            if self._ema_true_run_steps is None:
                self._ema_true_run_steps = streak
            else:
                a = self._ema_alpha
                self._ema_true_run_steps = a * streak + (1.0 - a) * self._ema_true_run_steps
        else:
            if self._ema_false_run_steps is None:
                self._ema_false_run_steps = streak
            else:
                a = self._ema_alpha
                self._ema_false_run_steps = a * streak + (1.0 - a) * self._ema_false_run_steps

        self._prev_has_spot = has_spot
        self._avail_streak = 1

    def _spot_availability_estimate(self) -> float:
        # Conservative prior to avoid overly optimistic pausing early.
        # Beta prior equivalent: alpha=2, beta=3 => mean=0.4
        alpha = 2.0
        beta = 3.0
        return (self._spot_avail_steps + alpha) / (self._total_steps + alpha + beta)

    def _avg_true_run_seconds(self, gap: float) -> float:
        if self._ema_true_run_steps is None:
            # Default to 1 hour worth of steps until we observe data
            return 3600.0
        return max(gap, float(self._ema_true_run_steps) * gap)

    def _overhead_delay_estimate(self, remaining_work: float, gap: float, p_est: float) -> float:
        # Very rough estimate of wallclock overhead due to restarts when running mostly on spot.
        # If spot runs are short, restarts are more frequent.
        avg_run = self._avg_true_run_seconds(gap)
        required_true_seconds = max(0.0, remaining_work)  # needs this many spot-available seconds to do work
        sessions = required_true_seconds / max(avg_run, gap)
        # If p_est is low, sessions estimate from avg_run may be too optimistic; increase slightly.
        inflation = 1.0 + max(0.0, (0.5 - p_est)) * 0.8
        sessions *= inflation
        sessions = max(0.0, sessions)
        return sessions * float(getattr(self, "restart_overhead", 0.0) or 0.0)

    def _must_make_progress_when_no_spot(self, elapsed: float, slack: float, remaining_work: float, remaining_time: float) -> bool:
        gap = float(getattr(self.env, "gap_seconds", 60.0) or 60.0)
        over = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # If close to deadline, always progress.
        hard_buffer = max(2.0 * over, 1800.0, 3.0 * gap)
        if slack <= hard_buffer:
            return True

        # Warmup period: avoid large initial pauses when spot might be scarce.
        if elapsed < self._warmup_seconds:
            # If slack is huge, allow short pauses; otherwise make progress.
            if slack > (6.0 * 3600.0):  # allow some early waiting only with lots of slack
                return False
            return True

        p_est = self._spot_availability_estimate()
        p_eff = max(0.05, min(0.98, p_est))

        # Expected extra time (beyond remaining_work) if we only run when spot is available:
        # remaining_work/p_eff total wallclock, so extra delay = remaining_work*(1/p_eff - 1)
        extra_wait = remaining_work * (1.0 / p_eff - 1.0)

        # Add overhead estimate from restarts.
        extra_over = self._overhead_delay_estimate(remaining_work, gap, p_eff)

        # Safety margin
        safety = max(2.0 * over, 1800.0, 3.0 * gap)

        # If slack comfortably covers expected delay, pausing is okay.
        # Otherwise, need OD during spot outages to keep progress.
        return slack < (extra_wait + extra_over + safety)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        if self._last_elapsed is None:
            self._reset_episode()
        elif elapsed < self._last_elapsed:
            self._reset_episode()
        self._last_elapsed = elapsed

        gap = float(getattr(self.env, "gap_seconds", 60.0) or 60.0)
        over = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        self._update_availability_stats(bool(has_spot))

        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)

        remaining_work = max(0.0, task_duration - done)
        remaining_time = max(0.0, deadline - elapsed)
        slack = remaining_time - remaining_work

        if remaining_work <= 0.0:
            return ClusterType.NONE

        # OD lock near deadline to avoid any chance of losing time to interruptions/restarts.
        lock_buffer = max(4.0 * over, 3600.0, 4.0 * gap)
        if not self._od_lock and slack <= lock_buffer:
            self._od_lock = True
            self._od_hold_until = None

        if self._od_lock:
            if last_cluster_type != ClusterType.ON_DEMAND:
                self._od_hold_until = elapsed + self._min_od_run_seconds
            return ClusterType.ON_DEMAND

        # If we're currently holding OD to avoid flapping, respect it.
        if last_cluster_type == ClusterType.ON_DEMAND and self._od_hold_until is not None and elapsed < self._od_hold_until:
            # Still allow switching to spot if we have *lots* of slack and spot is stable,
            # otherwise keep OD to avoid repeated restart overhead.
            stable_spot = has_spot and (self._consec_spot * gap >= self._stable_spot_seconds)
            if stable_spot and slack > (max(6.0 * over, 7200.0, 6.0 * gap)):
                self._od_hold_until = None
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Main policy
        if has_spot:
            stable_spot = (self._consec_spot * gap >= self._stable_spot_seconds)

            # If we're on OD and spot just appeared, only switch if stable and we have enough slack.
            if last_cluster_type == ClusterType.ON_DEMAND:
                switch_slack = max(3.0 * over, 1800.0, 3.0 * gap)
                if stable_spot and slack > switch_slack:
                    return ClusterType.SPOT
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        # No spot available: decide OD vs NONE
        must_progress = self._must_make_progress_when_no_spot(elapsed, slack, remaining_work, remaining_time)
        if must_progress:
            # Start/continue OD; hold it for a minimum duration to reduce flapping overhead.
            if last_cluster_type != ClusterType.ON_DEMAND:
                self._od_hold_until = elapsed + self._min_od_run_seconds
            return ClusterType.ON_DEMAND

        return ClusterType.NONE