import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cbl_deadline_guard_v3"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._initialized = False

        # Previous step bookkeeping (accounting happens on the next call).
        self._prev_choice: ClusterType = ClusterType.NONE
        self._prev_has_spot: Optional[bool] = None

        # Availability statistics.
        self._total_steps = 0
        self._spot_steps = 0
        self._trans_10 = 0  # spot -> no spot
        self._trans_01 = 0  # no spot -> spot

        # Streak stats.
        self._streak_type: Optional[bool] = None
        self._streak_len_steps = 0
        self._spot_run_steps_sum = 0
        self._spot_run_cnt = 0
        self._out_run_steps_sum = 0
        self._out_run_cnt = 0

        # EMA of spot availability.
        self._p_ema = 0.60

        # OD lock to reduce thrashing.
        self._od_lock_until = 0.0
        self._committed_od = False

        # Token-bucket-like credit for running OD during outages only a fraction of time.
        self._od_credit = 0.0

        # Progress tracking (for robustness).
        self._last_done_work = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            return None
        except Exception:
            return None

    def _get_done_work(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0
        try:
            if isinstance(tdt, (int, float)):
                return float(tdt)
            total = 0.0
            if isinstance(tdt, (list, tuple)):
                for e in tdt:
                    if isinstance(e, (int, float)):
                        total += float(e)
                    elif isinstance(e, (list, tuple)) and len(e) >= 2:
                        a = self._safe_float(e[0])
                        b = self._safe_float(e[1])
                        if a is not None and b is not None:
                            total += max(0.0, b - a)
                    elif isinstance(e, dict):
                        d = self._safe_float(e.get("duration"))
                        if d is not None:
                            total += max(0.0, d)
                        else:
                            a = self._safe_float(e.get("start_time"))
                            b = self._safe_float(e.get("end_time"))
                            if a is not None and b is not None:
                                total += max(0.0, b - a)
            return float(total)
        except Exception:
            return 0.0

    def _avg_spot_run_seconds(self, gap: float) -> float:
        if self._spot_run_cnt > 0:
            return max(gap, (self._spot_run_steps_sum / self._spot_run_cnt) * gap)
        return 2.0 * 3600.0

    def _update_availability_stats(self, has_spot: bool, alpha: float = 0.02) -> None:
        self._total_steps += 1
        if has_spot:
            self._spot_steps += 1

        if self._prev_has_spot is not None:
            if self._prev_has_spot and (not has_spot):
                self._trans_10 += 1
            elif (not self._prev_has_spot) and has_spot:
                self._trans_01 += 1

        if self._streak_type is None:
            self._streak_type = has_spot
            self._streak_len_steps = 1
        else:
            if has_spot == self._streak_type:
                self._streak_len_steps += 1
            else:
                if self._streak_type:
                    self._spot_run_cnt += 1
                    self._spot_run_steps_sum += self._streak_len_steps
                else:
                    self._out_run_cnt += 1
                    self._out_run_steps_sum += self._streak_len_steps
                self._streak_type = has_spot
                self._streak_len_steps = 1

        self._p_ema = (1.0 - alpha) * self._p_ema + alpha * (1.0 if has_spot else 0.0)
        self._prev_has_spot = has_spot

    def _estimate_p_low(self) -> float:
        # Conservative lower estimate for spot availability (online, with shrinkage + early margin).
        n = self._total_steps
        s = self._spot_steps

        # Mild prior around 0.55 to avoid being over-optimistic early.
        a = 5.5
        b = 4.5
        p_hat = (s + a) / (n + a + b) if (n + a + b) > 0 else 0.55

        # Use the minimum of global estimate and EMA to be conservative under regime changes.
        p_base = min(p_hat, self._p_ema)

        denom = (n + a + b + 1.0)
        se = math.sqrt(max(1e-12, p_base * (1.0 - p_base) / denom))

        # Dynamic margin: larger early, smaller later.
        margin = 0.06 + (0.18 / math.sqrt(n + 1.0))
        # ~10-20th percentile lower-ish bound
        p_low = p_base - 1.0 * se - margin
        return max(0.05, min(0.99, p_low))

    def _expected_overhead_future(self, p_low: float, time_rem: float, gap: float) -> float:
        # Estimate future restart overhead if we keep using spot (preemptions cause restarts).
        try:
            restart = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        except Exception:
            restart = 0.0
        if restart <= 0.0 or time_rem <= 0.0:
            return 0.0

        avg_run = self._avg_spot_run_seconds(gap)
        # Expected preemptions ≈ time_in_spot / avg_run
        expected_preemptions = (p_low * time_rem) / max(avg_run, gap)
        return max(0.0, expected_preemptions) * restart

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 60.0) or 60.0)

        if not self._initialized:
            self._initialized = True
            self._last_done_work = self._get_done_work()

        # Account for previous step choice (wall-time accounting; used only for smoothing/locks).
        # (We keep it minimal to avoid depending on simulator specifics.)
        # No-op currently; placeholder for future extensions.

        # Update availability stats with current observation.
        self._update_availability_stats(has_spot)

        done_work = self._get_done_work()
        if done_work < self._last_done_work:
            done_work = self._last_done_work
        self._last_done_work = done_work

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        remaining_work = max(0.0, task_duration - done_work)
        time_rem = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-9:
            self._prev_choice = ClusterType.NONE
            return ClusterType.NONE

        # Conservative finish safety.
        safety = max(2.0 * gap, 0.25 * restart, 300.0)

        # If we ran OD previously and are within lock, stay on OD to avoid thrashing.
        if last_cluster_type == ClusterType.ON_DEMAND and elapsed < self._od_lock_until:
            self._prev_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Compute OD-now hard feasibility check.
        od_start_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart
        t_needed_od_now = remaining_work + od_start_overhead

        if t_needed_od_now >= (time_rem - safety):
            self._committed_od = True

        # Conservative risk check for continuing to use spot (future preemptions eat slack).
        p_low = self._estimate_p_low()
        overhead_future = 0.0 if self._committed_od else self._expected_overhead_future(p_low, time_rem, gap)

        if not self._committed_od:
            if (remaining_work + overhead_future) >= (time_rem - safety):
                self._committed_od = True

        # Late-stage: reduce risk by committing to OD if any meaningful interruption could be fatal.
        if not self._committed_od:
            slack_now = time_rem - remaining_work
            if time_rem <= max(3.0 * restart + 2.0 * gap, 90.0 * 60.0):
                if slack_now <= max(2.5 * restart, 30.0 * 60.0):
                    self._committed_od = True

        if self._committed_od:
            # Set/refresh an OD lock when switching into OD.
            if last_cluster_type != ClusterType.ON_DEMAND:
                # Lock long enough to amortize restart overhead; bounded.
                avg_run = self._avg_spot_run_seconds(gap)
                lock = max(1800.0, min(7200.0, 0.5 * avg_run))
                self._od_lock_until = max(self._od_lock_until, elapsed + lock)
            self._prev_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Use effective time that discounts expected future overhead from spot interruptions.
        effective_time = max(0.0, time_rem - overhead_future)

        # Minimum OD work needed in the remaining horizon if we run spot whenever available.
        expected_spot_work = p_low * effective_time
        req_od_work = max(0.0, remaining_work - expected_spot_work)

        expected_outage_time = max(0.0, (1.0 - p_low) * effective_time)

        # If expected outages cannot cover required OD work, we must also use OD during availability.
        if expected_outage_time <= 1e-9 and req_od_work > 1e-9:
            # Commit OD (or at least run it now) since there is no "outage budget".
            if last_cluster_type != ClusterType.ON_DEMAND:
                avg_run = self._avg_spot_run_seconds(gap)
                lock = max(1800.0, min(7200.0, 0.5 * avg_run))
                self._od_lock_until = max(self._od_lock_until, elapsed + lock)
            self._prev_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # If spot is available, prefer spot unless OD work requirement is extreme (rare here due to check above).
        if has_spot:
            self._prev_choice = ClusterType.SPOT
            return ClusterType.SPOT

        # Spot unavailable: decide between OD and NONE to meet deadline at minimal OD usage.
        # Duty cycle for OD during outages.
        duty = 1.0
        if expected_outage_time > 1e-9:
            duty = min(1.0, max(0.0, req_od_work / expected_outage_time))

        # If we're near the edge, force OD.
        if remaining_work >= (time_rem - safety - restart):
            duty = 1.0

        # Token bucket: add credit proportional to duty, spend 1 gap when choosing OD.
        self._od_credit = min(self._od_credit, 2.0 * gap)
        self._od_credit += duty * gap

        if self._od_credit >= 0.98 * gap:
            self._od_credit -= gap
            # If starting OD, set lock to reduce switch churn.
            if last_cluster_type != ClusterType.ON_DEMAND:
                avg_run = self._avg_spot_run_seconds(gap)
                lock = max(1800.0, min(7200.0, 0.5 * avg_run))
                self._od_lock_until = max(self._od_lock_until, elapsed + lock)
            self._prev_choice = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        self._prev_choice = ClusterType.NONE
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)