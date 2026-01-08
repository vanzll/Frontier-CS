import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_adaptive_v1"

    def __init__(self, args: Any = None):
        super().__init__(args)
        self._initialized = False

        # Online spot availability stats
        self._prev_has_spot: Optional[bool] = None
        self._streak_seconds: float = 0.0
        self._avail_steps: int = 0
        self._total_steps: int = 0
        self._ema_avail_run: Optional[float] = None
        self._ema_unavail_run: Optional[float] = None

        # Control / hysteresis
        self._od_mode: bool = False  # if True, commit to on-demand until finish
        self._spot_stable_steps: int = 0

        # Tunables
        self._beta_a = 3.0
        self._beta_b = 2.0
        self._ema_alpha = 0.20

        self._risk_factor = 0.92
        self._spot_switch_back_stable_steps = 3  # require stability before switching OD -> SPOT
        self._min_p_eff = 0.05
        self._max_p_eff = 0.98

        # When slack is below this multiple of overhead, be conservative
        self._slack_overhead_multiplier_commit = 3.0
        self._slack_overhead_multiplier_no_wait = 1.5

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _get_done_work_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)

        if isinstance(td, (int, float)):
            return float(td)

        if isinstance(td, list) and td:
            # Tuple segments: (start,end) or similar
            any_tuple = any(isinstance(x, (tuple, list)) for x in td)
            if any_tuple:
                total = 0.0
                for x in td:
                    if not isinstance(x, (tuple, list)) or len(x) < 2:
                        continue
                    a = self._safe_float(x[0], 0.0)
                    b = self._safe_float(x[1], 0.0)
                    if b >= a and a >= 0.0:
                        total += (b - a)
                    else:
                        # Fallback: if it's a duration-like second element
                        if b > 0.0:
                            total += b
                return total

            # Numeric list: could be per-segment durations or cumulative done
            vals = []
            for x in td:
                if isinstance(x, (int, float)):
                    vals.append(float(x))
                else:
                    try:
                        vals.append(float(x))
                    except Exception:
                        pass
            if not vals:
                return 0.0

            s = sum(vals)
            mx = max(vals)
            # Heuristic: if values look cumulative (nondecreasing and sum explodes), use max/last.
            nondecreasing = True
            prev = vals[0]
            for v in vals[1:]:
                if v + 1e-9 < prev:
                    nondecreasing = False
                    break
                prev = v

            task_dur = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
            if task_dur > 0.0 and nondecreasing and s > task_dur * 1.2 and mx <= task_dur * 1.1:
                return mx
            return s

        # Fallbacks: try potential env attributes
        env = getattr(self, "env", None)
        if env is not None:
            for attr in ("task_done_seconds", "done_seconds", "work_done_seconds", "elapsed_task_seconds"):
                if hasattr(env, attr):
                    v = self._safe_float(getattr(env, attr), 0.0)
                    if v >= 0.0:
                        return v

        return 0.0

    def _update_spot_stats(self, has_spot: bool) -> None:
        gap = self._safe_float(getattr(self.env, "gap_seconds", 0.0), 0.0)
        if gap <= 0.0:
            gap = 60.0

        # Update Beta counts for p_hat
        self._total_steps += 1
        if has_spot:
            self._avail_steps += 1

        # Update run-length EMAs
        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._streak_seconds = gap
            return

        if has_spot == self._prev_has_spot:
            self._streak_seconds += gap
        else:
            finished = self._streak_seconds if self._streak_seconds > 0.0 else gap
            if self._prev_has_spot:
                if self._ema_avail_run is None:
                    self._ema_avail_run = finished
                else:
                    self._ema_avail_run = (1.0 - self._ema_alpha) * self._ema_avail_run + self._ema_alpha * finished
            else:
                if self._ema_unavail_run is None:
                    self._ema_unavail_run = finished
                else:
                    self._ema_unavail_run = (1.0 - self._ema_alpha) * self._ema_unavail_run + self._ema_alpha * finished

            self._prev_has_spot = has_spot
            self._streak_seconds = gap

    def _p_hat(self) -> float:
        return (self._avail_steps + self._beta_a) / (self._total_steps + self._beta_a + self._beta_b)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialized = True
            # Initialize streak tracking with the first observed has_spot
            self._prev_has_spot = None
            self._streak_seconds = 0.0

        self._update_spot_stats(has_spot)

        gap = self._safe_float(getattr(self.env, "gap_seconds", 0.0), 0.0)
        if gap <= 0.0:
            gap = 60.0

        elapsed = self._safe_float(getattr(self.env, "elapsed_seconds", 0.0), 0.0)
        deadline = self._safe_float(getattr(self, "deadline", 0.0), 0.0)
        remaining_time = max(deadline - elapsed, 0.0)

        task_duration = self._safe_float(getattr(self, "task_duration", 0.0), 0.0)
        done_work = self._get_done_work_seconds()
        remaining_work = max(task_duration - done_work, 0.0)

        if remaining_work <= 1e-6:
            return ClusterType.NONE

        restart_overhead = self._safe_float(getattr(self, "restart_overhead", 0.0), 0.0)

        # Track spot stability (for OD -> SPOT switch-back hysteresis)
        if has_spot:
            self._spot_stable_steps += 1
        else:
            self._spot_stable_steps = 0

        # If already committed to on-demand, stay on-demand.
        if self._od_mode:
            return ClusterType.ON_DEMAND

        # Feasibility check: if we started OD now and ran continuously, do we make it?
        od_start_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead
        time_needed_od = remaining_work + od_start_overhead
        if time_needed_od >= remaining_time - 1e-9:
            self._od_mode = True
            return ClusterType.ON_DEMAND

        # Estimate if spot-only (run when available, pause otherwise) is likely to finish in time.
        p = self._p_hat()
        p_eff = max(self._min_p_eff, min(self._max_p_eff, p * 0.85))

        ema_avail = self._ema_avail_run
        if ema_avail is None or ema_avail <= 0.0:
            ema_avail = 3600.0  # default 1 hour
        ema_avail = max(ema_avail, gap)

        # Approx expected number of spot "runs" to accumulate remaining work
        expected_runs = remaining_work / ema_avail
        overhead_runs = expected_runs * restart_overhead

        spot_start_overhead = 0.0
        if has_spot and last_cluster_type != ClusterType.SPOT:
            spot_start_overhead = restart_overhead

        expected_wall_spot = (remaining_work / p_eff) + overhead_runs + spot_start_overhead

        # Additional conservative commit criteria based on slack
        slack = remaining_time - remaining_work  # slack assuming perfect availability and no overhead
        if slack < self._slack_overhead_multiplier_commit * restart_overhead:
            self._od_mode = True
            return ClusterType.ON_DEMAND

        # If spot-only looks too risky, commit to on-demand.
        if expected_wall_spot > remaining_time * self._risk_factor:
            self._od_mode = True
            return ClusterType.ON_DEMAND

        # Otherwise, we can aim for spot-first policy:
        # - If spot is available: prefer spot, unless we're currently on OD and spot isn't stable yet.
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Only switch back if spot has been stable for a few steps and slack is comfortable.
                if self._spot_stable_steps < self._spot_switch_back_stable_steps:
                    return ClusterType.ON_DEMAND
                # If slack is getting tight, don't risk switching back.
                if slack < self._slack_overhead_multiplier_no_wait * restart_overhead:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot unavailable: decide to wait (NONE) or use on-demand temporarily.
        # Waiting budget: how much idle time we can afford if we switch to OD later and still meet deadline.
        waiting_budget = remaining_time - time_needed_od

        ema_unavail = self._ema_unavail_run
        if ema_unavail is None or ema_unavail <= 0.0:
            ema_unavail = 2.0 * gap
        ema_unavail = max(ema_unavail, gap)

        current_unavail = self._streak_seconds if (self._prev_has_spot is False) else gap
        expected_remaining_outage = max(ema_unavail - current_unavail, 0.0)

        safety = max(gap, 0.25 * ema_unavail)

        if waiting_budget > expected_remaining_outage + safety:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)