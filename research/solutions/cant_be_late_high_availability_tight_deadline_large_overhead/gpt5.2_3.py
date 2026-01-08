import math
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

        # Conservative Bayesian-ish prior for spot availability.
        self._n_obs = 20
        self._k_obs = 12

        self._last_has_spot = None
        self._spot_streak = 0

        # Run-length tracking (in steps).
        self._curr_spot_run_steps = 0
        self._avg_spot_run_steps = None  # initialized once gap is known
        self._run_ema_alpha = 0.10

        self._lock_od = False
        self._spot_switch_streak = 2

        # Tuning
        self._p_margin_base = 0.04  # extra conservatism beyond Wilson LCB
        self._min_p_eff = 0.02

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _wilson_lower_bound(k: int, n: int, z: float = 1.64) -> float:
        if n <= 0:
            return 0.0
        phat = k / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = phat + z2 / (2.0 * n)
        rad = z * math.sqrt(max(0.0, (phat * (1.0 - phat) + z2 / (4.0 * n)) / n))
        return max(0.0, (center - rad) / denom)

    def _get_progress_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            return float(td)
        try:
            return float(sum(td))
        except Exception:
            return 0.0

    def _update_spot_stats(self, has_spot: bool, gap_seconds: float) -> None:
        self._n_obs += 1
        if has_spot:
            self._k_obs += 1

        if has_spot:
            self._spot_streak += 1
        else:
            self._spot_streak = 0

        if self._avg_spot_run_steps is None:
            # Initialize to something moderately long to avoid overreacting early.
            # If gap is very small, this is still okay since it's in steps.
            self._avg_spot_run_steps = max(3.0, (3600.0 / max(gap_seconds, 1.0)))  # ~1 hour

        # Run length tracking based purely on availability, not our chosen cluster.
        if self._last_has_spot is None:
            self._last_has_spot = has_spot
            self._curr_spot_run_steps = 1 if has_spot else 0
            return

        if has_spot:
            if self._last_has_spot:
                self._curr_spot_run_steps += 1
            else:
                # New spot run starts.
                self._curr_spot_run_steps = 1
        else:
            if self._last_has_spot:
                # Spot run ended; update EMA.
                run_len = float(self._curr_spot_run_steps)
                self._avg_spot_run_steps = (
                    (1.0 - self._run_ema_alpha) * float(self._avg_spot_run_steps)
                    + self._run_ema_alpha * run_len
                )
                self._curr_spot_run_steps = 0

        self._last_has_spot = has_spot

    def _should_hard_lock_od(self, time_remaining: float, remaining_work: float) -> bool:
        # Hard guard to avoid missing deadline due to preemptions and restart overheads.
        ro = float(getattr(self, "restart_overhead", 0.0))
        safety = max(3600.0, 3.0 * ro)  # at least 1 hour safety, or 3x restart overhead
        return time_remaining <= remaining_work + ro + safety

    def _effective_p(self, gap_seconds: float) -> float:
        p_lcb = self._wilson_lower_bound(self._k_obs, self._n_obs, z=1.64)  # ~90% LCB
        p_eff = p_lcb - self._p_margin_base

        # Penalize if spot is "flappy" (short average run lengths), because switching costs overhead.
        ro = float(getattr(self, "restart_overhead", 0.0))
        avg_steps = float(self._avg_spot_run_steps) if self._avg_spot_run_steps is not None else 0.0
        avg_run_seconds = avg_steps * max(gap_seconds, 1.0)

        if avg_run_seconds > 0.0 and ro > 0.0:
            # If average availability run is only a few restart overheads, reduce trust in spot.
            if avg_run_seconds < 6.0 * ro:
                p_eff *= 0.85
            if avg_run_seconds < 3.0 * ro:
                p_eff *= 0.80

        return max(self._min_p_eff, min(1.0, p_eff))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        self._update_spot_stats(has_spot, gap)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        time_remaining = max(0.0, deadline - elapsed)
        progress = self._get_progress_seconds()
        remaining_work = max(0.0, task_duration - progress)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        if not self._lock_od and self._should_hard_lock_od(time_remaining, remaining_work):
            self._lock_od = True

        if self._lock_od:
            return ClusterType.ON_DEMAND

        p_eff = self._effective_p(gap)

        # If we only compute when spot is available and pause otherwise, expected work we can do:
        # p_eff * time_remaining. If remaining_work exceeds this, we should use OD during spot-unavailable times.
        # Add a small buffer for restart overhead uncertainty.
        ro = float(getattr(self, "restart_overhead", 0.0))
        remaining_work_adj = remaining_work + 0.35 * ro
        need_od_during_unavail = remaining_work_adj > p_eff * time_remaining

        if has_spot:
            # Prefer spot when available.
            if last_cluster_type == ClusterType.ON_DEMAND:
                # Avoid frequent switches: only switch back to spot after a brief stable streak and if not too close.
                avg_steps = float(self._avg_spot_run_steps) if self._avg_spot_run_steps is not None else 0.0
                avg_run_seconds = avg_steps * max(gap, 1.0)
                safe_to_switch = time_remaining > remaining_work + max(5400.0, 4.0 * ro)  # >=1.5h + overhead buffer
                stable_enough = self._spot_streak >= self._spot_switch_streak
                long_runs = (avg_run_seconds >= 4.0 * ro) if ro > 0.0 else stable_enough

                if stable_enough and long_runs and safe_to_switch:
                    return ClusterType.SPOT
                # Otherwise, if we were already on OD, keep OD to avoid overhead churn.
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot not available.
        if need_od_during_unavail:
            return ClusterType.ON_DEMAND
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)