import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v2"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._initialized = False

        self._total_time = 0.0
        self._up_time = 0.0
        self._down_transitions = 0

        self._prev_has_spot: Optional[bool] = None
        self._use_spot_this_upspell = False
        self._upspell_credit = 0.0

        self._committed_od = False
        self._od_cooldown_steps = 0

        self._last_done = 0.0

    def solve(self, spec_path: str) -> "Solution":
        self._initialized = True
        return self

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", 0.0)
        if isinstance(tdt, (list, tuple)):
            s = 0.0
            for x in tdt:
                try:
                    s += float(x)
                except Exception:
                    pass
            return s
        try:
            return float(tdt)
        except Exception:
            return 0.0

    def _est_spot_availability(self) -> float:
        # Bayesian smoothing with a weak prior.
        p0 = 0.55
        prior_w = 4.0 * 3600.0
        denom = self._total_time + prior_w
        if denom <= 1e-9:
            return p0
        p = (self._up_time + p0 * prior_w) / denom
        if p < 0.01:
            return 0.01
        if p > 0.99:
            return 0.99
        return p

    def _est_mean_up_seconds(self) -> float:
        # Mean up duration (seconds) with smoothing.
        mean0 = 2.0 * 3600.0
        prior_transitions = 1.5
        denom = self._down_transitions + prior_transitions
        mu = (self._up_time + mean0 * prior_transitions) / denom
        if mu < 5.0 * 60.0:
            mu = 5.0 * 60.0
        if mu > 24.0 * 3600.0:
            mu = 24.0 * 3600.0
        return mu

    def _compute_safety(self, gap: float, H: float) -> float:
        # A small deterministic margin for discretization/uncertainty.
        return max(3.0 * gap, 2.0 * H, 15.0 * 60.0)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialized = True

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0.0:
            gap = 300.0

        # Update availability stats for this step.
        self._total_time += gap
        if has_spot:
            self._up_time += gap
        if self._prev_has_spot is True and has_spot is False:
            self._down_transitions += 1
        new_upspell = has_spot and (self._prev_has_spot is False or self._prev_has_spot is None)
        self._prev_has_spot = has_spot

        done = self._get_done_seconds()
        self._last_done = done

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        H = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        remaining_work = task_duration - done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        remaining_time = deadline - elapsed
        if remaining_time <= 0.0:
            # Deadline already passed; try to progress anyway.
            return ClusterType.ON_DEMAND

        slack = remaining_time - remaining_work
        safety = self._compute_safety(gap, H)

        # If we're in OD cooldown (to avoid resetting restart overhead), stick to OD.
        if self._od_cooldown_steps > 0:
            self._od_cooldown_steps -= 1
            return ClusterType.ON_DEMAND

        # If already committed to OD, stay there to avoid future preemption overhead.
        if self._committed_od:
            return ClusterType.ON_DEMAND

        # If slack is too low, commit to OD.
        if slack <= safety or remaining_time <= remaining_work + safety:
            self._committed_od = True
            self._use_spot_this_upspell = False
            return ClusterType.ON_DEMAND

        # If spot just got interrupted (we were on spot previously and now it's unavailable),
        # restart immediately on OD and keep OD for ~restart_overhead duration to avoid
        # overhead reset effects.
        if not has_spot and last_cluster_type == ClusterType.SPOT:
            if H > 0.0:
                self._od_cooldown_steps = int(math.ceil(H / gap)) + 1
            return ClusterType.ON_DEMAND

        # Estimate overhead budget and choose how often to "ride" upspells on spot.
        p = self._est_spot_availability()
        mean_up = self._est_mean_up_seconds()

        # Expected number of down transitions during remaining compute wall time if we were
        # to rely on spot whenever it's up:
        expected_interruptions_full = (p * remaining_work) / max(mean_up, 1.0)
        overhead_mean_full = expected_interruptions_full * H

        # Determine the fraction of upspells to use spot (0..1) so expected overhead fits in slack.
        overhead_budget = max(0.0, slack - safety)
        if overhead_mean_full <= 1e-9 or H <= 1e-9:
            target_q = 1.0
        else:
            target_q = overhead_budget / overhead_mean_full
            if target_q < 0.0:
                target_q = 0.0
            elif target_q > 1.0:
                target_q = 1.0

        # If spot is available, decide whether to use spot for the current upspell.
        if has_spot:
            if new_upspell:
                self._upspell_credit += target_q
                if self._upspell_credit >= 1.0:
                    self._upspell_credit -= 1.0
                    self._use_spot_this_upspell = True
                else:
                    self._use_spot_this_upspell = False

            # Bail out to OD if slack becomes too small while on spot.
            bailout = max(safety, 3.0 * H + 3.0 * gap)
            if self._use_spot_this_upspell and slack <= bailout:
                self._committed_od = True
                self._use_spot_this_upspell = False
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT if self._use_spot_this_upspell else ClusterType.ON_DEMAND

        # Spot not available: choose NONE if we have slack beyond expected overhead needs, else OD.
        self._use_spot_this_upspell = False
        expected_overhead = target_q * overhead_mean_full
        if slack - gap >= expected_overhead + safety:
            return ClusterType.NONE
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)