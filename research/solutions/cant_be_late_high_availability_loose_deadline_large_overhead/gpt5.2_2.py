import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v2"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args

        self._steps_seen = 0
        self._spot_seen = 0

        self._last_has_spot: Optional[bool] = None
        self._down_events = 0

        self._cooldown_until = -1.0
        self._cooldown_type: Optional[ClusterType] = None

        self._outage_start_elapsed: Optional[float] = None

        self._lock_on_demand = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_sum_done(task_done_time: Any, task_duration: float) -> float:
        if not task_done_time:
            return 0.0
        try:
            first = task_done_time[0]
        except Exception:
            return 0.0

        done = 0.0
        try:
            if isinstance(first, (int, float)):
                vals = []
                for x in task_done_time:
                    if isinstance(x, (int, float)) and math.isfinite(float(x)):
                        vals.append(float(x))
                if not vals:
                    return 0.0
                s = float(sum(vals))
                mx = float(max(vals))
                # Heuristic: if entries look cumulative (sum far exceeds duration but max is plausible), use max.
                if task_duration > 0 and s > 1.2 * task_duration and mx <= 1.05 * task_duration:
                    done = mx
                else:
                    done = s
            elif isinstance(first, (tuple, list)) and len(first) >= 2:
                s = 0.0
                for seg in task_done_time:
                    if not isinstance(seg, (tuple, list)) or len(seg) < 2:
                        continue
                    a, b = seg[0], seg[1]
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        da = float(a)
                        db = float(b)
                        if math.isfinite(da) and math.isfinite(db) and db >= da:
                            s += (db - da)
                done = s
            else:
                done = 0.0
        except Exception:
            done = 0.0

        if not math.isfinite(done):
            done = 0.0
        if task_duration > 0:
            done = max(0.0, min(done, float(task_duration)))
        else:
            done = max(0.0, done)
        return done

    def _update_spot_stats(self, has_spot: bool) -> None:
        self._steps_seen += 1
        if has_spot:
            self._spot_seen += 1
        if self._last_has_spot is True and not has_spot:
            self._down_events += 1
        self._last_has_spot = has_spot

        if has_spot:
            self._outage_start_elapsed = None
        else:
            if self._outage_start_elapsed is None:
                try:
                    self._outage_start_elapsed = float(self.env.elapsed_seconds)
                except Exception:
                    self._outage_start_elapsed = 0.0

    def _spot_availability_estimate(self) -> float:
        # Bayesian smoothing prior to avoid extreme estimates early on.
        prior_p = 0.65
        prior_strength = 30.0
        denom = self._steps_seen + prior_strength
        if denom <= 0:
            return prior_p
        p = (self._spot_seen + prior_p * prior_strength) / denom
        return float(min(0.98, max(0.05, p)))

    def _maybe_enter_lock_on_demand(self, slack: float, time_left: float, work_left: float, gap: float, ro: float) -> None:
        if self._lock_on_demand:
            return
        tight = max(2.0 * ro + 2.0 * gap, 3600.0)  # >= 1 hour
        if time_left <= work_left + tight:
            self._lock_on_demand = True
            return
        od_lock_slack = max(6.0 * ro, 2.0 * 3600.0)  # >= 2 hours
        if slack <= od_lock_slack:
            self._lock_on_demand = True

    def _apply_cooldown(self, has_spot: bool) -> Optional[ClusterType]:
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0

        if self._cooldown_type is None:
            return None
        if elapsed < self._cooldown_until:
            if self._cooldown_type == ClusterType.SPOT and not has_spot:
                self._cooldown_type = None
                self._cooldown_until = -1.0
                return None
            return self._cooldown_type

        self._cooldown_type = None
        self._cooldown_until = -1.0
        return None

    def _set_cooldown_on_switch(self, last_cluster_type: ClusterType, new_cluster_type: ClusterType) -> None:
        if new_cluster_type == ClusterType.NONE:
            return
        if new_cluster_type == last_cluster_type:
            return
        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        if ro <= 0:
            return
        self._cooldown_type = new_cluster_type
        self._cooldown_until = elapsed + ro

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_spot_stats(has_spot)

        cooled = self._apply_cooldown(has_spot)
        if cooled is not None:
            return cooled

        try:
            gap = float(self.env.gap_seconds)
        except Exception:
            gap = 0.0
        if gap <= 0:
            gap = 300.0

        try:
            elapsed = float(self.env.elapsed_seconds)
        except Exception:
            elapsed = 0.0

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._safe_sum_done(getattr(self, "task_done_time", None), task_duration)
        work_left = max(0.0, task_duration - done)
        time_left = max(0.0, deadline - elapsed)
        slack = time_left - work_left

        if work_left <= 0:
            return ClusterType.NONE

        base_margin = max(2.0 * ro + 2.0 * gap, 900.0)  # >= 15 minutes

        # If already behind or essentially no slack, go on-demand and stay there.
        if slack <= 0.0:
            self._lock_on_demand = True
            decision = ClusterType.ON_DEMAND
            self._set_cooldown_on_switch(last_cluster_type, decision)
            return decision

        p = self._spot_availability_estimate()

        self._maybe_enter_lock_on_demand(slack, time_left, work_left, gap, ro)
        if self._lock_on_demand:
            decision = ClusterType.ON_DEMAND
            self._set_cooldown_on_switch(last_cluster_type, decision)
            return decision

        # Decide whether we can afford to wait for spot during outages.
        # If time_left is close to the expected wall time using only spot-available slots, avoid waiting.
        spot_only_wall = work_left / max(p, 0.15)
        spot_slack = time_left - spot_only_wall
        avoid_wait = spot_slack <= base_margin

        # Also avoid switching back to spot from OD when slack is low.
        switch_back_need_slack = max(3.0 * ro + 2.0 * gap, 3600.0)  # >= 1 hour

        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and slack <= switch_back_need_slack:
                decision = ClusterType.ON_DEMAND
            else:
                # If extremely tight, prefer OD even if spot exists.
                if time_left <= work_left + base_margin:
                    decision = ClusterType.ON_DEMAND
                else:
                    decision = ClusterType.SPOT
        else:
            # Spot not available.
            if avoid_wait:
                decision = ClusterType.ON_DEMAND
            else:
                expected_wait = gap / max(p, 0.05)
                required = expected_wait + base_margin

                outage_waited = 0.0
                if self._outage_start_elapsed is not None:
                    outage_waited = max(0.0, elapsed - float(self._outage_start_elapsed))

                # Cap waiting within a single outage to reduce risk of very long gaps.
                max_outage_wait = min(6.0 * 3600.0, max(0.0, slack - base_margin))
                if outage_waited >= max_outage_wait and max_outage_wait > 0:
                    decision = ClusterType.ON_DEMAND
                else:
                    decision = ClusterType.NONE if slack > required else ClusterType.ON_DEMAND

        if decision == ClusterType.SPOT and not has_spot:
            decision = ClusterType.ON_DEMAND

        self._set_cooldown_on_switch(last_cluster_type, decision)
        return decision

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)