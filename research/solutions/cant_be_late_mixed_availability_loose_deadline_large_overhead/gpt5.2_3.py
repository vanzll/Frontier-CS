import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_sched_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except Exception:
            pass
        self.args = args

        self._last_elapsed: Optional[float] = None
        self._overhead_remaining: float = 0.0

        self._prev_has_spot: Optional[bool] = None
        self._avail_total: int = 0
        self._avail_count: int = 0
        self._drop_count: int = 0

        self._streak_len: int = 0
        self._streak_total: int = 0
        self._streak_count: int = 0

        self._od_lock: bool = False
        self._od_cooldown_until: float = 0.0

        self._last_action: Optional[ClusterType] = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            return None
        return None

    def _get_work_done(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        total = 0.0
        for item in tdt:
            if isinstance(item, (int, float)):
                v = float(item)
                if math.isfinite(v) and v > 0:
                    total += v
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                a = self._safe_float(item[0])
                b = self._safe_float(item[1])
                if a is not None and b is not None:
                    d = b - a
                    if d > 0:
                        total += d
                continue

            if isinstance(item, dict):
                if "duration" in item:
                    v = self._safe_float(item.get("duration"))
                    if v is not None and v > 0:
                        total += v
                else:
                    a = self._safe_float(item.get("start"))
                    b = self._safe_float(item.get("end"))
                    if a is not None and b is not None:
                        d = b - a
                        if d > 0:
                            total += d
                continue

        td = getattr(self, "task_duration", None)
        if isinstance(td, (int, float)) and math.isfinite(float(td)) and total > float(td):
            total = float(td)
        return total

    def _update_availability_stats(self, has_spot: bool) -> None:
        self._avail_total += 1
        if has_spot:
            self._avail_count += 1

        prev = self._prev_has_spot
        if prev is True:
            if has_spot:
                if self._streak_len <= 0:
                    self._streak_len = 1
                else:
                    self._streak_len += 1
            else:
                self._drop_count += 1
                if self._streak_len > 0:
                    self._streak_total += self._streak_len
                    self._streak_count += 1
                self._streak_len = 0
        else:
            if has_spot:
                self._streak_len = 1
            else:
                self._streak_len = 0

        self._prev_has_spot = has_spot

    def _get_estimates(self, gap_seconds: float) -> tuple[float, float, float]:
        # Availability probability with Laplace smoothing
        p_avail = (self._avail_count + 1.0) / (self._avail_total + 2.0)

        # Drop probability conditioned on being available previous step (Laplace smoothing)
        denom = (self._avail_count + 2.0)
        p_drop = (self._drop_count + 1.0) / denom
        p_drop = min(max(p_drop, 1e-4), 0.999)

        # Average streak length (include ongoing streak as a partial observation)
        streak_sum = float(self._streak_total + (self._streak_len if self._streak_len > 0 else 0))
        streak_n = float(self._streak_count + (1 if self._streak_len > 0 else 0))
        if streak_n <= 0:
            avg_streak_steps = 1.0 / max(p_drop, 1e-3)
        else:
            avg_streak_steps = max(1.0, streak_sum / streak_n)

        avg_streak_time = avg_streak_steps * max(gap_seconds, 1e-6)
        return p_avail, p_drop, avg_streak_time

    def _overhead_if_choose(self, last_cluster_type: ClusterType, choose: ClusterType) -> float:
        if choose == ClusterType.NONE:
            return 0.0
        if choose != last_cluster_type:
            return float(getattr(self, "restart_overhead", 0.0) or 0.0)
        return float(self._overhead_remaining)

    def _apply_action_overhead(self, last_cluster_type: ClusterType, action: ClusterType) -> None:
        if action == ClusterType.NONE:
            self._overhead_remaining = 0.0
            return
        if action != last_cluster_type:
            self._overhead_remaining = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        # else keep current remaining overhead

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = getattr(self, "env", None)
        elapsed = float(getattr(env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(env, "gap_seconds", 0.0) or 0.0)
        gap = max(gap, 1e-6)

        # First-call initialization
        if self._last_elapsed is None:
            self._last_elapsed = elapsed
            self._prev_has_spot = None
            self._last_action = last_cluster_type
            self._update_availability_stats(bool(has_spot))
        else:
            delta = elapsed - self._last_elapsed
            if not math.isfinite(delta) or delta < 0:
                delta = gap
            self._overhead_remaining = max(0.0, self._overhead_remaining - delta)
            self._last_elapsed = elapsed
            self._update_availability_stats(bool(has_spot))

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart_overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._get_work_done()
        remaining = max(0.0, task_duration - done)
        time_left = max(0.0, deadline - elapsed)

        if remaining <= 1e-6:
            action = ClusterType.NONE
            self._apply_action_overhead(last_cluster_type, action)
            self._last_action = action
            return action

        p_avail, p_drop, avg_streak_time = self._get_estimates(gap)

        # Expected times (simple geometric assumptions), capped
        expected_wait = min(12.0 * 3600.0, gap / max(p_avail, 0.02))
        expected_rem_avail = min(12.0 * 3600.0, gap / max(p_drop, 0.02))

        # Effective spot productivity estimate (availability scaled by amortized overhead)
        amort = 1.0 - min(0.95, restart_overhead / max(avg_streak_time, restart_overhead))
        eff_spot_prod = max(0.0, min(1.0, p_avail * amort))

        slack = time_left - remaining
        req_util = remaining / max(time_left, 1e-9)

        # If OD from now cannot finish, OD is the best chance anyway.
        od_overhead_now = self._overhead_if_choose(last_cluster_type, ClusterType.ON_DEMAND)
        if remaining + od_overhead_now > time_left + 1e-9:
            self._od_lock = True

        # Lock to OD when slack is small (avoid last-minute interruptions/extra overhead)
        lock_slack = 2.5 * restart_overhead + 2.0 * gap
        if slack <= lock_slack:
            self._od_lock = True

        in_od_cooldown = elapsed < self._od_cooldown_until

        action: ClusterType

        if self._od_lock:
            action = ClusterType.ON_DEMAND
        else:
            if has_spot:
                if last_cluster_type == ClusterType.SPOT:
                    # If slack is getting tight, switch to OD before we risk a preemption.
                    if slack <= (restart_overhead + gap):
                        self._od_lock = True
                        action = ClusterType.ON_DEMAND
                    else:
                        action = ClusterType.SPOT
                else:
                    if in_od_cooldown:
                        action = ClusterType.ON_DEMAND
                    else:
                        flaky = (avg_streak_time < 1.2 * restart_overhead) or (expected_rem_avail < 1.1 * restart_overhead)
                        # Need enough slack to absorb: start overhead + potential fallback overhead
                        need_slack = 2.0 * restart_overhead + gap
                        if slack >= need_slack and (not flaky or slack >= 6.0 * restart_overhead):
                            action = ClusterType.SPOT
                        else:
                            action = ClusterType.ON_DEMAND
            else:
                # No spot available: decide to wait (NONE) or use OD
                # If we can't afford waiting given required utilization and effective spot productivity, use OD.
                if req_util > eff_spot_prod * 0.98:
                    action = ClusterType.ON_DEMAND
                else:
                    # Wait if we have ample slack to cover expected wait + restart overhead + one extra step.
                    min_slack_to_wait = expected_wait + restart_overhead + gap
                    if slack >= min_slack_to_wait:
                        action = ClusterType.NONE
                    else:
                        action = ClusterType.ON_DEMAND

        # Set OD cooldown when we switch into OD (reduce thrashing)
        if action == ClusterType.ON_DEMAND and last_cluster_type != ClusterType.ON_DEMAND:
            cooldown = max(3600.0, 2.0 * restart_overhead)
            self._od_cooldown_until = max(self._od_cooldown_until, elapsed + cooldown)

        # Apply overhead model for the next interval
        self._apply_action_overhead(last_cluster_type, action)
        self._last_action = action
        return action

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)