from __future__ import annotations

from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_jit_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass

        self._inited = False

        self._last_elapsed: Optional[float] = None
        self._last_choice: ClusterType = ClusterType.NONE

        self._overhead_remaining: float = 0.0
        self._prog_done: float = 0.0

        self._prev_has_spot: Optional[bool] = None
        self._total_steps: int = 0
        self._spot_steps: int = 0

        self._cur_up_run: float = 0.0
        self._cur_down_run: float = 0.0
        self._ewma_up_run: float = 3600.0  # prior: 1 hour
        self._ewma_down_run: float = 3600.0
        self._ewma_alpha: float = 0.15

        self._mode_start_elapsed: float = 0.0
        self._mode: ClusterType = ClusterType.NONE

        self._last_done_task: float = 0.0

        self._safety_const: float = 900.0  # 15 minutes

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            v = float(x)
            if v != v:  # NaN
                return None
            return v
        except Exception:
            return None

    def _parse_task_done_time_underestimate(self) -> Optional[float]:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0
        try:
            if isinstance(tdt, (int, float)):
                v = self._safe_float(tdt)
                if v is None:
                    return None
                return max(0.0, v)
            if not isinstance(tdt, (list, tuple)):
                return None
            if len(tdt) == 0:
                return 0.0

            # If segments as tuples/dicts: sum durations
            first = tdt[0]
            if isinstance(first, (tuple, list)) and len(first) >= 2:
                s = 0.0
                for seg in tdt:
                    if not (isinstance(seg, (tuple, list)) and len(seg) >= 2):
                        continue
                    a = self._safe_float(seg[0])
                    b = self._safe_float(seg[1])
                    if a is None or b is None:
                        continue
                    if b > a:
                        s += (b - a)
                return max(0.0, s)

            if isinstance(first, dict):
                s = 0.0
                for seg in tdt:
                    if not isinstance(seg, dict):
                        continue
                    a = self._safe_float(seg.get("start", None))
                    b = self._safe_float(seg.get("end", None))
                    if a is None or b is None:
                        d = self._safe_float(seg.get("duration", None))
                        if d is not None and d > 0:
                            s += d
                        continue
                    if b > a:
                        s += (b - a)
                return max(0.0, s)

            # Numeric list: could be segment durations or cumulative totals.
            vals = []
            for x in tdt:
                v = self._safe_float(x)
                if v is None:
                    continue
                if v >= 0:
                    vals.append(v)
            if not vals:
                return 0.0

            task_dur = self._safe_float(getattr(self, "task_duration", None))
            if task_dur is None or task_dur <= 0:
                # Conservative
                return 0.0

            sumv = float(sum(vals))
            last = float(vals[-1])
            is_nondec = True
            for i in range(1, len(vals)):
                if vals[i] + 1e-9 < vals[i - 1]:
                    is_nondec = False
                    break

            # If any value is far beyond task duration, likely timestamps; don't trust.
            if max(vals) > task_dur * 1.2:
                return None

            # If sum is within task duration, likely segment durations.
            if sumv <= task_dur * 1.05:
                return sumv

            # If looks like cumulative done-work.
            if is_nondec and last <= task_dur * 1.05:
                return last

            # Ambiguous; choose a conservative lower bound.
            # Taking min(sum, last) is conservative vs segment durations (but could be too small).
            return min(sumv, last)
        except Exception:
            return None

    def _update_internal_state(self, last_cluster_type: ClusterType, cur_has_spot: bool) -> None:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        if self._last_elapsed is None:
            self._last_elapsed = elapsed
            self._last_choice = last_cluster_type
            self._mode = last_cluster_type
            self._mode_start_elapsed = elapsed
            self._prev_has_spot = cur_has_spot
            return

        dt = elapsed - self._last_elapsed
        if dt < 0:
            dt = 0.0

        # Update availability stats for the *previous* step using stored prev_has_spot
        if self._prev_has_spot is not None and dt > 0:
            self._total_steps += 1
            if self._prev_has_spot:
                self._spot_steps += 1

            # Run-length tracking based on previous-step availability
            if self._prev_has_spot:
                self._cur_up_run += dt
                if not cur_has_spot:
                    # ended an up-run at boundary
                    self._ewma_up_run = (1.0 - self._ewma_alpha) * self._ewma_up_run + self._ewma_alpha * max(
                        dt, self._cur_up_run
                    )
                    self._cur_up_run = 0.0
            else:
                self._cur_down_run += dt
                if cur_has_spot:
                    self._ewma_down_run = (1.0 - self._ewma_alpha) * self._ewma_down_run + self._ewma_alpha * max(
                        dt, self._cur_down_run
                    )
                    self._cur_down_run = 0.0

        # Update conservative progress estimate from the previous step
        if dt > 0 and self._last_choice != ClusterType.NONE:
            if self._overhead_remaining >= dt:
                self._overhead_remaining -= dt
                prog = 0.0
            else:
                prog = dt - self._overhead_remaining
                self._overhead_remaining = 0.0
            if prog > 0:
                self._prog_done += prog

        self._last_elapsed = elapsed
        self._prev_has_spot = cur_has_spot

    def _get_done_work_seconds(self) -> float:
        task_dur = self._safe_float(getattr(self, "task_duration", None))
        if task_dur is None or task_dur <= 0:
            task_dur = 0.0

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))

        parsed = self._parse_task_done_time_underestimate()
        if parsed is not None:
            parsed = max(0.0, float(parsed))
            # basic plausibility checks (avoid timestamp misinterpretation)
            if parsed <= task_dur + 1e-6 and parsed <= elapsed + 1e-6 and parsed + 1e-6 >= self._last_done_task:
                self._last_done_task = parsed
                # keep internal estimate in sync as a lower bound too
                if self._prog_done > parsed:
                    self._prog_done = parsed
                return min(task_dur, parsed)

        # fallback: conservative internal estimate
        done = max(0.0, float(self._prog_done))
        done = min(done, task_dur)
        if done < self._last_done_task:
            # keep monotonic
            done = self._last_done_task
        else:
            self._last_done_task = done
        return done

    def _availability_p_hat(self) -> float:
        if self._total_steps <= 0:
            return 0.5
        return max(0.01, min(0.99, self._spot_steps / float(self._total_steps)))

    def _buffers(self) -> tuple[float, float, float, float]:
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        ro = float(getattr(self, "restart_overhead", 0.0))
        p = self._availability_p_hat()

        b_crit = max(ro + 2.0 * gap, 2.0 * gap + self._safety_const)
        b_wait = max(1.5 * ro + 2.0 * gap + self._safety_const, b_crit + 0.5 * ro)
        # volatility-aware switching: need longer expected up-run when p is low
        min_up_switch = max(2.0 * gap, ro) * (1.0 + (1.0 - p))
        min_od_run = max(2.0 * gap, 0.5 * ro)

        return b_crit, b_wait, min_up_switch, min_od_run

    def _should_switch_to_spot(self, slack: float, min_up_switch: float, min_od_run: float) -> bool:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        ro = float(getattr(self, "restart_overhead", 0.0))

        # Avoid immediate switch right after starting on-demand (likely paid/undergoing overhead)
        if self._mode == ClusterType.ON_DEMAND and (elapsed - self._mode_start_elapsed) < min_od_run:
            return False

        # If spot up-runs are typically too short, don't thrash (unless slack is very large)
        if self._ewma_up_run >= min_up_switch:
            return True

        # If we have plenty of slack, we can try spot even with short runs
        if slack >= (3.0 * ro + self._safety_const):
            return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_internal_state(last_cluster_type, has_spot)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        ro = float(getattr(self, "restart_overhead", 0.0))

        done = self._get_done_work_seconds()
        task_dur = float(getattr(self, "task_duration", 0.0))
        remaining_work = max(0.0, task_dur - done)
        remaining_time = max(0.0, deadline - elapsed)
        slack = remaining_time - remaining_work

        if remaining_work <= 1e-6:
            nxt = ClusterType.NONE
            self._apply_choice(nxt, last_cluster_type, has_spot)
            return nxt

        b_crit, b_wait, min_up_switch, min_od_run = self._buffers()

        # If effectively out of time, run on-demand.
        if remaining_time <= gap * 0.5:
            nxt = ClusterType.ON_DEMAND
            self._apply_choice(nxt, last_cluster_type, has_spot)
            return nxt

        # Hard critical region: avoid preemptions/overhead risks.
        if slack <= b_crit:
            nxt = ClusterType.ON_DEMAND
            self._apply_choice(nxt, last_cluster_type, has_spot)
            return nxt

        # Non-critical: prefer spot when available.
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND:
                nxt = ClusterType.SPOT if self._should_switch_to_spot(slack, min_up_switch, min_od_run) else ClusterType.ON_DEMAND
            else:
                nxt = ClusterType.SPOT
            self._apply_choice(nxt, last_cluster_type, has_spot)
            return nxt

        # No spot this step: wait if we have enough slack, else use on-demand.
        if slack >= b_wait:
            nxt = ClusterType.NONE
        else:
            nxt = ClusterType.ON_DEMAND

        self._apply_choice(nxt, last_cluster_type, has_spot)
        return nxt

    def _apply_choice(self, nxt: ClusterType, last_cluster_type: ClusterType, has_spot: bool) -> None:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        ro = float(getattr(self, "restart_overhead", 0.0))

        # Track mode switches
        if nxt != self._mode:
            self._mode = nxt
            self._mode_start_elapsed = elapsed

        preempted = (last_cluster_type == ClusterType.SPOT and (not has_spot))
        switched = (nxt != last_cluster_type)

        if nxt == ClusterType.NONE:
            # No instance: overhead doesn't progress; treat as no pending overhead on a stopped cluster.
            self._overhead_remaining = 0.0
        else:
            need_restart = preempted or switched or (last_cluster_type == ClusterType.NONE)
            if need_restart:
                self._overhead_remaining = ro

        self._last_choice = nxt

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)