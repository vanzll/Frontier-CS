import math
import json
import os
from collections import deque

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:
    from enum import Enum

    class ClusterType(Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, *args, **kwargs):
            self.env = None


class Solution(Strategy):
    NAME = "cant_be_late_adaptive_v1"

    def __init__(self, args=None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args
        self._reset_internal_state()

    def _reset_internal_state(self):
        self._t = 0
        self._s = 0
        self._recent = deque()
        self._recent_maxlen = 240
        self._recent_sum = 0

        self._prev_has_spot = None
        self._nn = 0
        self._ns = 0
        self._sn = 0
        self._ss = 0

        self._consec_no_spot = 0
        self._consec_spot = 0

        self._committed_on_demand = False

        self._z = 1.28  # ~90% one-sided LCB
        self._prior_a = 6.0
        self._prior_b = 4.0

        self._min_pause_slack_seconds = 0.0

    def solve(self, spec_path: str) -> "Solution":
        self._reset_internal_state()
        if spec_path and isinstance(spec_path, str) and os.path.exists(spec_path):
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    txt = f.read()
                cfg = None
                try:
                    cfg = json.loads(txt)
                except Exception:
                    cfg = None
                if isinstance(cfg, dict):
                    z = cfg.get("z_lcb", None)
                    if isinstance(z, (int, float)) and 0.0 < z < 5.0:
                        self._z = float(z)
                    a = cfg.get("prior_a", None)
                    b = cfg.get("prior_b", None)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b > 0:
                        self._prior_a = float(a)
                        self._prior_b = float(b)
                    rml = cfg.get("recent_window", None)
                    if isinstance(rml, int) and 20 <= rml <= 2000:
                        self._recent_maxlen = int(rml)
                        self._recent = deque(maxlen=self._recent_maxlen)
                        self._recent_sum = 0
                    mps = cfg.get("min_pause_slack_seconds", None)
                    if isinstance(mps, (int, float)) and mps >= 0:
                        self._min_pause_slack_seconds = float(mps)
            except Exception:
                pass
        return self

    @staticmethod
    def _clamp(x, lo, hi):
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _compute_done_seconds(self):
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0
        if isinstance(tdt, (int, float)):
            return float(tdt)
        if not isinstance(tdt, (list, tuple)):
            return 0.0

        total = 0.0
        for seg in tdt:
            if seg is None:
                continue
            if isinstance(seg, (int, float)):
                total += float(seg)
            elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                a, b = seg[0], seg[1]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    dt = float(b) - float(a)
                    if dt > 0:
                        total += dt
            else:
                try:
                    dur = float(getattr(seg, "duration"))
                    if dur > 0:
                        total += dur
                except Exception:
                    pass
        return total

    def _update_availability_stats(self, has_spot: bool):
        hs = bool(has_spot)
        self._t += 1
        if hs:
            self._s += 1

        if hs:
            self._consec_spot += 1
            self._consec_no_spot = 0
        else:
            self._consec_no_spot += 1
            self._consec_spot = 0

        if self._prev_has_spot is not None:
            prev = self._prev_has_spot
            if prev and hs:
                self._ss += 1
            elif prev and (not hs):
                self._sn += 1
            elif (not prev) and hs:
                self._ns += 1
            else:
                self._nn += 1
        self._prev_has_spot = hs

        if self._recent_maxlen > 0:
            if len(self._recent) == self._recent_maxlen:
                old = self._recent[0]
                self._recent_sum -= old
            self._recent.append(1 if hs else 0)
            self._recent_sum += 1 if hs else 0

    def _beta_lcb(self, s, t, a, b):
        denom = t + a + b
        if denom <= 0:
            return 0.5
        p = (s + a) / denom
        var = p * (1.0 - p) / (denom + 1.0)
        lcb = p - self._z * math.sqrt(max(0.0, var))
        return max(0.0, min(1.0, lcb))

    def _estimate_p_safe(self):
        p_g = self._beta_lcb(self._s, self._t, self._prior_a, self._prior_b)
        t_r = len(self._recent)
        if t_r >= 20:
            s_r = self._recent_sum
            p_r = self._beta_lcb(s_r, t_r, self._prior_a, self._prior_b)
            p = min(p_g, p_r)
        else:
            p = p_g

        # Conservative shrink; keep a reasonable minimum to avoid pathological early behavior
        p = max(0.05, min(0.99, p * 0.95))
        return p

    def _expected_wait_steps_no_spot(self):
        # Expected steps until spot returns given currently no-spot; use smoothed transition prob.
        # return_prob = P(spot | no-spot)
        denom = self._nn + self._ns + 2.0
        return_prob = (self._ns + 1.0) / denom
        return_prob = self._clamp(return_prob, 0.02, 0.98)
        exp_steps = 1.0 / return_prob
        # If outage has lasted, extend expected wait.
        exp_steps += min(20.0, 0.35 * float(self._consec_no_spot))
        return exp_steps

    def _expected_run_steps_spot(self):
        # Expected spot run length given currently spot; use smoothed stay prob.
        denom = self._ss + self._sn + 2.0
        stay_prob = (self._ss + 1.0) / denom
        stay_prob = self._clamp(stay_prob, 0.02, 0.98)
        exp_steps = 1.0 / (1.0 - stay_prob)
        return exp_steps

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_availability_stats(has_spot)

        gap = float(getattr(self.env, "gap_seconds", 60.0))
        now = float(getattr(self.env, "elapsed_seconds", 0.0))

        done = self._compute_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0))
        remaining = max(0.0, task_duration - done)

        if remaining <= 1e-6:
            return ClusterType.NONE

        deadline = float(getattr(self, "deadline", now))
        time_left = deadline - now
        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        restart_overhead = float(getattr(self, "restart_overhead", 0.0))

        # Hard commit checks (deadline safety)
        # Buffer accounts for uncertainty, overheads, and step granularity.
        buffer = max(6.0 * gap, restart_overhead, 600.0)
        if time_left <= (remaining + restart_overhead + buffer):
            self._committed_on_demand = True
        slack = time_left - (remaining + restart_overhead)
        if slack < 0.0:
            self._committed_on_demand = True
        if slack <= self._min_pause_slack_seconds:
            self._committed_on_demand = True

        if self._committed_on_demand:
            return ClusterType.ON_DEMAND

        p_safe = self._estimate_p_safe()
        required_rate = remaining / max(time_left, 1e-9)

        # Urgency in [0,1], higher = closer to needing full utilization
        slack_norm = slack / (8.0 * 3600.0)
        urgency = 1.0 - self._clamp(slack_norm, 0.0, 1.0)
        safety = 0.10 + 0.20 * urgency  # tighten as we get closer

        if has_spot:
            # If we are on on-demand already, avoid switching to spot if we're tight on time
            # or if spot just came back and looks unstable (reduce restart thrash).
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack <= (restart_overhead + 3.0 * gap):
                    return ClusterType.ON_DEMAND
                if self._consec_spot < 2:
                    exp_run = self._expected_run_steps_spot()
                    if exp_run < 3.0:
                        return ClusterType.ON_DEMAND

            # If we're extremely tight (need almost continuous compute), prefer on-demand for stability.
            if required_rate > 0.92 and slack < (restart_overhead + 2.0 * gap):
                return ClusterType.ON_DEMAND

            return ClusterType.SPOT

        # No spot available: decide between pausing and on-demand.
        # If we can still finish relying mostly on spot when it returns, pause to save cost.
        can_pause_by_rate = (required_rate <= p_safe * (1.0 - safety)) and (slack > 0.0)

        if can_pause_by_rate:
            exp_wait_steps = self._expected_wait_steps_no_spot()
            exp_wait_seconds = exp_wait_steps * gap
            # Only pause if we can afford the expected wait with margin.
            if slack > (exp_wait_seconds + max(2.0 * gap, 0.5 * restart_overhead)):
                return ClusterType.NONE

        # Otherwise, use on-demand during outage to avoid falling behind.
        # Soft-commit if we are getting close to the point where any further waiting is dangerous.
        if slack < 3600.0 or time_left < 12.0 * 3600.0 or required_rate > 0.85 or self._consec_no_spot >= 24:
            # Not a strict lock; but helps prevent oscillation late in the run.
            if time_left < 10.0 * 3600.0 or slack < 1800.0 or required_rate > 0.9:
                self._committed_on_demand = True

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)