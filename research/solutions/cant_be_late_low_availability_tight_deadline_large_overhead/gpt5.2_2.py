import math
import statistics
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_adaptive_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.args = args

        self._committed_od = False

        self._steps = 0
        self._p_hat = 0.5  # EMA of has_spot
        self._ema_alpha = 0.02

        self._last_has_spot: Optional[bool] = None

        self._off_steps = 0
        self._on_steps = 0
        self._off_to_on = 0
        self._on_to_off = 0

        self._off_run = 0
        self._on_run = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _gap(self) -> float:
        g = getattr(getattr(self, "env", None), "gap_seconds", None)
        try:
            g = float(g)
        except Exception:
            g = 1.0
        if not math.isfinite(g) or g <= 0:
            g = 1.0
        return g

    def _elapsed(self) -> float:
        e = getattr(getattr(self, "env", None), "elapsed_seconds", None)
        try:
            e = float(e)
        except Exception:
            e = 0.0
        if not math.isfinite(e) or e < 0:
            e = 0.0
        return e

    def _done_work(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0
        if isinstance(tdt, (int, float)):
            v = float(tdt)
            return v if math.isfinite(v) and v > 0 else 0.0

        if isinstance(tdt, (list, tuple)):
            if not tdt:
                return 0.0

            gap = self._gap()

            if all(isinstance(x, (int, float)) for x in tdt):
                vals = [float(x) for x in tdt if x is not None and math.isfinite(float(x))]
                if not vals:
                    return 0.0
                last = vals[-1]
                # Heuristic: if values grow beyond a few gaps, likely cumulative-done timestamps.
                if last > 4.0 * gap:
                    return max(0.0, last)
                return max(0.0, sum(vals))

            total = 0.0
            for seg in tdt:
                if seg is None:
                    continue
                if isinstance(seg, (int, float)):
                    v = float(seg)
                    if math.isfinite(v) and v > 0:
                        total += v
                elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    a, b = seg[0], seg[1]
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        a = float(a)
                        b = float(b)
                        if math.isfinite(a) and math.isfinite(b):
                            total += max(0.0, b - a)
                elif isinstance(seg, dict):
                    if "duration" in seg:
                        try:
                            v = float(seg["duration"])
                            if math.isfinite(v) and v > 0:
                                total += v
                        except Exception:
                            pass
                    elif "start" in seg and "end" in seg:
                        try:
                            a = float(seg["start"])
                            b = float(seg["end"])
                            if math.isfinite(a) and math.isfinite(b):
                                total += max(0.0, b - a)
                        except Exception:
                            pass
            return max(0.0, total)

        return 0.0

    def _update_spot_stats(self, has_spot: bool) -> None:
        self._steps += 1

        x = 1.0 if has_spot else 0.0
        if self._steps == 1:
            self._p_hat = x
        else:
            a = self._ema_alpha
            self._p_hat = (1.0 - a) * self._p_hat + a * x

        if self._last_has_spot is not None:
            if (not self._last_has_spot) and has_spot:
                self._off_to_on += 1
            elif self._last_has_spot and (not has_spot):
                self._on_to_off += 1

        if has_spot:
            self._on_steps += 1
            self._on_run += 1
            self._off_run = 0
        else:
            self._off_steps += 1
            self._off_run += 1
            self._on_run = 0

        self._last_has_spot = has_spot

    def _q_off_to_on(self) -> float:
        # P(spot becomes available next step | currently off), with smoothing
        alpha = 1.0
        beta = 2.0
        return (self._off_to_on + alpha) / (self._off_steps + beta)

    def _should_commit_od_now(
        self,
        elapsed: float,
        gap: float,
        deadline: float,
        remaining: float,
        last_cluster_type: ClusterType,
        has_spot: bool,
    ) -> bool:
        if self._committed_od:
            return True

        if remaining <= 0.0:
            return False

        overhead = getattr(self, "restart_overhead", 0.0)
        try:
            overhead = float(overhead)
        except Exception:
            overhead = 0.0
        if not math.isfinite(overhead) or overhead < 0:
            overhead = 0.0

        # Conservative: if starting OD from anything other than OD, assume one restart overhead.
        od_start_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead

        od_latest_start = deadline - (remaining + od_start_overhead)

        # If we choose NONE now, next decision is after +gap; ensure we don't overshoot.
        hard_buffer = gap

        if elapsed + hard_buffer >= od_latest_start:
            return True

        # If spot is unavailable, consider expected waiting time risk.
        if not has_spot:
            q = self._q_off_to_on()
            q = max(1e-6, min(1.0, q))
            expected_wait = gap * (1.0 / q)  # conservative (includes the success step)

            # Make expected_wait more conservative when estimated availability is low.
            p = max(0.0, min(1.0, self._p_hat))
            risk_mult = 1.0 + 2.0 * max(0.0, 0.4 - p)  # up to +80% at p=0
            expected_wait *= risk_mult

            time_to_latest = od_latest_start - elapsed
            safety = max(gap, 0.25 * overhead)

            if expected_wait + safety >= time_to_latest:
                return True

            # Additional guard: don't wait through too many consecutive off steps near the end.
            off_run_time = self._off_run * gap
            if time_to_latest <= (2.0 * overhead + 3.0 * gap) and off_run_time >= gap:
                return True

        else:
            # If spot is available but we are in the final critical window, avoid interruption risk.
            overhead = getattr(self, "restart_overhead", 0.0)
            try:
                overhead = float(overhead)
            except Exception:
                overhead = 0.0
            if not math.isfinite(overhead) or overhead < 0:
                overhead = 0.0

            time_to_latest = od_latest_start - elapsed
            critical = overhead + 2.0 * gap
            if time_to_latest <= critical:
                return True

        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not has_spot:
            # Enforce API constraint.
            pass

        gap = self._gap()
        elapsed = self._elapsed()

        deadline = getattr(self, "deadline", 0.0)
        try:
            deadline = float(deadline)
        except Exception:
            deadline = 0.0
        if not math.isfinite(deadline) or deadline <= 0:
            deadline = elapsed + 1e18

        task_duration = getattr(self, "task_duration", 0.0)
        try:
            task_duration = float(task_duration)
        except Exception:
            task_duration = 0.0
        if not math.isfinite(task_duration) or task_duration < 0:
            task_duration = 0.0

        done = self._done_work()
        remaining = max(0.0, task_duration - done)

        self._update_spot_stats(has_spot)

        if remaining <= 0.0:
            return ClusterType.NONE

        if self._should_commit_od_now(elapsed, gap, deadline, remaining, last_cluster_type, has_spot):
            self._committed_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)