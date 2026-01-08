import math
from collections import deque
from typing import Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_adaptive_v2"

    def __init__(self, args=None):
        super().__init__(args)
        self._spot_hist: Optional[deque] = None
        self._spot_hist_window_seconds = 6 * 3600.0

        self._ewma_avail = 0.65
        self._ewma_instability = 0.0
        self._alpha = 0.03

        self._last_elapsed: Optional[float] = None
        self._idle_time = 0.0

        self._prev_done = 0.0
        self._force_od = False

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_done_work_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return 0.0

        if isinstance(tdt, (int, float)):
            v = float(tdt)
            return max(0.0, min(v, float(self.task_duration)))

        try:
            if len(tdt) == 0:
                return 0.0
        except Exception:
            return 0.0

        vals = []
        for x in tdt:
            if x is None:
                continue
            if isinstance(x, (int, float)):
                vals.append(float(x))
            elif isinstance(x, (tuple, list)) and len(x) > 0:
                if len(x) >= 2 and isinstance(x[0], (int, float)) and isinstance(x[1], (int, float)):
                    a = float(x[0])
                    b = float(x[1])
                    if b >= a and (b - a) <= float(self.task_duration) * 1.2:
                        vals.append(b - a)
                    else:
                        vals.append(b)
                else:
                    for y in x:
                        if isinstance(y, (int, float)):
                            vals.append(float(y))
                            break
            elif isinstance(x, dict):
                for k in ("duration", "work", "done", "time", "seconds"):
                    v = x.get(k, None)
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                        break

        if not vals:
            return 0.0

        last = float(vals[-1])
        total = float(sum(vals))
        nondecreasing = True
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                nondecreasing = False
                break

        td = float(self.task_duration)
        if nondecreasing and last <= td * 1.1 and total > last * 2.0:
            return max(0.0, min(last, td))

        return max(0.0, min(total, td))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 1.0))
        gap = max(gap, 1e-6)

        if self._spot_hist is None:
            w = int(round(self._spot_hist_window_seconds / gap))
            w = max(20, min(300, w))
            self._spot_hist = deque(maxlen=w)

        self._spot_hist.append(1 if has_spot else 0)

        # Update idle time using actual last_cluster_type (more reliable than our last action).
        if self._last_elapsed is None:
            self._last_elapsed = elapsed
        else:
            dt = elapsed - self._last_elapsed
            if dt > 0 and last_cluster_type == ClusterType.NONE:
                self._idle_time += dt
            self._last_elapsed = elapsed

        # Progress accounting
        done = self._get_done_work_seconds()
        self._prev_done = done

        remaining_work = max(0.0, float(self.task_duration) - done)
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        time_left = float(self.deadline) - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        slack = time_left - remaining_work

        # Spot availability stats over recent window
        hist_len = len(self._spot_hist)
        mean_avail = (sum(self._spot_hist) / hist_len) if hist_len > 0 else 0.0
        transitions = 0
        if hist_len >= 2:
            prev = self._spot_hist[0]
            for v in list(self._spot_hist)[1:]:
                if v != prev:
                    transitions += 1
                prev = v
            instability = transitions / (hist_len - 1)
        else:
            instability = 0.0

        # EWMA stats
        x = 1.0 if has_spot else 0.0
        a = self._alpha
        self._ewma_avail = (1.0 - a) * self._ewma_avail + a * x
        self._ewma_instability = (1.0 - a) * self._ewma_instability + a * instability

        # Estimate time lost to overheads so far (everything not work and not explicit NONE).
        overhead_spent = max(0.0, elapsed - self._idle_time - done)
        active_time = max(1.0, elapsed - self._idle_time)
        overhead_ratio = min(0.9, overhead_spent / active_time)

        oh = float(getattr(self, "restart_overhead", 0.0))

        # Safety reserve (slack we want to keep to absorb future overhead/instability/discretization).
        base_reserve = max(2.0 * oh + 2.0 * gap, 2.0 * gap, 600.0)
        extra_reserve = 0.0
        if hist_len >= 10:
            extra_reserve += max(0.0, 0.50 - mean_avail) * 5.0 * 3600.0  # up to 2.5h
            extra_reserve += instability * 3.0 * 3600.0  # up to 3h
            extra_reserve += overhead_ratio * 4.0 * 3600.0  # up to 3.6h
        reserve = base_reserve + extra_reserve

        # Force stable on-demand near the end if spot is unstable and we cannot afford wasted time.
        force_slack = max(2.0 * 3600.0, 12.0 * oh + 8.0 * gap)
        if not self._force_od and slack <= force_slack and hist_len >= 20:
            if (mean_avail < 0.45 and instability > 0.20) or (overhead_ratio > 0.18 and instability > 0.25):
                self._force_od = True

        if self._force_od:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # No spot available: pause if we can afford it, else use on-demand.
        if slack > reserve:
            return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)