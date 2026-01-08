import math
from typing import Any, Optional

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except Exception:  # pragma: no cover
    from enum import Enum

    class ClusterType(str, Enum):
        SPOT = "spot"
        ON_DEMAND = "on_demand"
        NONE = "none"

    class Strategy:
        def __init__(self, *args, **kwargs):
            self.env = None
            self.task_duration = 0.0
            self.task_done_time = []
            self.deadline = 0.0
            self.restart_overhead = 0.0


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

        self._initialized = False

        self._spot_p_ema = 0.0
        self._spot_streak = 0
        self._alpha_spot = 0.05

        self._must_run_od = False
        self._od_cooldown = 0

        self._last_work_done = None
        self._rate_od_ema = 1.0
        self._rate_spot_ema = 0.7

        self._beta_rate = 0.2

        self._entry_min_streak_finish = 3
        self._od_cooldown_steps = 6

        self._buffer_low = 0.0
        self._buffer_high = 0.0
        self._buffer_very_low = 0.0

    def solve(self, spec_path: str) -> "Solution":
        self._initialized = False
        self._spot_p_ema = 0.0
        self._spot_streak = 0
        self._must_run_od = False
        self._od_cooldown = 0
        self._last_work_done = None
        self._rate_od_ema = 1.0
        self._rate_spot_ema = 0.7
        return self

    @staticmethod
    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _compute_work_done(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        if isinstance(tdt, (int, float)):
            return float(tdt)

        if not isinstance(tdt, (list, tuple)):
            return 0.0

        # Common patterns:
        # - list of durations (seconds)
        # - list of (start, end)
        # - list of dicts {start,end} or {duration}
        scalars = True
        vals = []
        for seg in tdt:
            if self._is_num(seg):
                vals.append(float(seg))
            else:
                scalars = False
                break

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0) if getattr(self, "env", None) is not None else 0.0
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0) if getattr(self, "env", None) is not None else 0.0

        if scalars:
            if not vals:
                return 0.0
            maxv = max(vals)
            # If these look like timestamps rather than durations, use count * gap
            if gap > 0 and maxv > self.task_duration and maxv <= elapsed + 2.0 * gap:
                return float(len(vals) * gap)
            # If these look like "per-step completed markers" (0/1), also use count * gap
            if gap > 0 and maxv <= 1.0 and min(vals) >= 0.0:
                return float(sum(1.0 for v in vals if v > 0.5) * gap)
            return float(sum(vals))

        total = 0.0
        for seg in tdt:
            if seg is None:
                continue
            if self._is_num(seg):
                total += float(seg)
                continue
            if isinstance(seg, (tuple, list)) and len(seg) >= 2 and self._is_num(seg[0]) and self._is_num(seg[1]):
                total += float(seg[1] - seg[0])
                continue
            if isinstance(seg, dict):
                if "duration" in seg and self._is_num(seg["duration"]):
                    total += float(seg["duration"])
                    continue
                if "work" in seg and self._is_num(seg["work"]):
                    total += float(seg["work"])
                    continue
                if "start" in seg and "end" in seg and self._is_num(seg["start"]) and self._is_num(seg["end"]):
                    total += float(seg["end"] - seg["start"])
                    continue

        # Fallback if still looks like just "segments" without duration
        if gap > 0 and total <= 1.0 and len(tdt) > 0:
            return float(len(tdt) * gap)
        return float(total)

    def _init_thresholds(self) -> None:
        gap = float(getattr(self.env, "gap_seconds", 300.0) or 300.0)
        over = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # Require ~15 minutes of stable availability before switching to spot in finish mode.
        self._entry_min_streak_finish = max(2, int(math.ceil(900.0 / max(gap, 1.0))))

        # Avoid thrashing by staying on OD for ~30 minutes after switching to OD (finish mode).
        self._od_cooldown_steps = max(3, int(math.ceil(1800.0 / max(gap, 1.0))))

        # Slack buffers (seconds)
        # buffer_low: when slack drops below this, prioritize guaranteed progress
        self._buffer_low = max(6.0 * gap, 12.0 * over, 1800.0)  # at least 30 minutes
        self._buffer_high = self._buffer_low + max(12.0 * gap, 1800.0)  # hysteresis ~30-60min+
        self._buffer_very_low = max(gap, 2.0 * over, 600.0)  # at least 10 minutes

    def _update_spot_stats(self, has_spot: bool) -> None:
        x = 1.0 if has_spot else 0.0
        self._spot_p_ema = (1.0 - self._alpha_spot) * self._spot_p_ema + self._alpha_spot * x
        if has_spot:
            self._spot_streak += 1
        else:
            self._spot_streak = 0

    def _update_rate_stats(self, last_cluster_type: ClusterType) -> float:
        work_done = self._compute_work_done()
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if self._last_work_done is not None and gap > 0:
            delta = work_done - self._last_work_done
            if not self._is_num(delta):
                delta = 0.0
            delta = float(max(0.0, min(delta, gap)))
            rate = delta / gap

            if last_cluster_type == ClusterType.ON_DEMAND:
                self._rate_od_ema = (1.0 - self._beta_rate) * self._rate_od_ema + self._beta_rate * rate
                self._rate_od_ema = float(min(1.0, max(0.05, self._rate_od_ema)))
            elif last_cluster_type == ClusterType.SPOT:
                self._rate_spot_ema = (1.0 - self._beta_rate) * self._rate_spot_ema + self._beta_rate * rate
                self._rate_spot_ema = float(min(1.0, max(0.01, self._rate_spot_ema)))

        self._last_work_done = work_done
        return work_done

    def _should_try_spot_in_finish(self, slack: float, last_cluster_type: ClusterType, has_spot: bool) -> bool:
        if not has_spot:
            return False
        if slack <= self._buffer_low + self._buffer_very_low:
            return False
        if self._od_cooldown > 0:
            return False
        if last_cluster_type == ClusterType.SPOT:
            return True
        if self._spot_streak < self._entry_min_streak_finish:
            return False
        if self._spot_p_ema < 0.25 and slack < (self._buffer_high + 2.0 * self._buffer_low):
            return False
        return True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if getattr(self, "env", None) is None:
            return ClusterType.ON_DEMAND if not has_spot else ClusterType.SPOT

        if not self._initialized:
            self._init_thresholds()
            self._initialized = True

        self._update_spot_stats(has_spot)
        work_done = self._update_rate_stats(last_cluster_type)

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        time_left = deadline - elapsed
        if time_left <= 0.0:
            return ClusterType.NONE

        work_left = max(0.0, task_duration - work_done)
        if work_left <= 0.0:
            return ClusterType.NONE

        slack = time_left - work_left

        if self._od_cooldown > 0:
            self._od_cooldown -= 1

        # Hysteresis around must-run OD mode
        if slack <= self._buffer_low:
            self._must_run_od = True
        elif slack >= self._buffer_high:
            self._must_run_od = False

        # Hard safety if we're behind schedule
        if slack <= 0.0:
            self._must_run_od = True

        # Decisions
        if self._must_run_od:
            # Very low slack: avoid spot risk even if currently on spot.
            if slack <= self._buffer_very_low:
                self._od_cooldown = self._od_cooldown_steps
                return ClusterType.ON_DEMAND

            if self._should_try_spot_in_finish(slack, last_cluster_type, has_spot):
                return ClusterType.SPOT

            self._od_cooldown = self._od_cooldown_steps
            return ClusterType.ON_DEMAND

        # Not in must-run mode: minimize cost, harvest spot, otherwise idle.
        if has_spot:
            return ClusterType.SPOT

        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)