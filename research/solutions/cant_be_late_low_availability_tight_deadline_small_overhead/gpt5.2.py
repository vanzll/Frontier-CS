import math
from collections import deque
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._initialized = False
        self._spot_hist = deque()
        self._hist_maxlen = 0
        self._spot_up_streak = 0
        self._spot_last = None
        self._min_up_steps = 1

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _work_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if not td:
            return 0.0

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        if isinstance(td, (int, float)):
            return max(0.0, min(float(td), task_duration if task_duration > 0 else float(td)))

        if not isinstance(td, (list, tuple)):
            return 0.0

        # If list of numbers, it can be either segment durations or cumulative progress.
        if all(self._is_number(x) for x in td):
            vals = [float(x) for x in td]
            if len(vals) >= 3:
                nondecreasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
                if nondecreasing and task_duration > 0 and vals[-1] <= task_duration * 1.05:
                    return max(0.0, min(vals[-1], task_duration))
            total = sum(v for v in vals if v > 0)
            if task_duration > 0:
                total = min(total, task_duration)
            return max(0.0, total)

        total = 0.0
        for seg in td:
            if seg is None:
                continue
            if self._is_number(seg):
                v = float(seg)
                if v > 0:
                    total += v
                continue
            if isinstance(seg, dict):
                if "duration" in seg and self._is_number(seg["duration"]):
                    v = float(seg["duration"])
                    if v > 0:
                        total += v
                elif "start" in seg and "end" in seg and self._is_number(seg["start"]) and self._is_number(seg["end"]):
                    a = float(seg["start"])
                    b = float(seg["end"])
                    if b >= a:
                        total += (b - a)
                continue
            if isinstance(seg, (tuple, list)) and len(seg) == 2 and self._is_number(seg[0]) and self._is_number(seg[1]):
                a = float(seg[0])
                b = float(seg[1])
                if b >= a:
                    total += (b - a)
                elif a > 0:
                    total += a

        if task_duration > 0:
            total = min(total, task_duration)
        return max(0.0, total)

    def _spot_avail_prob(self) -> float:
        n = len(self._spot_hist)
        if n <= 0:
            return 0.0
        s = float(sum(self._spot_hist))
        return (s + 1.0) / (n + 2.0)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 300.0
        self._hist_maxlen = max(30, int(round((3.0 * 3600.0) / gap)))
        self._spot_hist = deque(maxlen=self._hist_maxlen)
        self._min_up_steps = max(1, int(math.ceil(900.0 / gap)))  # require ~15 mins stable spot before switching from OD
        self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_initialized()

        # Update availability history and streaks.
        self._spot_hist.append(1 if has_spot else 0)
        if has_spot:
            self._spot_up_streak = self._spot_up_streak + 1 if self._spot_last is True else 1
            self._spot_last = True
        else:
            self._spot_up_streak = 0
            self._spot_last = False

        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        if gap <= 0:
            gap = 300.0

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        restart = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        done = self._work_done_seconds()
        remaining = max(0.0, task_duration - done)
        if remaining <= 0.0:
            return ClusterType.NONE

        time_left = deadline - now
        slack = time_left - remaining

        # Conservative safety buffers (to handle discretization + possible restart overhead on switching).
        critical_buffer = restart + 3.0 * gap + 600.0  # commit to OD when slack is this low
        no_idle_buffer = max(3600.0, 3.0 * critical_buffer)  # stop idling when slack is getting tight

        # If we are too close, commit to on-demand and finish.
        if slack <= critical_buffer:
            return ClusterType.ON_DEMAND

        # Three-phase policy:
        # A) slack > no_idle_buffer: use SPOT if available, else idle (NONE) to save OD for later and align work to spot.
        # B) critical_buffer < slack <= no_idle_buffer: use SPOT if available, else ON_DEMAND (no idling).
        # C) slack <= critical_buffer: ON_DEMAND (handled above).
        if slack > no_idle_buffer:
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.NONE

        # Phase B: no idling.
        if not has_spot:
            return ClusterType.ON_DEMAND

        # Spot is available in Phase B.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Avoid thrashing: only switch back if spot has been stable for a little while.
            if self._spot_up_streak >= self._min_up_steps and self._spot_avail_prob() >= 0.05:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        return ClusterType.SPOT

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)