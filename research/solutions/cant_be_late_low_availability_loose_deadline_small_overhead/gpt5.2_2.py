import math
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_deadline_aware_v1"

    def __init__(self, args: Optional[Any] = None):
        try:
            super().__init__(args)
        except TypeError:
            super().__init__()
        self._initialized = False
        self._lock_on_demand = False
        self._hard_slack_s = 0.0
        self._soft_slack_s = 0.0
        self._prev_done = None
        self._prev_elapsed = None
        self._spot_streak = 0
        self._no_spot_streak = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_thresholds(self) -> None:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        # Hard slack: once below this, run OD continuously to avoid missing deadline.
        hard = 3.0 * gap + 3.0 * ro
        hard = max(hard, 30.0 * 60.0)  # at least 30 minutes
        self._hard_slack_s = hard

        # Soft slack: once below this, never pause when spot is unavailable.
        soft = max(6.0 * 3600.0, 8.0 * gap)
        # Cap to avoid overreacting on extremely large gap sizes
        soft = min(soft, 10.0 * 3600.0)  # 10 hours cap
        self._soft_slack_s = max(soft, self._hard_slack_s + gap)

        self._initialized = True

    def _work_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        # Common cases:
        # - list of segment durations (sum)
        # - list of cumulative done values (take last)
        # - list of (start, end) segments (sum of end-start)
        try:
            last = tdt[-1]
        except Exception:
            return 0.0

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)

        # Case: list of numbers
        if isinstance(last, (int, float)):
            try:
                s = float(sum(float(x) for x in tdt))
            except Exception:
                s = float(last)

            lastv = float(last)

            if s <= task_duration + 1e-6:
                done = s
            elif lastv <= task_duration + 1e-6:
                done = lastv
            else:
                done = min(s, lastv, task_duration)
            return max(0.0, min(done, task_duration))

        # Case: list of tuples/lists (start,end) or objects
        if isinstance(last, (tuple, list)) and len(last) == 2:
            total = 0.0
            for seg in tdt:
                try:
                    a, b = seg
                    total += max(0.0, float(b) - float(a))
                except Exception:
                    continue
            return max(0.0, min(total, task_duration))

        # Case: list of objects/dicts with duration
        total = 0.0
        for seg in tdt:
            try:
                if isinstance(seg, dict):
                    d = seg.get("duration", None)
                    if d is None:
                        a = seg.get("start", None)
                        b = seg.get("end", None)
                        if a is not None and b is not None:
                            d = float(b) - float(a)
                    if d is not None:
                        total += max(0.0, float(d))
                else:
                    d = getattr(seg, "duration", None)
                    if d is None:
                        a = getattr(seg, "start", None)
                        b = getattr(seg, "end", None)
                        if a is not None and b is not None:
                            d = float(b) - float(a)
                    if d is not None:
                        total += max(0.0, float(d))
            except Exception:
                continue
        return max(0.0, min(total, task_duration))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._init_thresholds()

        if has_spot:
            self._spot_streak += 1
            self._no_spot_streak = 0
        else:
            self._no_spot_streak += 1
            self._spot_streak = 0

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)

        done = self._work_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        remaining_work = max(0.0, task_duration - done)
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        remaining_time = max(0.0, deadline - elapsed)

        # Detect likely restart overhead (no progress while running a cluster).
        in_overhead = False
        if self._prev_done is not None and self._prev_elapsed is not None:
            if elapsed > self._prev_elapsed + 1e-9:
                if done <= self._prev_done + 1e-9 and last_cluster_type != ClusterType.NONE:
                    in_overhead = True
        self._prev_done = done
        self._prev_elapsed = elapsed

        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        # Time budget feasibility check: account for needing (at least) one restart overhead
        # if we're not already on on-demand when we commit to OD.
        need_start_overhead_for_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro
        slack = remaining_time - remaining_work

        # Lock on-demand in the critical regime.
        if self._lock_on_demand:
            return ClusterType.ON_DEMAND

        if slack <= self._hard_slack_s + need_start_overhead_for_od:
            self._lock_on_demand = True
            return ClusterType.ON_DEMAND

        # If we're currently in restart overhead, keep the same cluster type to avoid resetting it.
        if in_overhead:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND

        # Soft regime: don't pause when close; use OD if spot unavailable.
        if slack <= self._soft_slack_s:
            if has_spot:
                # Avoid switching OD->SPOT on the first available step to reduce churn.
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                if last_cluster_type == ClusterType.ON_DEMAND and self._spot_streak < 2:
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Far from deadline: use spot when available; otherwise pause to save cost.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)