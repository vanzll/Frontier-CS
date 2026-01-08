import os
import json
from typing import Any, Optional

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
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except Exception:
            try:
                super().__init__()
            except Exception:
                pass

        self._spec = {}
        self._last_elapsed: Optional[float] = None
        self._last_action: Optional[ClusterType] = None
        self._prev_has_spot: Optional[bool] = None

        self._est_done_work: float = 0.0
        self._est_overhead_remaining: float = 0.0

        self._in_outage: bool = False
        self._outage_waited: float = 0.0

        self._spot_run_len: float = 0.0
        self._outage_run_len: float = 0.0
        self._mean_uptime: float = 2.0 * 3600.0
        self._mean_outage: float = 10.0 * 60.0

        self._od_hold_until: float = 0.0
        self._commit_od: bool = False

    def solve(self, spec_path: str) -> "Solution":
        try:
            if spec_path and os.path.exists(spec_path):
                with open(spec_path, "r") as f:
                    if spec_path.endswith(".json"):
                        self._spec = json.load(f)
                    else:
                        # Best-effort parse for non-json specs; ignore on failure
                        self._spec = {}
        except Exception:
            self._spec = {}
        return self

    def _parse_done_work(self) -> Optional[float]:
        tdt = getattr(self, "task_done_time", None)
        if tdt is None:
            return None

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        if task_duration <= 0:
            task_duration = float("inf")

        try:
            if isinstance(tdt, (int, float)):
                v = float(tdt)
                if 0.0 <= v <= task_duration:
                    return v
                return None

            if not isinstance(tdt, (list, tuple)):
                return None

            if len(tdt) == 0:
                return 0.0

            # List of segments [(start,end), ...]
            is_seg = True
            total = 0.0
            for x in tdt:
                if not (isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], (int, float)) and isinstance(x[1], (int, float))):
                    is_seg = False
                    break
                a = float(x[0])
                b = float(x[1])
                if b > a:
                    total += (b - a)
            if is_seg:
                if 0.0 <= total <= task_duration:
                    return total
                return None

            # List of numeric values
            if all(isinstance(x, (int, float)) for x in tdt):
                arr = [float(x) for x in tdt]
                nondec = True
                for i in range(len(arr) - 1):
                    if arr[i] > arr[i + 1] + 1e-9:
                        nondec = False
                        break

                last = arr[-1]
                if nondec and 0.0 <= last <= task_duration:
                    return last

                s = sum(arr)
                if 0.0 <= s <= task_duration:
                    return s

                # Try sum of positive diffs (handles cumulative-ish traces that might reset)
                diffs = 0.0
                prev = arr[0]
                for i in range(1, len(arr)):
                    cur = arr[i]
                    if cur >= prev:
                        diffs += (cur - prev)
                    prev = cur
                if 0.0 <= diffs <= task_duration:
                    return diffs

            return None
        except Exception:
            return None

    def _get_done_work(self) -> float:
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        done_env = self._parse_done_work()

        done = None
        if done_env is not None and done_env >= 0.0:
            done = done_env

        if done is None:
            done = self._est_done_work

        # Conservative: never exceed task_duration, never below 0.
        if task_duration > 0.0:
            if done < 0.0:
                done = 0.0
            if done > task_duration:
                done = task_duration
        return float(done)

    def _update_internal_estimators(self, last_cluster_type: ClusterType, has_spot: bool) -> None:
        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)

        if self._last_elapsed is None:
            self._last_elapsed = elapsed
            self._last_action = last_cluster_type
            self._prev_has_spot = has_spot
            self._in_outage = (not has_spot)
            self._outage_waited = 0.0
            self._spot_run_len = 0.0
            self._outage_run_len = 0.0
            self._est_done_work = 0.0
            self._est_overhead_remaining = 0.0
            return

        delta = elapsed - self._last_elapsed
        if delta < 0:
            delta = gap if gap > 0 else 0.0

        # Attribute the last interval's availability to _prev_has_spot.
        if self._prev_has_spot is True:
            self._spot_run_len += delta
        elif self._prev_has_spot is False:
            self._outage_run_len += delta

        # Update internal progress estimate based on last action and overhead.
        if self._last_action is not None and self._last_action != ClusterType.NONE and delta > 0:
            used = min(delta, self._est_overhead_remaining)
            self._est_overhead_remaining -= used
            progressed = delta - used
            if progressed > 0:
                self._est_done_work += progressed

        # Detect transitions in spot availability (at the boundary to current step).
        if self._prev_has_spot is not None and self._prev_has_spot != has_spot:
            if self._prev_has_spot is True and has_spot is False:
                # Uptime ended
                if self._spot_run_len > 0:
                    self._mean_uptime = 0.8 * self._mean_uptime + 0.2 * self._spot_run_len
                self._spot_run_len = 0.0
                self._outage_waited = 0.0
                self._in_outage = True
            elif self._prev_has_spot is False and has_spot is True:
                # Outage ended
                if self._outage_run_len > 0:
                    self._mean_outage = 0.8 * self._mean_outage + 0.2 * self._outage_run_len
                self._outage_run_len = 0.0
                self._outage_waited = 0.0
                self._in_outage = False

        self._last_elapsed = elapsed
        self._prev_has_spot = has_spot

    def _compute_safety(self, gap: float, overhead: float) -> float:
        # A small buffer to avoid last-moment failures due to discretization/switching.
        return max(300.0, 1.0 * gap, 0.25 * overhead)

    def _allowed_wait_in_outage(self, slack: float) -> float:
        # Per-outage waiting cap; dynamic based on observed outage lengths, but never too large.
        base = max(300.0, min(3600.0, 1.2 * float(self._mean_outage)))
        if slack <= 0:
            return 0.0
        return min(base, 0.5 * slack)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_internal_estimators(last_cluster_type, has_spot)

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        if deadline > 0 and elapsed >= deadline:
            self._last_action = ClusterType.NONE
            return ClusterType.NONE

        done_work = self._get_done_work()
        if task_duration > 0 and done_work >= task_duration - 1e-6:
            self._last_action = ClusterType.NONE
            return ClusterType.NONE

        remaining_work = max(0.0, task_duration - done_work) if task_duration > 0 else 0.0
        remaining_time = (deadline - elapsed) if deadline > 0 else float("inf")
        if remaining_time < 0:
            remaining_time = 0.0

        slack = remaining_time - remaining_work
        safety = self._compute_safety(gap, overhead)

        min_slack_for_switch = overhead + 2.0 * gap + safety
        commit_slack = overhead + 3.0 * gap + safety
        wait_slack_buffer = 2.0 * overhead + 4.0 * gap + safety

        # Hard safety: if we're too tight, commit to OD to eliminate further risk.
        if remaining_time <= remaining_work + overhead + safety:
            self._commit_od = True
        if slack <= commit_slack:
            self._commit_od = True
        if self._commit_od:
            action = ClusterType.ON_DEMAND
            if action != ClusterType.NONE and action != last_cluster_type:
                self._est_overhead_remaining = overhead
            self._last_action = action
            return action

        # Dynamic OD hold to reduce thrashing when spot availability flickers.
        od_hold_sec = max(900.0, 2.0 * overhead, 4.0 * gap)

        # Prefer SPOT when available, but avoid switching into SPOT if it likely causes too much overhead.
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND and elapsed < self._od_hold_until:
                action = ClusterType.ON_DEMAND
            else:
                # Require enough slack and reasonable observed stability to switch OD/NONE -> SPOT.
                if last_cluster_type != ClusterType.SPOT:
                    stable_enough = self._mean_uptime >= (3.0 * overhead if overhead > 0 else 0.0)
                    if slack > min_slack_for_switch and stable_enough:
                        action = ClusterType.SPOT
                    else:
                        action = ClusterType.ON_DEMAND
                else:
                    action = ClusterType.SPOT
        else:
            # Spot not available: wait (NONE) if we can afford it and outages tend to be short.
            allowed_wait = self._allowed_wait_in_outage(slack)
            headroom = max(0.0, slack - commit_slack)

            if slack > wait_slack_buffer and self._outage_waited + gap <= min(allowed_wait, headroom):
                action = ClusterType.NONE
                self._outage_waited += gap
            else:
                action = ClusterType.ON_DEMAND
                if last_cluster_type == ClusterType.SPOT:
                    self._od_hold_until = elapsed + od_hold_sec

        # Respect API constraint: never request spot when unavailable.
        if action == ClusterType.SPOT and not has_spot:
            action = ClusterType.ON_DEMAND

        # Update internal overhead estimate on cluster launch/change.
        if action != ClusterType.NONE and action != last_cluster_type:
            self._est_overhead_remaining = overhead

        self._last_action = action
        return action

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)