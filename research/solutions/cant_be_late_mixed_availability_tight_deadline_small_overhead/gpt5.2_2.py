import os
from typing import Any, Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Optional[Any] = None):
        super().__init__(args)
        self._initialized = False

        self._steps = 0
        self._p_ewma = 0.5
        self._p_alpha = 0.03

        self._prev_has_spot: Optional[bool] = None
        self._unavail_run = 0
        self._unavail_ewma = 2.0
        self._unavail_beta = 0.2

        self._wait_factor = 1.25

    def solve(self, spec_path: str) -> "Solution":
        self._initialized = True
        if spec_path and os.path.exists(spec_path):
            # Optional config hook; ignored by default.
            pass
        return self

    def _get_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        dur = float(getattr(self, "task_duration", 0.0) or 0.0)

        if td is None:
            return 0.0
        if isinstance(td, (int, float)):
            v = float(td)
            return min(v, dur) if dur > 0 else v

        if isinstance(td, list):
            if not td:
                return 0.0

            first = td[0]

            if isinstance(first, (int, float)):
                nums = [float(x) for x in td if isinstance(x, (int, float))]
                if not nums:
                    return 0.0
                s = sum(nums)
                mx = max(nums)
                last = nums[-1]
                if dur > 0:
                    # Heuristic: if sum is far larger than task duration but max/last is plausible, treat as cumulative list.
                    if s > dur * 1.2 and mx <= dur * 1.05:
                        return mx
                    # If monotone non-decreasing and last is plausible, treat as cumulative.
                    monotone = True
                    for i in range(len(nums) - 1):
                        if nums[i] > nums[i + 1] + 1e-9:
                            monotone = False
                            break
                    if monotone and last <= dur * 1.05:
                        return last
                    return min(s, dur)
                return last if s > 10 * last else s

            if isinstance(first, (tuple, list)) and len(first) >= 2:
                total = 0.0
                for seg in td:
                    if not (isinstance(seg, (tuple, list)) and len(seg) >= 2):
                        continue
                    try:
                        a = float(seg[0])
                        b = float(seg[1])
                        total += abs(b - a)
                    except Exception:
                        continue
                return min(total, dur) if dur > 0 else total

            if isinstance(first, dict):
                total = 0.0
                for seg in td:
                    if not isinstance(seg, dict):
                        continue
                    a = seg.get("start", seg.get("s", 0.0))
                    b = seg.get("end", seg.get("e", 0.0))
                    try:
                        total += abs(float(b) - float(a))
                    except Exception:
                        continue
                return min(total, dur) if dur > 0 else total

        return 0.0

    def _update_availability_stats(self, has_spot: bool) -> None:
        self._steps += 1
        self._p_ewma = (1.0 - self._p_alpha) * self._p_ewma + self._p_alpha * (1.0 if has_spot else 0.0)

        if self._prev_has_spot is None:
            self._prev_has_spot = has_spot
            self._unavail_run = 0 if has_spot else 1
            return

        if has_spot:
            if self._prev_has_spot is False:
                # outage ended; update ewma with the run length
                run = float(self._unavail_run)
                self._unavail_ewma = (1.0 - self._unavail_beta) * self._unavail_ewma + self._unavail_beta * run
                self._unavail_run = 0
        else:
            self._unavail_run += 1

        self._prev_has_spot = has_spot

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_availability_stats(has_spot)

        done = self._get_done_seconds()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        remaining_work = max(0.0, task_duration - done)

        if remaining_work <= 0.0:
            return ClusterType.NONE

        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        time_left = max(0.0, deadline - now)

        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        overhead = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        slack = time_left - remaining_work

        # If behind schedule (or extremely tight), always run on-demand.
        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        # Conservative safety: allow for at least one restart and a decision granularity.
        safety = 2.0 * overhead + 2.0 * gap

        # If near the deadline, avoid any waiting and prefer on-demand even if spot is available.
        if time_left <= remaining_work + overhead + gap:
            return ClusterType.ON_DEMAND

        if slack <= safety:
            return ClusterType.ON_DEMAND

        p = self._p_ewma

        if has_spot:
            # In very low-availability regimes and with low slack, avoid churn by staying on-demand.
            if last_cluster_type == ClusterType.ON_DEMAND and (p < 0.16 and slack < 3600.0):
                return ClusterType.ON_DEMAND
            # If already on on-demand and slack is extremely small, don't switch.
            if last_cluster_type == ClusterType.ON_DEMAND and slack <= overhead + 2.0 * gap:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # No spot available: decide whether to wait or use on-demand.
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Once on-demand, generally keep going during outages.
            return ClusterType.ON_DEMAND

        # If availability is extremely low, waiting is usually harmful.
        if p < 0.12:
            return ClusterType.ON_DEMAND

        # Wait a bit for short outages when we have slack.
        if slack > safety + gap:
            max_wait_steps_by_slack = (slack - safety) / max(gap, 1e-9)
            predicted_outage_steps = max(1.0, self._unavail_ewma)
            wait_cap_steps = min(predicted_outage_steps * self._wait_factor, max_wait_steps_by_slack)

            # Be more willing to wait when spot is reasonably available.
            if p >= 0.22 and float(self._unavail_run) < wait_cap_steps:
                return ClusterType.NONE

        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)