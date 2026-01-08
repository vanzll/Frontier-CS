import math
from typing import Any

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


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

        self._args = args
        self._locked_on_demand = False

        self._n_steps = 0
        self._n_spot_available = 0
        self._ema_spot = None
        self._ema_alpha = 0.05

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _get_done_seconds(self) -> float:
        tdt = getattr(self, "task_done_time", None)
        if not tdt:
            return 0.0

        try:
            if all(isinstance(x, (int, float)) for x in tdt):
                nums = [float(x) for x in tdt]
                nondecreasing = True
                for i in range(len(nums) - 1):
                    if nums[i] > nums[i + 1] + 1e-9:
                        nondecreasing = False
                        break
                if nondecreasing:
                    return max(0.0, nums[-1])
                return max(0.0, sum(nums))

            done = 0.0
            for seg in tdt:
                if isinstance(seg, (int, float)):
                    done += float(seg)
                elif isinstance(seg, (tuple, list)) and len(seg) == 2:
                    a, b = seg
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        done += float(b) - float(a)
            return max(0.0, done)
        except Exception:
            return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update spot availability stats
        self._n_steps += 1
        if has_spot:
            self._n_spot_available += 1
        x = 1.0 if has_spot else 0.0
        if self._ema_spot is None:
            self._ema_spot = x
        else:
            self._ema_spot = (1.0 - self._ema_alpha) * self._ema_spot + self._ema_alpha * x

        elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", 0.0))
        deadline = float(getattr(self, "deadline", 0.0))
        overhead = float(getattr(self, "restart_overhead", 0.0))
        task_duration = float(getattr(self, "task_duration", 0.0))

        done = self._get_done_seconds()
        remaining_work = max(0.0, task_duration - done)
        time_left = max(0.0, deadline - elapsed)

        if remaining_work <= 1e-9:
            return ClusterType.NONE

        # If we're too close to the deadline, commit to on-demand to guarantee completion.
        start_od_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else overhead
        hard_margin = 2.0 * gap  # discrete-time safety margin
        if time_left <= remaining_work + start_od_overhead + hard_margin + 1e-9:
            self._locked_on_demand = True
            return ClusterType.ON_DEMAND

        if self._locked_on_demand:
            return ClusterType.ON_DEMAND

        slack = time_left - remaining_work  # seconds of slack assuming uninterrupted progress

        # When slack is small, avoid spot risk and lock to on-demand.
        min_slack_for_spot = max(3600.0, 5.0 * overhead)
        if slack <= min_slack_for_spot + 1e-9:
            self._locked_on_demand = True
            return ClusterType.ON_DEMAND

        # If already on-demand, prefer to stay (avoid extra restart overhead and spot risk).
        if last_cluster_type == ClusterType.ON_DEMAND:
            # Optional switch-back (rarely beneficial); keep conservative.
            switch_back_slack = max(3.0 * 3600.0, 10.0 * overhead)
            if has_spot and slack >= switch_back_slack and (self._ema_spot is not None and self._ema_spot >= 0.6):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Otherwise, use spot whenever available; if unavailable, pause until forced onto on-demand.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)