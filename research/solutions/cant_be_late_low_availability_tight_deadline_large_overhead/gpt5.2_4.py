import math
from typing import Any

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_v1"

    def __init__(self, args: Any = None):
        try:
            super().__init__(args)
        except TypeError:
            try:
                super().__init__()
            except Exception:
                pass
        self.args = args

        self._initialized = False
        self._last_elapsed = -1.0

        self._committed_od = False
        self._outage_wait = 0.0
        self._interruptions = 0

        self._spot_ema = 0.0

        self._done_idx = 0
        self._done_sum = 0.0

        self._last_launch_time = -1e30
        self._last_launch_cluster = ClusterType.NONE

        self._min_switch_seconds = 0.0
        self._wait_before_od_seconds = 0.0
        self._base_commit_margin = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _seg_duration(seg: Any) -> float:
        if isinstance(seg, (int, float)):
            return float(seg)
        if isinstance(seg, (tuple, list)) and len(seg) == 2:
            a, b = seg
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return float(b) - float(a)
        if isinstance(seg, dict):
            for k in ("duration", "work", "seconds", "done"):
                v = seg.get(k, None)
                if isinstance(v, (int, float)):
                    return float(v)
        return 0.0

    def _update_done(self) -> None:
        td = getattr(self, "task_done_time", None)
        if not td:
            self._done_idx = 0
            self._done_sum = 0.0
            return
        n = len(td)
        if self._done_idx > n:
            self._done_idx = 0
            self._done_sum = 0.0
        for i in range(self._done_idx, n):
            d = self._seg_duration(td[i])
            if d > 0:
                self._done_sum += d
        self._done_idx = n

    def _reset_episode(self) -> None:
        self._committed_od = False
        self._outage_wait = 0.0
        self._interruptions = 0
        self._spot_ema = 0.0
        self._done_idx = 0
        self._done_sum = 0.0
        self._last_launch_time = -1e30
        self._last_launch_cluster = ClusterType.NONE

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)
        self._min_switch_seconds = max(0.0, 1.05 * ro)
        self._wait_before_od_seconds = max(0.0, 0.5 * ro)
        self._base_commit_margin = max(1800.0, 3600.0)  # 0.5h..1h
        self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        gap = float(getattr(self.env, "gap_seconds", 1.0) or 1.0)
        if now < self._last_elapsed:
            self._reset_episode()
        self._last_elapsed = now

        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self._interruptions += 1
            self._outage_wait = 0.0
        if has_spot:
            self._outage_wait = 0.0

        # Update spot availability EMA (time-constant ~2 hours)
        tau = 7200.0
        alpha = 1.0 - math.exp(-max(0.0, gap) / max(1.0, tau))
        self._spot_ema = (1.0 - alpha) * self._spot_ema + alpha * (1.0 if has_spot else 0.0)

        self._update_done()
        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        remaining_work = max(0.0, task_duration - self._done_sum)
        time_left = max(0.0, deadline - now)

        if remaining_work <= 1e-9:
            return ClusterType.NONE

        od_start_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro

        commit_margin = self._base_commit_margin + min(7200.0, 0.25 * self._interruptions * ro)

        if time_left <= remaining_work + od_start_overhead + commit_margin:
            self._committed_od = True

        if time_left <= remaining_work + od_start_overhead:
            self._committed_od = True

        if self._committed_od:
            action = ClusterType.ON_DEMAND
        else:
            if has_spot:
                if self._last_launch_cluster == ClusterType.ON_DEMAND:
                    if (now - self._last_launch_time) < self._min_switch_seconds:
                        action = ClusterType.ON_DEMAND
                    else:
                        action = ClusterType.SPOT
                else:
                    action = ClusterType.SPOT
            else:
                if last_cluster_type == ClusterType.ON_DEMAND:
                    action = ClusterType.ON_DEMAND
                else:
                    overhead_if_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else ro
                    slack_after_finish = time_left - (remaining_work + overhead_if_od + commit_margin)

                    if slack_after_finish <= 0.0:
                        action = ClusterType.ON_DEMAND
                    else:
                        can_wait_full = slack_after_finish > self._wait_before_od_seconds
                        if can_wait_full and self._outage_wait < self._wait_before_od_seconds:
                            self._outage_wait += gap
                            action = ClusterType.NONE
                        else:
                            action = ClusterType.ON_DEMAND

        if action in (ClusterType.SPOT, ClusterType.ON_DEMAND) and action != last_cluster_type:
            self._last_launch_cluster = action
            self._last_launch_time = now

        if not has_spot and action != ClusterType.NONE:
            self._outage_wait = self._wait_before_od_seconds

        if not has_spot and action == ClusterType.SPOT:
            action = ClusterType.ON_DEMAND

        return action

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)