import math
from typing import Optional

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_slack_ema_v1"

    def __init__(self, args=None):
        super().__init__(args)
        self._alpha = 0.08
        self._p_ema = 0.7
        self._steps = 0
        self._consec_no_spot = 0
        self._commit_on_demand = False
        self._od_lock_until = 0.0

    def solve(self, spec_path: str) -> "Solution":
        # Optional: spec-based configuration could go here.
        self._alpha = 0.08
        self._p_ema = 0.7
        self._steps = 0
        self._consec_no_spot = 0
        self._commit_on_demand = False
        self._od_lock_until = 0.0
        return self

    def _sum_done(self) -> float:
        done = 0.0
        try:
            for x in self.task_done_time:
                if x:
                    done += float(x)
        except Exception:
            try:
                done = float(sum(self.task_done_time))
            except Exception:
                done = 0.0
        return done

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._steps += 1
        obs = 1.0 if has_spot else 0.0
        self._p_ema = (1.0 - self._alpha) * self._p_ema + self._alpha * obs

        done = self._sum_done()
        remaining = float(self.task_duration) - done
        if remaining <= 0.0:
            return ClusterType.NONE

        elapsed = float(self.env.elapsed_seconds)
        gap = float(self.env.gap_seconds)
        time_left = float(self.deadline) - elapsed
        if time_left <= 0.0:
            self._commit_on_demand = True
            return ClusterType.ON_DEMAND

        restart_overhead = float(self.restart_overhead)
        safety = max(2.5 * restart_overhead, gap)
        slack = time_left - remaining
        needed_speed = (remaining + safety) / time_left if time_left > 0 else float("inf")

        commit_threshold = max(0.5 * 3600.0, (1.0 - self._p_ema) * 2.0 * 3600.0 + 4.0 * restart_overhead)
        if (not self._commit_on_demand) and (slack <= commit_threshold or needed_speed >= 0.98):
            self._commit_on_demand = True

        if self._commit_on_demand:
            return ClusterType.ON_DEMAND

        if elapsed < self._od_lock_until:
            return ClusterType.ON_DEMAND

        if has_spot:
            self._consec_no_spot = 0
            return ClusterType.SPOT

        self._consec_no_spot += 1

        if last_cluster_type == ClusterType.ON_DEMAND:
            if slack > 3.0 * 3600.0 and self._p_ema > 0.7:
                return ClusterType.NONE
            self._od_lock_until = elapsed + max(0.25 * 3600.0, 3.0 * restart_overhead)
            return ClusterType.ON_DEMAND

        reserve = max(1.0 * 3600.0, commit_threshold)
        max_wait_streak = 3 if self._p_ema > 0.6 else 1
        if slack > reserve and self._p_ema > 0.55 and self._consec_no_spot <= max_wait_streak:
            return ClusterType.NONE

        lock_dur = max(0.5 * 3600.0, 6.0 * restart_overhead)
        self._od_lock_until = elapsed + lock_dur
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)