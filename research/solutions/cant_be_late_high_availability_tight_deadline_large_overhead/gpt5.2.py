from __future__ import annotations

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
            except Exception:
                pass

        self._inited = False

        self._last_work_done = 0.0
        self._last_decision: ClusterType = ClusterType.NONE

        self._steps = 0
        self._avail_steps = 0

        self._down_streak = 0.0
        self._up_streak = 0.0
        self._max_down_seen = 0.0

        self._spot_wall = 0.0
        self._spot_progress = 0.0
        self._od_wall = 0.0
        self._od_progress = 0.0

        self._od_forced_until = 0.0
        self._od_entries = 0
        self._revert_count = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    @staticmethod
    def _is_non_decreasing(vals, window: int = 50) -> bool:
        n = len(vals)
        if n <= 1:
            return True
        start = max(0, n - 1 - window)
        prev = vals[start]
        for i in range(start + 1, n):
            v = vals[i]
            if v + 1e-12 < prev:
                return False
            prev = v
        return True

    def _get_work_done_seconds(self) -> float:
        td = getattr(self, "task_done_time", None)
        if td is None and hasattr(self, "env"):
            td = getattr(self.env, "task_done_time", None)

        if td is None:
            return 0.0

        task_dur = float(getattr(self, "task_duration", 0.0) or 0.0)

        if isinstance(td, (int, float)):
            done = float(td)
            if task_dur > 0:
                done = min(done, task_dur)
            return max(0.0, done)

        try:
            vals = list(td)
        except Exception:
            try:
                done = float(td)
                if task_dur > 0:
                    done = min(done, task_dur)
                return max(0.0, done)
            except Exception:
                return 0.0

        if not vals:
            return 0.0

        # Heuristic to interpret td:
        # - If cumulative (monotone increasing), use last element.
        # - Else, treat as increments/segments and sum.
        s = 0.0
        last = 0.0
        for x in vals:
            try:
                fx = float(x)
            except Exception:
                fx = 0.0
            s += fx
            last = fx

        n = len(vals)
        avg = s / max(1, n)
        monotone = self._is_non_decreasing(vals)

        if monotone and task_dur > 0 and last <= task_dur + 1e-6:
            ratio = last / max(avg, 1e-9)
            if ratio > 1.4:
                done = last
            else:
                # Could be constant increments; in that case, sum is correct.
                done = s
        else:
            done = s

        if task_dur > 0:
            done = min(done, task_dur)
        return max(0.0, float(done))

    def _set_od_forced(self, min_commit_seconds: float) -> None:
        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)
        until = now + max(0.0, min_commit_seconds)
        if until > self._od_forced_until:
            self._od_forced_until = until

    def _should_revert_to_spot(self, slack_left: float, remaining_work: float, ro: float, slack_total: float) -> bool:
        if self._revert_count >= 2:
            return False
        if remaining_work < max(6 * 3600.0, 0.12 * float(getattr(self, "task_duration", 0.0) or 0.0)):
            return False

        revert_slack = max(2.5 * 3600.0 + ro, 0.60 * slack_total + ro)
        if slack_left < revert_slack:
            return False

        if self._up_streak < max(1800.0, 0.08 * slack_total):
            return False

        # If spot has been extremely inefficient so far, don't revert.
        if self._spot_wall > 6 * 3600.0:
            spot_eff = self._spot_progress / max(self._spot_wall, 1e-9)
            if spot_eff < 0.6:
                return False

        return True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        gap = float(getattr(self.env, "gap_seconds", 0.0) or 0.0)
        now = float(getattr(self.env, "elapsed_seconds", 0.0) or 0.0)

        if not self._inited:
            self._inited = True
            self._last_work_done = self._get_work_done_seconds()
            self._last_decision = last_cluster_type if isinstance(last_cluster_type, ClusterType) else ClusterType.NONE

        # Update stats from the previous decision based on observed progress.
        work_done = self._get_work_done_seconds()
        progress_delta = work_done - self._last_work_done
        if progress_delta < 0:
            progress_delta = 0.0

        if self._steps > 0:
            if self._last_decision == ClusterType.SPOT:
                self._spot_wall += gap
                self._spot_progress += progress_delta
            elif self._last_decision == ClusterType.ON_DEMAND:
                self._od_wall += gap
                self._od_progress += progress_delta

        # Update availability stats/streaks using current has_spot observation.
        self._steps += 1
        if has_spot:
            self._avail_steps += 1

        if has_spot:
            if self._down_streak > 0:
                if self._down_streak > self._max_down_seen:
                    self._max_down_seen = self._down_streak
            self._down_streak = 0.0
            self._up_streak += gap
        else:
            self._up_streak = 0.0
            self._down_streak += gap

        self._last_work_done = work_done

        task_duration = float(getattr(self, "task_duration", 0.0) or 0.0)
        deadline = float(getattr(self, "deadline", 0.0) or 0.0)
        ro = float(getattr(self, "restart_overhead", 0.0) or 0.0)

        remaining_work = task_duration - work_done
        if remaining_work <= 1e-6:
            self._last_decision = ClusterType.NONE
            return ClusterType.NONE

        remaining_time = deadline - now
        if remaining_time <= 1e-6:
            self._last_decision = ClusterType.NONE
            return ClusterType.NONE

        slack_left = remaining_time - remaining_work
        slack_total = max(0.0, deadline - task_duration)

        # Dynamic safety thresholds (in wall-time seconds).
        critical_od = max(2.0 * 3600.0, 0.45 * slack_total) + ro
        reserve_slack = max(1.5 * 3600.0, 0.65 * slack_total) + ro

        # If spot has been very unreliable, widen the "always OD" zone slightly.
        if self._spot_wall > 6 * 3600.0:
            spot_eff = self._spot_progress / max(self._spot_wall, 1e-9)
            if spot_eff < 0.65:
                critical_od = max(critical_od, 3.0 * 3600.0)

        # If we're at real risk, use on-demand no matter what.
        if remaining_time <= remaining_work + ro:
            if self._last_decision != ClusterType.ON_DEMAND:
                self._od_entries += 1
                self._set_od_forced(min_commit_seconds=max(3600.0, 0.25 * slack_total))
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        if slack_left <= critical_od:
            if self._last_decision != ClusterType.ON_DEMAND:
                self._od_entries += 1
                self._set_od_forced(min_commit_seconds=max(3600.0, 0.25 * slack_total))
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Honor a minimum on-demand commitment window to avoid thrashing.
        if now < self._od_forced_until:
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Main policy:
        # - Prefer SPOT when available.
        # - When SPOT is unavailable:
        #     - if already on OD, continue OD
        #     - else optionally pause (NONE) while we still have slack budget
        #     - otherwise switch to OD with a minimum commitment.
        if has_spot:
            if last_cluster_type == ClusterType.ON_DEMAND or self._last_decision == ClusterType.ON_DEMAND:
                if self._should_revert_to_spot(slack_left, remaining_work, ro, slack_total):
                    self._revert_count += 1
                    self._last_decision = ClusterType.SPOT
                    return ClusterType.SPOT
                # If we cannot safely revert, keep OD to avoid extra restart overheads.
                self._last_decision = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND

            self._last_decision = ClusterType.SPOT
            return ClusterType.SPOT

        # has_spot is False:
        if last_cluster_type == ClusterType.ON_DEMAND or self._last_decision == ClusterType.ON_DEMAND:
            self._last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND

        # Consider pausing briefly to save cost if we have enough slack.
        wait_budget = slack_left - reserve_slack
        if wait_budget > 0:
            max_pause_cap = max(3600.0, 0.35 * slack_total)
            wait_max = min(max_pause_cap, wait_budget)
            # Also avoid waiting too long if we've already observed big outages.
            if self._max_down_seen > 0:
                wait_max = min(wait_max, max(1800.0, 0.8 * self._max_down_seen))
            if self._down_streak < wait_max:
                self._last_decision = ClusterType.NONE
                return ClusterType.NONE

        # Otherwise, go OD and commit for a while.
        self._od_entries += 1
        self._set_od_forced(min_commit_seconds=max(3600.0, 0.25 * slack_total))
        self._last_decision = ClusterType.ON_DEMAND
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)