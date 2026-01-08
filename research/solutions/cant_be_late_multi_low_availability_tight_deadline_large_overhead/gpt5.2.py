import json
from argparse import Namespace
from typing import List, Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Cant-Be-Late multi-region scheduling strategy."""

    NAME = "cant_be_late_mr_v1"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        return self

    def _lazy_init(self) -> None:
        if getattr(self, "_inited", False):
            return
        self._inited = True

        self._ct_spot = getattr(ClusterType, "SPOT")
        self._ct_od = getattr(ClusterType, "ON_DEMAND")
        self._ct_none = getattr(ClusterType, "NONE", None)
        if self._ct_none is None:
            self._ct_none = getattr(ClusterType, "None")

        self._gap = float(self.env.gap_seconds)

        self._done_sum = 0.0
        self._done_len = 0

        n = int(self.env.get_num_regions())
        self._n_regions = n
        self._reg_total: List[int] = [0] * n
        self._reg_avail: List[int] = [0] * n
        self._reg_last_seen: List[float] = [-1.0] * n

        self._unavail_streak_steps = 0
        self._last_region = int(self.env.get_current_region())

        self._committed_od = False
        self._total_switches = 0
        self._last_switch_elapsed = -1e18

    def _get_scalar(self, x):
        if isinstance(x, (list, tuple)):
            return float(x[0]) if x else 0.0
        return float(x)

    def _update_done_sum(self) -> None:
        td = self.task_done_time
        ln = len(td)
        if ln > self._done_len:
            # Usually one append per step; incremental sum keeps per-step cost O(1).
            self._done_sum += sum(td[self._done_len : ln])
            self._done_len = ln

    def _region_est(self, idx: int) -> float:
        # Beta(1,1) mean prior.
        t = self._reg_total[idx]
        a = self._reg_avail[idx]
        return (a + 1.0) / (t + 2.0)

    def _pick_switch_target(self, cur: int) -> int:
        n = self._n_regions
        best = cur
        best_score = -1.0
        best_total = 1 << 30
        for i in range(n):
            if i == cur:
                continue
            score = self._region_est(i)
            t = self._reg_total[i]
            # Prefer better estimated availability; break ties toward less explored.
            if (score > best_score) or (score == best_score and t < best_total):
                best = i
                best_score = score
                best_total = t
        if best == cur:
            best = (cur + 1) % n
        return best

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        cur = int(self.env.get_current_region())
        if cur != self._last_region:
            self._last_region = cur
            self._unavail_streak_steps = 0

        self._reg_total[cur] += 1
        if has_spot:
            self._reg_avail[cur] += 1
        self._reg_last_seen[cur] = float(self.env.elapsed_seconds)

        self._update_done_sum()

        task_duration = self._get_scalar(self.task_duration)
        deadline = self._get_scalar(self.deadline)
        restart_overhead = self._get_scalar(self.restart_overhead)

        remaining_work = task_duration - self._done_sum
        if remaining_work <= 0.0:
            return self._ct_none

        elapsed = float(self.env.elapsed_seconds)
        time_left = deadline - elapsed
        if time_left <= 0.0:
            return self._ct_none

        # Conservative finish buffer to avoid penalty under modeling quirks/discretization.
        finish_buffer = max(2.0 * self._gap, 60.0, 0.1 * restart_overhead)

        # Commit to on-demand if we are close to the latest safe point.
        # Use 2*restart_overhead buffer to be conservative about restart timing/modeling.
        if time_left <= remaining_work + 2.0 * restart_overhead + finish_buffer:
            self._committed_od = True

        if self._committed_od:
            return self._ct_od

        # Not committed: use spot when available; otherwise wait (NONE) and optionally switch regions.
        if has_spot:
            self._unavail_streak_steps = 0
            return self._ct_spot

        self._unavail_streak_steps += 1
        unavail_seconds = self._unavail_streak_steps * self._gap

        slack = time_left - (remaining_work + 2.0 * restart_overhead + finish_buffer)

        # Only switch regions if we've been without spot for a while and we have ample slack.
        if self._n_regions > 1:
            switch_after = max(1800.0, 2.0 * restart_overhead)
            cooldown = max(1800.0, 2.0 * restart_overhead)
            allow_switch = (
                unavail_seconds >= switch_after
                and slack >= (switch_after + 3.0 * restart_overhead)
                and (elapsed - self._last_switch_elapsed) >= cooldown
                and self._total_switches < max(6, 2 * self._n_regions)
            )
            if allow_switch:
                target = self._pick_switch_target(cur)
                if target != cur:
                    self.env.switch_region(target)
                    self._last_switch_elapsed = elapsed
                    self._total_switches += 1
                    self._unavail_streak_steps = 0
                    return self._ct_none

        return self._ct_none