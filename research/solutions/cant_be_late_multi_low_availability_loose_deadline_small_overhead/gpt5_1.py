import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
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
        # Lazy initialization of region stats (after env is available)
        self._initialized = False
        return self

    def _lazy_init(self):
        if self._initialized:
            return
        try:
            num_regions = self.env.get_num_regions()
        except Exception:
            num_regions = 1
        self.num_regions = num_regions
        self.alpha = [1.0] * num_regions  # successes
        self.beta = [1.0] * num_regions   # failures
        self.up_streak = [0] * num_regions
        self.down_streak = [0] * num_regions
        self._last_stats_update_step = -1
        self._last_switch_step = -1000000
        self._switch_cooldown_steps = 1
        self._latched_on_demand = False
        self._initialized = True

    def _current_step_idx(self):
        gap = max(self.env.gap_seconds, 1.0)
        return int(self.env.elapsed_seconds // gap)

    def _update_stats(self, region_idx: int, has_spot: bool):
        step_idx = self._current_step_idx()
        if self._last_stats_update_step == step_idx:
            return
        self._last_stats_update_step = step_idx
        if 0 <= region_idx < self.num_regions:
            if has_spot:
                self.alpha[region_idx] += 1.0
                self.up_streak[region_idx] += 1
                self.down_streak[region_idx] = 0
            else:
                self.beta[region_idx] += 1.0
                self.down_streak[region_idx] += 1
                self.up_streak[region_idx] = 0

    def _region_score(self, idx: int) -> float:
        # Bayesian mean with mild prior; add small recency adjustments
        a = self.alpha[idx]
        b = self.beta[idx]
        if a + b <= 0:
            base = 0.5
        else:
            base = a / (a + b)
        # Recency: reward short up streaks slightly, penalize down streaks slightly
        bonus = min(self.up_streak[idx] * 0.02, 0.08) - min(self.down_streak[idx] * 0.01, 0.06)
        score = base + bonus
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score

    def _best_region(self, exclude_idx: int = -1) -> int:
        best_idx = 0
        best_score = -1.0
        for i in range(self.num_regions):
            if i == exclude_idx:
                continue
            s = self._region_score(i)
            if s > best_score:
                best_score = s
                best_idx = i
        return best_idx

    def _time_needed_on_demand(self, last_cluster_type: ClusterType) -> float:
        # Remaining work (seconds) + remaining/restart overhead + rounding fudge
        rem = max(self.task_duration - sum(self.task_done_time), 0.0)
        if rem <= 0:
            return 0.0
        overhead_remaining = self.remaining_restart_overhead if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead
        # Add one gap to account for step granularity
        fudge = self.env.gap_seconds
        return rem + overhead_remaining + fudge

    def _should_commit_on_demand(self, last_cluster_type: ClusterType) -> bool:
        time_left = self.deadline - self.env.elapsed_seconds
        need = self._time_needed_on_demand(last_cluster_type)
        return time_left <= need

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()
        cur_region = self.env.get_current_region()
        self._update_stats(cur_region, has_spot)

        # If already latched to on-demand, keep going to ensure completion
        if self._latched_on_demand:
            return ClusterType.ON_DEMAND

        # If task already done (env should stop, but safe-guard)
        rem = max(self.task_duration - sum(self.task_done_time), 0.0)
        if rem <= 0:
            return ClusterType.NONE

        # Commit to on-demand if required to guarantee deadline
        if self._should_commit_on_demand(last_cluster_type):
            self._latched_on_demand = True
            return ClusterType.ON_DEMAND

        # Prefer SPOT if available in current region; avoid switching when currently available
        if has_spot:
            return ClusterType.SPOT

        # If SPOT unavailable here, decide to wait (NONE) and optionally reposition to a better region
        # Only switch regions while idling to avoid accidental SPOT selection with unknown availability.
        time_left = self.deadline - self.env.elapsed_seconds
        need_next = self.restart_overhead + rem + self.env.gap_seconds  # approximate if we start OD next step
        slack = time_left - need_next

        # Choose to wait only if we have slack; else fall back to on-demand
        if slack > 0:
            # Reposition to best region if significantly better than current
            current_score = self._region_score(cur_region)
            best_idx = self._best_region(exclude_idx=-1)
            best_score = self._region_score(best_idx)
            if best_idx != cur_region and best_score >= current_score + 0.05:
                # apply minimal cooldown to prevent thrashing
                step_idx = self._current_step_idx()
                if step_idx - self._last_switch_step >= self._switch_cooldown_steps:
                    self.env.switch_region(best_idx)
                    self._last_switch_step = step_idx
            return ClusterType.NONE

        # Not enough slack to wait further; go on-demand now and latch
        self._latched_on_demand = True
        return ClusterType.ON_DEMAND