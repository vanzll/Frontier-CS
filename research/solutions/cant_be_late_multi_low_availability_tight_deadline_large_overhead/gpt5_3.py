import json
from argparse import Namespace
from typing import List, Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "deadline_guard_mr_v1"

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

        # Internal state; set lazily when env is ready in _step
        self._initialized = False
        self._commit_to_od = False
        self._od_commit_region: Optional[int] = None
        self._last_region_switch_time: float = -1e18
        self._switch_cooldown_seconds: float = 300.0  # avoid thrashing region switches while waiting
        self._od_margin_seconds: float = 0.0  # set at first step based on env.gap_seconds
        # Region stats
        self._region_obs: List[int] = []
        self._region_spot: List[int] = []
        self._region_last_spot_time: List[float] = []
        # Prior for Beta smoothing
        self._prior_alpha = 1.0
        self._prior_beta = 1.0
        return self

    def _lazy_init(self):
        if self._initialized:
            return
        num_regions = self.env.get_num_regions()
        self._region_obs = [0 for _ in range(num_regions)]
        self._region_spot = [0 for _ in range(num_regions)]
        self._region_last_spot_time = [-1e18 for _ in range(num_regions)]
        # Set margin to be slightly conservative: one gap + 10% overhead
        gap = getattr(self.env, "gap_seconds", 60.0)
        self._od_margin_seconds = max(gap, 0.1 * self.restart_overhead)
        # Switch cooldown at least half overhead, at most 15 minutes
        self._switch_cooldown_seconds = max(0.5 * self.restart_overhead, min(900.0, self._switch_cooldown_seconds))
        self._initialized = True

    def _update_region_stats(self, region_idx: int, has_spot: bool, now: float):
        if region_idx < 0 or region_idx >= len(self._region_obs):
            return
        self._region_obs[region_idx] += 1
        if has_spot:
            self._region_spot[region_idx] += 1
            self._region_last_spot_time[region_idx] = now

    def _best_region(self, prefer_current: int) -> int:
        # Compute posterior mean for spot availability using Beta prior
        best_idx = prefer_current
        best_score = -1.0
        for i in range(len(self._region_obs)):
            obs = self._region_obs[i]
            spot = self._region_spot[i]
            alpha = self._prior_alpha + spot
            beta = self._prior_beta + max(0, obs - spot)
            score = alpha / (alpha + beta)
            # Break ties by recent spot timestamp, then prefer current region
            # Slight bonus for regions with very recent spot availability
            recent_bonus = 0.0
            if self._region_last_spot_time[i] > -1e17:
                recent_bonus = min(0.05, (self._region_last_spot_time[i] - max(self._region_last_spot_time)) / 1e7)  # tiny influence
            score += recent_bonus
            if score > best_score or (abs(score - best_score) < 1e-12 and i == prefer_current):
                best_score = score
                best_idx = i
        return best_idx

    def _must_start_od_now(self, now: float, remaining_work: float) -> bool:
        # Latest time to start OD from non-OD state and still finish:
        latest_start = self.deadline - (remaining_work + self.restart_overhead + self._od_margin_seconds)
        return now >= latest_start

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        now = self.env.elapsed_seconds
        current_region = self.env.get_current_region()

        # Update region stats with current observation
        self._update_region_stats(current_region, has_spot, now)

        # Compute remaining work
        progress = 0.0
        if self.task_done_time:
            progress = sum(self.task_done_time)
        remaining_work = max(0.0, self.task_duration - progress)

        # If done, do nothing
        if remaining_work <= 1e-9:
            return ClusterType.NONE

        # If we have already committed to OD, keep using OD until finish
        if self._commit_to_od:
            # Ensure we stay in the chosen OD region to avoid extra overhead
            if self._od_commit_region is not None and current_region != self._od_commit_region:
                self.env.switch_region(self._od_commit_region)
            return ClusterType.ON_DEMAND

        # If it's time (or past time) we must start OD to guarantee deadline
        if self._must_start_od_now(now, remaining_work):
            self._commit_to_od = True
            self._od_commit_region = current_region
            return ClusterType.ON_DEMAND

        # Otherwise, opportunistically use SPOT if available
        if has_spot:
            return ClusterType.SPOT

        # No spot now and not time to start OD: wait (NONE), optionally reposition to a better region
        # Avoid rapid thrashing via cooldown
        if len(self._region_obs) > 1:
            best_region = self._best_region(prefer_current=current_region)
            if (
                best_region != current_region
                and (now - self._last_region_switch_time) >= self._switch_cooldown_seconds
            ):
                self.env.switch_region(best_region)
                self._last_region_switch_time = now

        return ClusterType.NONE