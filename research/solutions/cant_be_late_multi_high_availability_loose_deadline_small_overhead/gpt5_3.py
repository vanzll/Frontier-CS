import json
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cbc_multiregion_heuristic_v1"

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

        # Internal state
        self._od_locked: bool = False
        self._last_od_commit_time: Optional[float] = None
        self._initialized_regions: bool = False
        self._region_count: int = 0
        self._obs = []
        self._avail = []
        self._rr_ptr = 0

        # Priors for availability estimation (Beta prior)
        self._prior_a = 6.0
        self._prior_b = 2.0

        # Threshold for switching to another region when waiting
        self._score_diff_threshold = 0.05

        return self

    def _init_region_state(self):
        if self._initialized_regions:
            return
        self._region_count = self.env.get_num_regions()
        self._obs = [0] * self._region_count
        self._avail = [0] * self._region_count
        self._rr_ptr = self.env.get_current_region() % self._region_count
        self._initialized_regions = True

    def _safety_margin(self) -> float:
        # Safety margin to account for discretization and small uncertainties
        return max(self.env.gap_seconds, 2.0 * self.restart_overhead)

    def _region_score(self, idx: int) -> float:
        return (self._avail[idx] + self._prior_a) / (self._obs[idx] + self._prior_a + self._prior_b)

    def _select_best_region(self, current: int) -> int:
        # Prefer exploring unseen regions first while waiting
        for i in range(self._region_count):
            if self._obs[i] == 0 and i != current:
                return i

        # Otherwise pick the highest-score region if it's meaningfully better
        scores = [self._region_score(i) for i in range(self._region_count)]
        best_idx = max(range(self._region_count), key=lambda i: scores[i])
        current_score = scores[current]
        best_score = scores[best_idx]

        if best_idx != current and (best_score - current_score) > self._score_diff_threshold:
            return best_idx

        # Otherwise round-robin to avoid being stuck
        for k in range(1, self._region_count + 1):
            idx = (self._rr_ptr + k) % self._region_count
            if idx != current:
                self._rr_ptr = idx
                return idx
        return current

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_region_state()

        current_region = self.env.get_current_region()

        # Update availability stats for the region we are currently observing
        self._obs[current_region] += 1
        if has_spot:
            self._avail[current_region] += 1

        # If already finished, do nothing
        rem_work = max(0.0, self.task_duration - sum(self.task_done_time))
        if rem_work <= 0.0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        if time_left <= 0.0:
            # Already out of time; still try OD to mitigate penalty
            self._od_locked = True
            return ClusterType.ON_DEMAND

        # Estimate time to finish on On-Demand starting now
        od_overhead_now = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead
        t_needed_od_now = od_overhead_now + rem_work
        margin = self._safety_margin()

        # If we've committed to On-Demand, keep using it to avoid extra overheads
        if self._od_locked:
            return ClusterType.ON_DEMAND

        # If deadline is tight, commit to OD now
        if time_left <= t_needed_od_now + margin:
            self._od_locked = True
            self._last_od_commit_time = self.env.elapsed_seconds
            return ClusterType.ON_DEMAND

        # Otherwise, prefer Spot if available
        if has_spot:
            return ClusterType.SPOT

        # Spot not available: decide to wait (NONE) or commit to OD
        # Check if we can afford waiting one more step and still finish on OD
        t_needed_od_after_wait = self.restart_overhead + rem_work
        if (time_left - self.env.gap_seconds) > (t_needed_od_after_wait + margin):
            # We can wait safely; switch to a promising region to increase chances next step
            next_region = self._select_best_region(current_region)
            if next_region != current_region:
                self.env.switch_region(next_region)
            return ClusterType.NONE

        # Can't wait any longer; commit to OD
        self._od_locked = True
        self._last_od_commit_time = self.env.elapsed_seconds
        return ClusterType.ON_DEMAND