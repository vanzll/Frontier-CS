import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "risk_aware_multiregion_v1"

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

        # Internal state initialization
        self._initialized = False
        self._committed_to_od = False
        self._cached_len = 0
        self._cached_work_done = 0.0
        self._no_spot_streak = []
        self._region_success = []
        self._region_fail = []
        self._scan_next_idx = 0
        return self

    def _init_internal(self):
        if self._initialized:
            return
        try:
            num_regions = self.env.get_num_regions()
        except Exception:
            num_regions = 1
        self._num_regions = max(1, int(num_regions))
        self._no_spot_streak = [0] * self._num_regions
        # Initialize with Beta(1,1) priors
        self._region_success = [1.0] * self._num_regions
        self._region_fail = [1.0] * self._num_regions
        try:
            cur = self.env.get_current_region()
        except Exception:
            cur = 0
        self._scan_next_idx = (cur + 1) % self._num_regions
        # Initialize cached work done
        self._cached_len = len(self.task_done_time)
        if self._cached_len > 0:
            self._cached_work_done = float(sum(self.task_done_time))
        else:
            self._cached_work_done = 0.0
        self._initialized = True

    def _update_progress_cache(self):
        l = len(self.task_done_time)
        if l != self._cached_len:
            # Usually increments by 1; safe to sum slice
            add = sum(self.task_done_time[self._cached_len : l])
            self._cached_work_done += float(add)
            self._cached_len = l

    def _select_region_on_unavailable(self, current_region: int):
        # Prefer region with highest empirical success rate (Beta prior 1,1)
        # If best is current, rotate to next region to explore; otherwise jump to best.
        best_idx = current_region
        best_score = -1.0
        for i in range(self._num_regions):
            s = self._region_success[i]
            f = self._region_fail[i]
            score = s / (s + f)  # posterior mean with prior 1,1
            if score > best_score:
                best_score = score
                best_idx = i

        target = best_idx
        if target == current_region:
            # Rotate to next to avoid being stuck
            target = self._scan_next_idx
            self._scan_next_idx = (self._scan_next_idx + 1) % self._num_regions
        if target != current_region:
            try:
                self.env.switch_region(target)
            except Exception:
                pass

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._init_internal()

        # Update cached work progress
        self._update_progress_cache()

        # Update region availability stats
        try:
            current_region = self.env.get_current_region()
        except Exception:
            current_region = 0
        if has_spot:
            self._region_success[current_region] += 1.0
            self._no_spot_streak[current_region] = 0
        else:
            self._region_fail[current_region] += 1.0
            self._no_spot_streak[current_region] += 1

        # Quick completion check
        work_done = self._cached_work_done
        task_total = float(self.task_duration)
        work_remaining = max(0.0, task_total - work_done)
        if work_remaining <= 0.0:
            return ClusterType.NONE

        # Timing parameters
        step = float(self.env.gap_seconds)
        time_left = max(0.0, float(self.deadline) - float(self.env.elapsed_seconds))
        overhead = float(self.restart_overhead)

        # If already committed to On-Demand, stick with it
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # If Spot available, use it by default
        if has_spot:
            # Optional safety: if we are far behind and even OD cannot save us, still run Spot to minimize cost
            return ClusterType.SPOT

        # Spot not available: decide to wait or commit to On-Demand
        # Slack with respect to committing to OD now (non-negative means can still finish with OD)
        slack_commit = time_left - (work_remaining + overhead)

        # If we can't afford to wait a full step, commit to On-Demand now
        if slack_commit < step:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, we can wait for Spot and try another region
        self._select_region_on_unavailable(current_region)
        return ClusterType.NONE