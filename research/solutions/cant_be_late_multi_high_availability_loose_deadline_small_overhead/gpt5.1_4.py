import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline guarantee and cost minimization."""

    NAME = "cant_be_late_mr_scheduler"

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
        self._internal_initialized = False
        return self

    # --- Internal helpers -------------------------------------------------

    def _initialize_internal_state(self) -> None:
        if getattr(self, "_internal_initialized", False):
            return
        self._internal_initialized = True

        # Environment / problem parameters
        self.num_regions = self.env.get_num_regions()
        self.region_avail = [0] * self.num_regions
        self.region_total = [0] * self.num_regions
        self.steps_seen = 0

        # Safety margin used for deciding latest safe switch to on-demand.
        # Derived to ensure we never miss deadline even in worst-case single-step loss.
        self._margin = self.env.gap_seconds + self.restart_overhead

        # Track whether we've committed to on-demand-only mode.
        self._force_on_demand = False

        # Cached accumulated work to avoid O(n^2) summation over task_done_time.
        lst = getattr(self, "task_done_time", [])
        self._last_task_done_len = len(lst)
        self._cached_work_done = float(sum(lst)) if lst else 0.0

    def _update_cached_work_done(self) -> None:
        lst = self.task_done_time
        n = len(lst)
        last_n = self._last_task_done_len
        if n > last_n:
            inc = 0.0
            for i in range(last_n, n):
                inc += lst[i]
            self._cached_work_done += inc
            self._last_task_done_len = n

    def _choose_next_region(self) -> int:
        """Choose region for the next step using UCB1 over observed spot availability."""
        n_regions = self.num_regions

        # Ensure every region is explored at least once.
        for idx in range(n_regions):
            if self.region_total[idx] == 0:
                return idx

        if self.steps_seen <= 1:
            return self.env.get_current_region()

        log_t = math.log(self.steps_seen)
        best_idx = 0
        best_score = float("-inf")
        for idx in range(n_regions):
            count = self.region_total[idx]
            if count == 0:
                score = float("inf")
            else:
                mean = self.region_avail[idx] / count
                bonus = math.sqrt(2.0 * log_t / count)
                score = mean + bonus
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    # --- Core decision logic ----------------------------------------------

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._initialize_internal_state()
        self._update_cached_work_done()

        # Remaining work and slack (seconds)
        remaining_work = self.task_duration - self._cached_work_done
        if remaining_work <= 0.0:
            # Task is complete; incur no more cost.
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        remaining_slack = self.deadline - elapsed
        if remaining_slack <= 0.0:
            # Deadline already passed; further work can't help.
            return ClusterType.NONE

        # Update region statistics using current observation.
        curr_region = self.env.get_current_region()
        if 0 <= curr_region < self.num_regions:
            self.region_total[curr_region] += 1
            if has_spot:
                self.region_avail[curr_region] += 1
            self.steps_seen += 1

        H = self.restart_overhead

        # Decide if we must commit to on-demand to guarantee deadline.
        if not self._force_on_demand:
            # If from now we used only on-demand, time needed <= remaining_work + H.
            # We allow at most one more worst-case step loss (gap + H) before switching.
            # So we switch when remaining_slack is no more than this safe bound.
            if remaining_slack <= remaining_work + H + self._margin:
                self._force_on_demand = True

        if self._force_on_demand:
            # Once committed, stay on on-demand until completion.
            return ClusterType.ON_DEMAND

        # Pre-commit phase: prefer spot, otherwise idle and search better region.

        if has_spot:
            # Spot available here; run on it.
            return ClusterType.SPOT

        # No spot in current region: idle this step and move to a more promising region.
        next_region = self._choose_next_region()
        if next_region != curr_region:
            self.env.switch_region(next_region)
        return ClusterType.NONE