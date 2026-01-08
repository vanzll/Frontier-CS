import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy for cant-be-late problem."""

    NAME = "cant_be_late_multi_region_v1"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize the solution from spec_path config."""
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Initialize region statistics.
        try:
            self.num_regions = self.env.get_num_regions()
        except Exception:
            self.num_regions = 1

        self.region_total_steps = [0] * self.num_regions
        self.region_spot_available_steps = [0] * self.num_regions

        # Track work done incrementally to avoid O(N^2) summations.
        self._work_done = 0.0
        self._last_done_idx = 0

        # Conservative ON_DEMAND start time to guarantee completion even if
        # no work is done before this time.
        gap = getattr(self.env, "gap_seconds", 0.0)
        # Ensure non-negative.
        self.od_start_time = max(
            0.0,
            self.deadline - (self.task_duration + self.restart_overhead + gap),
        )
        self.force_on_demand = False

        return self

    def _update_work_done(self) -> None:
        """Incrementally track total work done from task_done_time segments."""
        task_done = getattr(self, "task_done_time", None)
        if task_done is None:
            return
        n = len(task_done)
        if n > self._last_done_idx:
            total_added = 0.0
            # Sum only new segments since last step.
            for i in range(self._last_done_idx, n):
                total_added += task_done[i]
            self._work_done += total_added
            self._last_done_idx = n

    def _choose_best_region(self, current_region: int) -> int:
        """Select region with highest estimated spot availability."""
        best_region = current_region
        best_score = -1.0
        for i in range(self.num_regions):
            total = self.region_total_steps[i]
            avail = self.region_spot_available_steps[i]
            if total == 0:
                # Prioritize unexplored regions.
                score = 2.0
            else:
                # Empirical availability.
                score = avail / total
            if score > best_score:
                best_score = score
                best_region = i
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Update internal accounting of work done.
        self._update_work_done()
        if self._work_done >= self.task_duration:
            # Task already completed; avoid incurring any additional cost.
            return ClusterType.NONE

        # Update region statistics using the current observation.
        try:
            current_region = self.env.get_current_region()
        except Exception:
            current_region = 0

        if 0 <= current_region < self.num_regions:
            self.region_total_steps[current_region] += 1
            if has_spot:
                self.region_spot_available_steps[current_region] += 1

        # Decide whether to switch permanently to on-demand to guarantee deadline.
        if not self.force_on_demand:
            elapsed = getattr(self.env, "elapsed_seconds", 0.0)
            if elapsed >= self.od_start_time:
                self.force_on_demand = True

        if self.force_on_demand:
            # From this point onward, always use on-demand instances.
            return ClusterType.ON_DEMAND

        # Before the on-demand commit point, prefer spot instances.
        if has_spot:
            return ClusterType.SPOT

        # No spot available in the current region.
        # Move to a region with historically better availability and pause.
        if self.num_regions > 1:
            best_region = self._choose_best_region(current_region)
            if best_region != current_region:
                try:
                    self.env.switch_region(best_region)
                except Exception:
                    # If switching fails for any reason, just stay.
                    pass

        # Pause to avoid paying for expensive on-demand while we still have slack.
        return ClusterType.NONE