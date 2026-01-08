import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline-aware Spot/On-Demand usage."""

    NAME = "cant_be_late_multi_region_v1"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Custom state
        self._total_progress = 0.0
        self._last_seg_index = 0
        self._committed_on_demand = False

        # Extra safety buffer (seconds) beyond minimal required time for On-Demand.
        # This reduces risk from modeling/rounding issues while keeping cost low.
        self._safety_margin = 600.0  # 10 minutes

        return self

    def _update_progress_cache(self) -> None:
        """Incrementally maintain total progress from task_done_time list."""
        seg_list = self.task_done_time
        if not seg_list:
            return
        last_idx = self._last_seg_index
        cur_len = len(seg_list)
        if cur_len > last_idx:
            # Sum only new segments once; overall O(N) over the whole run.
            self._total_progress += sum(seg_list[last_idx:cur_len])
            self._last_seg_index = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        # Defensive initialization in case solve() was not called for some reason.
        if not hasattr(self, "_total_progress"):
            self._total_progress = 0.0
            self._last_seg_index = 0
            self._committed_on_demand = False
            self._safety_margin = 600.0

        # Efficiently update completed work amount.
        self._update_progress_cache()

        # Compute remaining work (seconds) and time left to deadline (seconds).
        remaining_work = self.task_duration - self._total_progress
        if remaining_work <= 0.0:
            # Task finished: no need to pay for more compute.
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds

        # If we are at/past the deadline, still try to run on-demand to minimize penalty.
        if time_left <= 0.0:
            return ClusterType.ON_DEMAND

        # Determine if we must now commit to On-Demand to guarantee finishing.
        # Need remaining_work seconds of compute plus at most one restart_overhead
        # for the final switch to On-Demand.
        required_time_for_safe_on_demand = remaining_work + self.restart_overhead
        slack = time_left - required_time_for_safe_on_demand

        if (not self._committed_on_demand) and (slack <= self._safety_margin):
            self._committed_on_demand = True

        if self._committed_on_demand:
            # Once committed, always run on On-Demand to avoid any further risk.
            return ClusterType.ON_DEMAND

        # Early (safe) phase: use Spot whenever available, otherwise wait for Spot.
        if has_spot:
            return ClusterType.SPOT

        # No Spot available and we still have comfortable slack: just wait.
        return ClusterType.NONE