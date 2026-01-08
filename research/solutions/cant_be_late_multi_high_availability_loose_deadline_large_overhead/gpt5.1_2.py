import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with conservative deadline guarantees."""

    NAME = "my_strategy"

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
        self.force_on_demand = False

        # Cached work-done tracking to avoid O(n^2) summations.
        self._work_done = 0.0
        self._last_task_done_len = 0

        # Multi-region handling
        try:
            self._num_regions = self.env.get_num_regions()
        except Exception:
            self._num_regions = 1
        self._spot_unavailable_steps = 0
        self._region_switch_threshold = 3  # steps of no-spot before switching region

        # Normalize scalar parameters
        task_duration = self.task_duration
        if isinstance(task_duration, (list, tuple)):
            task_duration = task_duration[0] if task_duration else 0.0
            self.task_duration = task_duration

        restart_overhead = self.restart_overhead
        if isinstance(restart_overhead, (list, tuple)):
            restart_overhead = restart_overhead[0] if restart_overhead else 0.0
            self.restart_overhead = restart_overhead

        # Safety margin for switching to On-Demand.
        gap = getattr(self.env, "gap_seconds", 0.0)
        self._max_diff_drop_per_step = gap + restart_overhead
        if self._max_diff_drop_per_step <= 0.0:
            self._max_diff_drop_per_step = 1.0
        # Conservative margin (upper bound on drop in "slack vs OD-time" between checks)
        self._safety_margin = 2.0 * self._max_diff_drop_per_step

        if not hasattr(self, "remaining_restart_overhead"):
            self.remaining_restart_overhead = 0.0

        self._last_diff = None  # for debugging/analysis if needed

        return self

    # Helper to incrementally update total work done.
    def _update_work_done(self) -> None:
        segments = getattr(self, "task_done_time", None)
        if segments is None:
            return
        n = len(segments)
        last_n = self._last_task_done_len
        if n > last_n:
            self._work_done += sum(segments[last_n:n])
            self._last_task_done_len = n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached progress.
        self._update_work_done()

        remaining_work = max(0.0, self.task_duration - self._work_done)

        # If job is effectively done, don't run anything more.
        if remaining_work <= 0.0:
            self.force_on_demand = True
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        slack_time = self.deadline - current_time

        # Upper bound on overhead if we immediately switch to ON_DEMAND.
        base_overhead = self.restart_overhead
        rem_overhead = getattr(self, "remaining_restart_overhead", 0.0) or 0.0
        overhead_for_commit = base_overhead if base_overhead >= rem_overhead else rem_overhead

        # If we've run out of slack, we must commit to On-Demand (even if it's too late).
        if slack_time <= 0.0:
            self.force_on_demand = True
        else:
            # diff = (time until deadline) - (time needed on OD from now)
            diff = slack_time - (overhead_for_commit + remaining_work)
            self._last_diff = diff
            if not self.force_on_demand and diff <= self._safety_margin:
                # Commit to On-Demand mode; never go back to Spot.
                self.force_on_demand = True

        if self.force_on_demand:
            # Always run On-Demand once committed.
            return ClusterType.ON_DEMAND

        # Still in Spot-preferred mode.
        if has_spot:
            # Spot available: use it.
            self._spot_unavailable_steps = 0
            return ClusterType.SPOT

        # No Spot in current region; consider switching regions and/or idling.
        self._spot_unavailable_steps += 1

        if self._num_regions > 1 and self._spot_unavailable_steps >= self._region_switch_threshold:
            try:
                current_region = self.env.get_current_region()
                new_region = (current_region + 1) % self._num_regions
                self.env.switch_region(new_region)
            except Exception:
                # If environment does not support region switching, ignore.
                pass
            self._spot_unavailable_steps = 0

        # Wait for Spot to become available again.
        return ClusterType.NONE