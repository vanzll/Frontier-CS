import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy that prefers Spot with a safe on-demand fallback."""

    NAME = "cant_be_late_v1"

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

        # Internal state for incremental tracking of work done.
        self._cached_work_done = 0.0
        self._last_task_done_length = 0
        self._last_task_done_list_id = id(self.task_done_time) if hasattr(self, "task_done_time") else None

        # Flag indicating we've switched to always using on-demand to guarantee completion.
        self._bail_to_on_demand = False

        # Configure how much slack (in seconds) we require before switching permanently to on-demand.
        restart_overhead = getattr(self, "restart_overhead", float(config["overhead"]) * 3600.0)
        gap = getattr(self.env, "gap_seconds", 1.0)

        # Minimum slack needed to safely absorb:
        # - one worst-case "wasted" step (gap + restart_overhead)
        # - plus one additional restart_overhead due to the switch to on-demand.
        # Add a modest safety margin.
        min_safe_slack = gap + 2.0 * restart_overhead

        # Use a slightly more conservative threshold but still small relative to typical slack.
        self._od_switch_slack_seconds = max(
            min_safe_slack * 1.5,          # 1.5x theoretical minimum
            5.0 * restart_overhead,        # or 5x overhead
            10.0 * 60.0                    # at least 10 minutes
        )

        return self

    def _update_work_done_cache(self) -> None:
        """Incrementally update cached total work done based on task_done_time list."""
        td_list = getattr(self, "task_done_time", None)
        if td_list is None:
            self._cached_work_done = 0.0
            self._last_task_done_length = 0
            self._last_task_done_list_id = None
            return

        current_id = id(td_list)

        # If the underlying list object changed (e.g., env reset), recompute from scratch.
        if current_id != self._last_task_done_list_id:
            total = 0.0
            for v in td_list:
                total += v
            self._cached_work_done = total
            self._last_task_done_length = len(td_list)
            self._last_task_done_list_id = current_id
            return

        l_prev = self._last_task_done_length
        l_now = len(td_list)

        # List shrank (e.g., new episode); recompute from scratch.
        if l_now < l_prev:
            total = 0.0
            for v in td_list:
                total += v
            self._cached_work_done = total
            self._last_task_done_length = l_now
            # list id unchanged; keep
        elif l_now > l_prev:
            # Sum only the newly appended segments.
            total_new = 0.0
            for i in range(l_prev, l_now):
                total_new += td_list[i]
            self._cached_work_done += total_new
            self._last_task_done_length = l_now
        # If lengths are equal, nothing to do.

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update cached work done.
        self._update_work_done_cache()
        work_done = self._cached_work_done

        task_duration = self.task_duration
        remaining_work = task_duration - work_done

        # If task already finished, do nothing.
        if remaining_work <= 0.0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        time_remaining = deadline - elapsed

        # If somehow past deadline, best effort: always use on-demand.
        if time_remaining <= 0.0:
            self._bail_to_on_demand = True
            return ClusterType.ON_DEMAND

        slack = time_remaining - remaining_work  # extra time beyond ideal on-demand run

        # Decide if we must permanently switch to on-demand to guarantee completion.
        if not self._bail_to_on_demand:
            if slack <= self._od_switch_slack_seconds:
                self._bail_to_on_demand = True

        if self._bail_to_on_demand:
            return ClusterType.ON_DEMAND

        # Spot-preferred phase.
        if has_spot:
            return ClusterType.SPOT

        # No spot currently available and plenty of slack: wait (no cost).
        return ClusterType.NONE