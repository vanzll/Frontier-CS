import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v7"

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
        self._committed_to_od = False
        self._pause_streak = 0
        self._max_pause_steps = 1  # pause at most 1 step when spot unavailable
        return self

    def _total_task_duration(self) -> float:
        td = getattr(self, "task_duration", 0.0)
        if isinstance(td, (list, tuple)):
            return float(td[0]) if td else 0.0
        return float(td)

    def _remaining_work(self) -> float:
        total = self._total_task_duration()
        done = sum(self.task_done_time) if hasattr(self, "task_done_time") else 0.0
        remaining = max(0.0, total - done)
        return remaining

    def _safe_od_time(self, last_cluster_type: ClusterType) -> float:
        # Conservative time needed (seconds) if we switch to OD now and run to completion.
        remaining = self._remaining_work()
        if remaining <= 0:
            return 0.0
        overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead
        # Add a buffer of one step for discretization effects.
        buffer_rounding = self.env.gap_seconds
        return remaining + overhead + buffer_rounding

    def _maybe_commit(self, last_cluster_type: ClusterType) -> None:
        if self._committed_to_od:
            return
        time_remaining = self.deadline - self.env.elapsed_seconds
        safe_time = self._safe_od_time(last_cluster_type)
        if time_remaining <= safe_time:
            self._committed_to_od = True

    def _select_next_region(self) -> int:
        n = self.env.get_num_regions()
        if n <= 1:
            return self.env.get_current_region()
        curr = self.env.get_current_region()
        return (curr + 1) % n

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update commitment decision
        self._maybe_commit(last_cluster_type)

        # If not committed and current spot is unavailable, consider switching region for next step
        if not has_spot and not self._committed_to_od:
            next_region = self._select_next_region()
            if next_region != self.env.get_current_region():
                self.env.switch_region(next_region)

        # If committed to OD, always use ON_DEMAND
        if self._committed_to_od:
            self._pause_streak = 0
            return ClusterType.ON_DEMAND

        # Prefer SPOT when available
        if has_spot:
            self._pause_streak = 0
            return ClusterType.SPOT

        # Spot unavailable; decide between pausing briefly, or using OD
        time_remaining = self.deadline - self.env.elapsed_seconds
        safe_time = self._safe_od_time(last_cluster_type)

        # If we must, switch to OD immediately
        if time_remaining <= safe_time:
            self._pause_streak = 0
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise, wait briefly for spot to come back
        if self._pause_streak < self._max_pause_steps:
            self._pause_streak += 1
            return ClusterType.NONE

        # If waited already, use OD temporarily (but not committed, may switch back later)
        self._pause_streak = 0
        return ClusterType.ON_DEMAND