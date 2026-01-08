import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
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
        # Internal state
        self._committed_to_od = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Compute remaining work and time
        work_done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = max(0.0, self.task_duration - work_done)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_remaining = max(0.0, self.deadline - self.env.elapsed_seconds)

        # Determine overhead if switching to OD now
        overhead_if_start_od = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead

        # Safety margin to protect against discretization/overhead reset
        gap = getattr(self.env, "gap_seconds", 3600.0)
        safety_margin = min(max(self.restart_overhead * 2.0, gap * 0.1), gap * 0.5)

        # Latest time to safely start OD now to guarantee finish
        required_time_on_od = remaining_work + overhead_if_start_od

        # Commit to on-demand if we are close to deadline
        if not self._committed_to_od and time_remaining <= required_time_on_od + safety_margin:
            self._committed_to_od = True

        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Not committed to OD yet; prefer Spot when available, else wait
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable; if we must act to ensure deadline, commit to OD
        if time_remaining <= required_time_on_od + safety_margin:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Otherwise wait for spot to return
        return ClusterType.NONE