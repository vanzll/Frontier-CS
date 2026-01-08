import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_safe_v1"

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
        self._committed_od = False
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # If we've already committed to on-demand, keep it to ensure timely completion.
        if self._committed_od:
            return ClusterType.ON_DEMAND

        # Compute remaining work and time.
        gap = float(self.env.gap_seconds)
        elapsed = float(self.env.elapsed_seconds)
        deadline = float(self.deadline)
        ro = float(self.restart_overhead)

        done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = max(0.0, float(self.task_duration) - done)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_left = max(0.0, deadline - elapsed)

        # If we're already running on-demand (e.g., from previous step), commit and continue.
        if last_cluster_type == ClusterType.ON_DEMAND:
            self._committed_od = True
            return ClusterType.ON_DEMAND

        # Decide using safety checks:
        # 1) If Spot is available, run Spot only if after this step we can still finish with OD (worst-case).
        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                overhead_this_step = float(self.remaining_restart_overhead)
            else:
                overhead_this_step = ro
            effective_work = max(0.0, gap - overhead_this_step)
            time_left_after_step = max(0.0, time_left - gap)
            worst_case_needed_after = max(0.0, remaining_work - effective_work) + ro

            if time_left_after_step >= worst_case_needed_after:
                return ClusterType.SPOT
            else:
                # Not safe to keep using Spot; commit to On-Demand now.
                self._committed_od = True
                return ClusterType.ON_DEMAND

        # 2) If Spot is not available, pause if safe; otherwise commit to on-demand now.
        time_left_after_step = max(0.0, time_left - gap)
        safe_to_pause = time_left_after_step >= (remaining_work + ro)

        if safe_to_pause:
            return ClusterType.NONE
        else:
            self._committed_od = True
            return ClusterType.ON_DEMAND