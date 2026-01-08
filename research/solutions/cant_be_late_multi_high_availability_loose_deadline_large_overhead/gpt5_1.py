import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Always continue on ON_DEMAND once chosen to avoid extra overhead/risk.
        if last_cluster_type == ClusterType.ON_DEMAND:
            return ClusterType.ON_DEMAND

        remaining_work = max(0.0, self.task_duration - sum(self.task_done_time))
        time_left = self.deadline - self.env.elapsed_seconds

        # If we start ON_DEMAND now, we pay one restart overhead unless we are already on OD.
        time_to_finish_on_od = remaining_work + self.restart_overhead

        # Commit to ON_DEMAND if we're at or past the safe fallback threshold.
        if time_left <= time_to_finish_on_od:
            return ClusterType.ON_DEMAND

        # Otherwise, prefer SPOT when available; pause if not.
        if has_spot:
            return ClusterType.SPOT
        return ClusterType.NONE