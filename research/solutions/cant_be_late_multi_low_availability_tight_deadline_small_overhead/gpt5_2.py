import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "deadline_guard_rr_v1"

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

        # Internal policy state
        self._od_committed = False
        self._last_region_switch_step = -1
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Compute remaining work and time
        done = sum(self.task_done_time) if self.task_done_time else 0.0
        remaining_work = max(self.task_duration - done, 0.0)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        if time_left <= 0.0:
            # Out of time; best effort to run OD
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # If we've already committed to On-Demand, stick with it
        if self._od_committed:
            return ClusterType.ON_DEMAND

        # Conservative slack accounting:
        # We assume that when we eventually fall back to OD, we'll pay one restart overhead.
        # Slack time is how much we can idle while still being able to finish with OD.
        slack_time = time_left - remaining_work - self.restart_overhead

        # If we already cannot afford to wait any longer, commit to OD now.
        if slack_time <= 0.0:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Heuristic early-commit when spot is unavailable and the next idle step would exceed slack.
        # This guards against step discretization effects.
        if not has_spot and slack_time <= self.env.gap_seconds:
            self._od_committed = True
            return ClusterType.ON_DEMAND

        # Prefer SPOT whenever available
        if has_spot:
            return ClusterType.SPOT

        # Otherwise, wait (NONE) and opportunistically rotate regions to try to find spot next step.
        num_regions = self.env.get_num_regions()
        if num_regions > 1:
            # Round-robin switch to the next region only when we're idling.
            next_region = (self.env.get_current_region() + 1) % num_regions
            self.env.switch_region(next_region)

        return ClusterType.NONE