import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "threshold_switch"

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

        # Initialize strategy state
        self.step_count = 0
        self.consecutive_no_spot = 0
        self.last_switch_step = -10  # ensures we can switch early

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self.step_count += 1

        # Compute remaining work and time left
        completed = sum(self.task_done_time)
        remaining_work = self.task_duration - completed
        time_left = self.deadline - self.env.elapsed_seconds

        # Threshold for switching to on‑demand
        threshold = remaining_work + self.restart_overhead
        eps = 1e-9

        if time_left <= threshold + eps:
            return ClusterType.ON_DEMAND

        # If currently in a restart overhead, wait
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE

        # Spot available -> use it
        if has_spot:
            self.consecutive_no_spot = 0
            return ClusterType.SPOT

        # No spot in current region
        self.consecutive_no_spot += 1

        # Consider switching region if:
        # - no spot for at least 2 consecutive steps
        # - haven't switched in the last 2 steps
        # - not in a restart overhead (already handled above)
        if (self.consecutive_no_spot >= 2 and
                (self.step_count - self.last_switch_step) > 2):
            next_region = (self.env.get_current_region() + 1) % self.env.get_num_regions()
            self.env.switch_region(next_region)
            self.consecutive_no_spot = 0
            self.last_switch_step = self.step_count
            # After switching we incur overhead, so do not run this step
            return ClusterType.NONE

        # If we've waited 3 steps without spot, fall back to on‑demand
        if self.consecutive_no_spot >= 3:
            return ClusterType.ON_DEMAND

        # Otherwise wait (NONE) hoping spot returns
        return ClusterType.NONE