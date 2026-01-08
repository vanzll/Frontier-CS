import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "deadline_safety_buffer_v1"

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

        self._committed_to_od = False
        self._sum_done = 0.0
        self._last_len = 0
        return self

    def _update_done_sum(self):
        n = len(self.task_done_time)
        if n > self._last_len:
            # Sum only new segments to keep O(1) average per step.
            s = 0.0
            for i in range(self._last_len, n):
                s += self.task_done_time[i]
            self._sum_done += s
            self._last_len = n

    def _od_time_to_finish_now(self, last_cluster_type: ClusterType, remain_work: float) -> float:
        if remain_work <= 0.0:
            return 0.0
        dt = self.env.gap_seconds
        # Overhead if switching to/on OD now.
        # If already on OD, no new overhead.
        H = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else self.restart_overhead

        # Number of full steps entirely consumed by overhead
        n_full_ov = int(H // dt)
        ov_rem = H - n_full_ov * dt  # remainder in [0, dt)
        time = n_full_ov * dt

        # First step after consuming full overhead blocks:
        # This step incurs ov_rem overhead and yields (dt - ov_rem) work.
        step_work = dt - ov_rem
        # We will take at least one step for any positive remaining work.
        time += dt
        remain_after_first = max(0.0, remain_work - step_work)

        if remain_after_first <= 0.0:
            return time

        # Additional full steps needed to finish remaining work
        n_add = int(math.ceil(remain_after_first / dt))
        time += n_add * dt
        return time

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update progress cache
        self._update_done_sum()

        remaining_work = max(0.0, self.task_duration - self._sum_done)
        if remaining_work <= 0.0:
            return ClusterType.NONE

        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        time_left = self.deadline - self.env.elapsed_seconds
        dt = self.env.gap_seconds

        # Compute guaranteed OD finish time if switching now
        t_od_now = self._od_time_to_finish_now(last_cluster_type, remaining_work)

        # Safety margin: ensure we can afford to wait up to one step with zero progress.
        if time_left <= t_od_now + dt:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        # No spot available and still safe to wait.
        return ClusterType.NONE