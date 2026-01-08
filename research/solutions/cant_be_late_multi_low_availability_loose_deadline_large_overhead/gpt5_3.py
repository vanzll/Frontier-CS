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
        self._last_done_len = 0
        self._done_sum = 0.0
        self._last_switch_time = -1.0

        # Region stats for simple adaptive rotation during wait
        num = self.env.get_num_regions()
        self._region_avail_count = [0] * num
        self._region_steps_count = [0] * num
        self._target_region = self.env.get_current_region()

        return self

    def _update_progress_cache(self):
        n = len(self.task_done_time)
        if n != self._last_done_len:
            # Only sum new segments
            new_segments = self.task_done_time[self._last_done_len:]
            if new_segments:
                self._done_sum += sum(new_segments)
            self._last_done_len = n

    def _estimate_remaining_work(self):
        self._update_progress_cache()
        rem = self.task_duration - self._done_sum
        return rem if rem > 0 else 0.0

    def _record_region_observation(self, has_spot: bool):
        idx = self.env.get_current_region()
        if 0 <= idx < len(self._region_steps_count):
            self._region_steps_count[idx] += 1
            if has_spot:
                self._region_avail_count[idx] += 1

    def _choose_best_region(self):
        # Simple Bayesian smoothing: (successes + 1) / (trials + 2)
        # Choose region with highest smoothed availability.
        num = self.env.get_num_regions()
        best_idx = self.env.get_current_region()
        best_score = -1.0
        for i in range(num):
            s = self._region_avail_count[i]
            t = self._region_steps_count[i]
            score = (s + 1.0) / (t + 2.0)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _maybe_rotate_region_while_waiting(self):
        # Switch to region with highest observed availability while waiting
        # Limit to at most once per step
        if self._last_switch_time == self.env.elapsed_seconds:
            return
        target = self._choose_best_region()
        cur = self.env.get_current_region()
        if target != cur:
            self.env.switch_region(target)
            self._last_switch_time = self.env.elapsed_seconds
            self._target_region = target

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Record observation for current region
        self._record_region_observation(has_spot)

        # If already on OD, keep committed
        if self.env.cluster_type == ClusterType.ON_DEMAND:
            self._committed_to_od = True

        remaining_work = self._estimate_remaining_work()
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_left = self.deadline - self.env.elapsed_seconds
        if time_left <= 0:
            # Out of time: run OD as last resort
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        gap = self.env.gap_seconds
        # Safety buffer to avoid cutting it too close near the deadline
        safety_buffer = max(2.0 * gap, 3.0 * self.restart_overhead)

        # Overhead to pay if we commit to OD now
        if self._committed_to_od or self.env.cluster_type == ClusterType.ON_DEMAND:
            overhead_to_commit = self.remaining_restart_overhead
        else:
            overhead_to_commit = self.restart_overhead

        od_time_needed = remaining_work + overhead_to_commit

        # Commit to On-Demand if we are close to the deadline
        if not self._committed_to_od and time_left <= od_time_needed + safety_buffer:
            self._committed_to_od = True

        if self._committed_to_od:
            # Stay in current region to avoid extra restarts
            return ClusterType.ON_DEMAND

        # Not committed to OD; try to use Spot if available
        if has_spot:
            self._target_region = self.env.get_current_region()
            return ClusterType.SPOT

        # Spot not available; decide to wait or switch to OD based on slack
        slack = time_left - (remaining_work + self.restart_overhead + safety_buffer)
        if slack > 0:
            # Wait to save cost; while waiting, rotate to best region based on past observations
            self._maybe_rotate_region_while_waiting()
            return ClusterType.NONE
        else:
            # Slack too low: commit to OD to guarantee completion
            self._committed_to_od = True
            return ClusterType.ON_DEMAND