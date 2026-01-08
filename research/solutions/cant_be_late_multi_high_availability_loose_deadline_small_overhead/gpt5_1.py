import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_mr"

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
        self._od_lock = False
        self._margin_seconds = None
        self._progress_cache = 0.0
        self._last_seen_len = 0
        self._eps = 1e-9

        return self

    def _update_progress_cache(self):
        # Incrementally update cached progress to avoid O(n) summations every step.
        if self._last_seen_len < len(self.task_done_time):
            new_sum = 0.0
            # Sum only new segments appended since last call.
            for i in range(self._last_seen_len, len(self.task_done_time)):
                new_sum += self.task_done_time[i]
            self._progress_cache += new_sum
            self._last_seen_len = len(self.task_done_time)

    def _ensure_margin(self):
        if self._margin_seconds is None:
            # Safety margin to absorb preemptions/step discretization.
            # Clamp between 15 minutes and 1 hour, scaling with gap and overhead.
            gap = getattr(self.env, "gap_seconds", 60.0)
            m = self.restart_overhead
            base = max(6.0 * m, 2.0 * gap)
            if base < 900.0:
                base = 900.0
            if base > 3600.0:
                base = 3600.0
            self._margin_seconds = base

    def _remaining_work(self):
        self._update_progress_cache()
        remaining = self.task_duration - self._progress_cache
        if remaining < 0.0:
            remaining = 0.0
        return remaining

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize margin lazily
        self._ensure_margin()

        # Compute time and work stats
        gap = self.env.gap_seconds
        time_left = self.deadline - self.env.elapsed_seconds
        rem_work = self._remaining_work()

        # If all work is done or no time left, do nothing
        if rem_work <= self._eps or time_left <= self._eps:
            return ClusterType.NONE

        # If there's outstanding restart overhead, prefer to wait it out without paying
        if getattr(self, "remaining_restart_overhead", 0.0) > self._eps:
            return ClusterType.NONE

        # Determine if we must lock to on-demand to guarantee completion
        # Add one restart_overhead for switching to OD (worst-case)
        must_lock_to_od = time_left <= (rem_work + self.restart_overhead + self._margin_seconds + self._eps)
        if not self._od_lock and must_lock_to_od:
            self._od_lock = True

        if self._od_lock:
            # Once locked, stay on OD for guaranteed completion
            return ClusterType.ON_DEMAND

        # If spot is available, use it
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable and not locked: decide to wait or go OD
        # If enough slack remains, wait (NONE) to save cost; otherwise lock and go OD
        if time_left > (rem_work + self.restart_overhead + self._margin_seconds + self._eps):
            return ClusterType.NONE

        # Time is limited; switch to OD to ensure completion
        self._od_lock = True
        return ClusterType.ON_DEMAND