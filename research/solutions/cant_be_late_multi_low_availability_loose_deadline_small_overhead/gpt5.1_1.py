import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
        """
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
        """
        Decide next action based on current state.

        Available attributes:
        - self.env.get_current_region(): Get current region index
        - self.env.get_num_regions(): Get total number of regions
        - self.env.switch_region(idx): Switch to region by index
        - self.env.elapsed_seconds: Current time elapsed
        - self.task_duration: Total task duration needed (seconds)
        - self.deadline: Deadline time (seconds)
        - self.restart_overhead: Restart overhead (seconds)
        - self.task_done_time: List of completed work segments
        - self.remaining_restart_overhead: Current pending overhead

        Returns: ClusterType.SPOT, ClusterType.ON_DEMAND, or ClusterType.NONE
        """
        env = self.env
        elapsed = env.elapsed_seconds

        # Initialize or reset cached progress tracking across episodes.
        if not hasattr(self, "_cached_done"):
            self._cached_done = 0.0
            self._cached_len = 0
            self._last_elapsed = elapsed
        else:
            # Detect environment reset (new scenario).
            if elapsed < self._last_elapsed or len(self.task_done_time) < self._cached_len:
                self._cached_done = 0.0
                self._cached_len = 0

        # Incrementally update total work done.
        segs = self.task_done_time
        n = len(segs)
        if n > self._cached_len:
            add_sum = 0.0
            for i in range(self._cached_len, n):
                add_sum += segs[i]
            self._cached_done += add_sum
            self._cached_len = n
        self._last_elapsed = elapsed

        done_work = self._cached_done

        # Total task duration (seconds).
        td = getattr(self, "task_duration", None)
        if isinstance(td, (list, tuple)):
            total_duration = float(td[0])
        else:
            total_duration = float(td)

        # Deadline (seconds).
        dl = getattr(self, "deadline", None)
        if isinstance(dl, (list, tuple)):
            deadline = float(dl[0])
        else:
            deadline = float(dl)

        remaining_work = max(0.0, total_duration - done_work)
        if remaining_work <= 0.0:
            # Task already finished; no need to run.
            return ClusterType.NONE

        gap = getattr(env, "gap_seconds", 1.0)
        if gap <= 0.0:
            gap = 1.0

        ro = getattr(self, "restart_overhead", 0.0)
        if isinstance(ro, (list, tuple)):
            restart_overhead = float(ro[0])
        else:
            restart_overhead = float(ro)

        # Safety buffer: finish at least one time step before the hard deadline.
        safety_buffer = gap

        # Effective remaining slack time available.
        slack = deadline - elapsed - safety_buffer

        # If we're already too close (or past) the effective deadline, use On-Demand.
        if slack <= 0.0:
            return ClusterType.ON_DEMAND

        # Worst-case additional delay from gambling one more step:
        # - Up to one full step of time (gap)
        # - Plus a restart overhead during that step
        # After that, if we then switch to On-Demand we may incur another restart overhead.
        # So total worst-case extra time we must account for is:
        #   delta (gap + restart_overhead) + (restart_overhead + remaining_work)
        delta = gap + restart_overhead
        threshold = delta + restart_overhead + remaining_work  # = gap + 2R + remaining_work

        if slack > threshold:
            # We can afford to risk one more step without committing to On-Demand.
            if has_spot:
                # Spot is cheap; use it whenever it's available and safe.
                return ClusterType.SPOT
            else:
                # Spot unavailable and plenty of slack left: wait to avoid expensive On-Demand.
                return ClusterType.NONE
        else:
            # Not enough slack to risk another step without On-Demand.
            return ClusterType.ON_DEMAND