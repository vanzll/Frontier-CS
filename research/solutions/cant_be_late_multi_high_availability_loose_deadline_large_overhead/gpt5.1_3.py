import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with deadline guarantee."""

    NAME = "cant_be_late_multi_region_v2"

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

        # Internal scheduling state
        self._committed_to_on_demand = False

        # Efficient cache for total completed work
        self._last_done_len = len(getattr(self, "task_done_time", []))
        if self._last_done_len > 0:
            self._last_done_sum = float(sum(self.task_done_time))
        else:
            self._last_done_sum = 0.0
        self._done_work = self._last_done_sum

        # Cache gap_seconds and corresponding safety margin
        if hasattr(self, "env") and hasattr(self.env, "gap_seconds"):
            self._gap_seconds = float(self.env.gap_seconds)
        else:
            self._gap_seconds = 0.0

        return self

    def _update_done_work_cache(self) -> None:
        """Incrementally maintain sum(self.task_done_time) in O(1) amortized time."""
        lst = self.task_done_time
        cur_len = len(lst)

        if cur_len > self._last_done_len:
            # Sum only newly added segments
            additional = 0.0
            for v in lst[self._last_done_len:cur_len]:
                additional += v
            self._last_done_sum += additional
            self._last_done_len = cur_len
        elif cur_len < self._last_done_len:
            # In unexpected case list shrank, recompute from scratch
            total = 0.0
            for v in lst:
                total += v
            self._last_done_sum = total
            self._last_done_len = cur_len

        self._done_work = self._last_done_sum

    def _get_remaining_work(self) -> float:
        """Return remaining task work in seconds."""
        self._update_done_work_cache()
        remaining = self.task_duration - self._done_work
        if remaining < 0.0:
            return 0.0
        return remaining

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
        # Ensure gap_seconds cached (in case env wasn't ready in solve)
        if self._gap_seconds == 0.0 and hasattr(self.env, "gap_seconds"):
            self._gap_seconds = float(self.env.gap_seconds)

        remaining_work = self._get_remaining_work()

        # If task already complete, don't run any more compute.
        if remaining_work <= 0.0:
            self._committed_to_on_demand = True
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed

        # If we're at or past deadline, just choose On-Demand (penalty is already decided).
        if time_left <= 0.0:
            self._committed_to_on_demand = True
            return ClusterType.ON_DEMAND

        if not self._committed_to_on_demand:
            # Time needed to safely finish using On-Demand from now:
            #   remaining_work (no interruptions) + one restart_overhead.
            overhead = self.restart_overhead
            safety_margin = self._gap_seconds  # account for discretization of steps

            threshold = remaining_work + overhead + safety_margin

            # If we no longer have enough slack to keep waiting for Spot,
            # commit to On-Demand and stick with it thereafter.
            if time_left <= threshold:
                self._committed_to_on_demand = True
                return ClusterType.ON_DEMAND

            # Before commitment: use Spot when available, otherwise wait (NONE).
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

        # After commitment: always run on On-Demand until completion.
        return ClusterType.ON_DEMAND