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

        # Internal state for strategy
        self._initialized = False
        self._force_on_demand = False

        # Progress tracking
        self._work_done = 0.0
        self._last_task_len = 0

        # Region statistics
        self._num_regions = None
        self._region_time = None
        self._region_spot_time = None

        # Exploration/exploitation of regions
        self._explore_phase = True
        self._per_region_sample = None
        self._explore_region_idx = None
        self._explore_region_start_time = None
        self._home_region = None

        return self

    # Helper to lazily initialize state that depends on env
    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True

        self._num_regions = self.env.get_num_regions()
        if self._num_regions is None or self._num_regions <= 0:
            self._num_regions = 1

        self._region_time = [0.0 for _ in range(self._num_regions)]
        self._region_spot_time = [0.0 for _ in range(self._num_regions)]

        # Initialize work tracking from any existing segments
        self._last_task_len = len(self.task_done_time)
        if self._last_task_len > 0:
            self._work_done = float(sum(self.task_done_time))
        else:
            self._work_done = 0.0

        # Region exploration setup
        current_region = self.env.get_current_region()
        if current_region is None or current_region < 0 or current_region >= self._num_regions:
            current_region = 0

        self._home_region = current_region
        self._explore_region_idx = current_region
        self._explore_region_start_time = self.env.elapsed_seconds

        # Total exploration budget: up to 2 hours or 20% of deadline, but at least one step per region
        gap = getattr(self.env, "gap_seconds", 1.0)
        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 36.0 * 3600.0

        total_explore_time = min(2.0 * 3600.0, 0.2 * deadline)
        min_total = gap * self._num_regions
        if total_explore_time < min_total:
            total_explore_time = min_total

        self._per_region_sample = max(total_explore_time / self._num_regions, gap)
        # If only one region, exploration is trivial
        if self._num_regions <= 1:
            self._explore_phase = False

    def _update_progress(self):
        """Incrementally update total work done based on new segments."""
        curr_len = len(self.task_done_time)
        if curr_len > self._last_task_len:
            # Sum only new entries
            delta = sum(self.task_done_time[self._last_task_len : curr_len])
            self._work_done += float(delta)
            self._last_task_len = curr_len

    def _update_region_stats(self, has_spot: bool):
        """Update per-region spot availability statistics."""
        region_idx = self.env.get_current_region()
        if region_idx is None or region_idx < 0 or region_idx >= self._num_regions:
            return
        dt = getattr(self.env, "gap_seconds", 1.0)
        self._region_time[region_idx] += dt
        if has_spot:
            self._region_spot_time[region_idx] += dt

    def _maybe_commit_on_demand(self, remaining_work: float) -> bool:
        """
        Decide whether we must switch to ON_DEMAND to safely meet the deadline.
        Returns True if we should (or already have) committed to ON_DEMAND.
        """
        if self._force_on_demand:
            return True

        time_now = float(self.env.elapsed_seconds)
        remaining_time = float(self.deadline) - time_now
        if remaining_time <= 0.0:
            # Out of time; best effort is to run ON_DEMAND
            self._force_on_demand = True
            return True

        gap = getattr(self.env, "gap_seconds", 1.0)
        overhead = float(self.restart_overhead)

        # Worst-case time needed if we commit to ON_DEMAND now:
        # one full restart_overhead plus the remaining compute work.
        worst_need = overhead + remaining_work

        # Safety margin to account for discretization/rounding
        fudge = 2.0 * gap + overhead

        slack = remaining_time - worst_need

        if slack <= fudge:
            self._force_on_demand = True
            return True
        return False

    def _handle_region_exploration(self):
        """Optionally switch regions during spot phase to find a good home region."""
        if self._num_regions <= 1:
            self._explore_phase = False
            return

        time_now = float(self.env.elapsed_seconds)
        gap = getattr(self.env, "gap_seconds", 1.0)
        current_region = self.env.get_current_region()
        if current_region is None or current_region < 0 or current_region >= self._num_regions:
            current_region = 0

        if not self._explore_phase:
            # Exploitation phase: stick to home_region when not on-demand
            if current_region != self._home_region:
                self.env.switch_region(self._home_region)
            return

        # Ensure we are in the intended exploration region
        if current_region != self._explore_region_idx:
            self.env.switch_region(self._explore_region_idx)
            # Do not update _explore_region_start_time here; elapsed_seconds has not advanced yet.

        # Time spent in this exploration region including this coming step
        time_in_region = time_now - self._explore_region_start_time + gap
        if time_in_region + 1e-6 < self._per_region_sample:
            return

        # Move to next region in exploration cycle
        next_idx = (self._explore_region_idx + 1) % self._num_regions

        # After completing a full cycle, end exploration and choose best home region
        if next_idx == self._home_region:
            # Choose region with highest observed spot availability ratio
            best_region = self._home_region
            best_ratio = -1.0
            for r in range(self._num_regions):
                t = self._region_time[r]
                if t <= 0.0:
                    continue
                ratio = self._region_spot_time[r] / t
                if ratio > best_ratio + 1e-12:
                    best_ratio = ratio
                    best_region = r
            self._home_region = best_region
            self._explore_phase = False

            current_region = self.env.get_current_region()
            if current_region != self._home_region:
                self.env.switch_region(self._home_region)
            return

        # Continue exploration with next region
        self._explore_region_idx = next_idx
        self._explore_region_start_time = time_now + gap
        if current_region != self._explore_region_idx:
            self.env.switch_region(self._explore_region_idx)

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
        # Ensure environment-dependent state is initialized
        self._lazy_init()

        # Update total work done and region statistics
        self._update_progress()
        self._update_region_stats(has_spot)

        # Check if task already completed (defensive)
        remaining_work = float(self.task_duration) - self._work_done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Decide whether we must commit to ON_DEMAND to meet deadline
        if self._maybe_commit_on_demand(remaining_work):
            return ClusterType.ON_DEMAND

        # Still in spot/idle phase: manage regions (exploration/exploitation)
        self._handle_region_exploration()

        # Cluster choice within spot phase
        if has_spot:
            return ClusterType.SPOT

        # No spot right now and we still have slack: wait to save cost
        return ClusterType.NONE