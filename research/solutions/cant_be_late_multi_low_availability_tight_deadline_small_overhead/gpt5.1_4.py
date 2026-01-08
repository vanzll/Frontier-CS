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
        # Per-episode state (initialized lazily in _step when env is ready)
        self._episode_initialized = False
        return self

    def _initialize_episode_state(self):
        """Initialize state for a new environment episode."""
        # Initialize progress tracking using current task_done_time list.
        try:
            task_done = self.task_done_time
        except AttributeError:
            task_done = None

        if task_done:
            total_progress = 0.0
            for v in task_done:
                total_progress += float(v)
            last_len = len(task_done)
        else:
            total_progress = 0.0
            last_len = 0

        self._progress_done = total_progress
        self._last_task_len = last_len

        # Detect environment capabilities (regions, has_spot attribute, etc.).
        env = getattr(self, "env", None)
        self._supports_regions = False
        self._has_spot_attr = False
        self._num_regions = 0
        self._region_stats = []

        if env is not None:
            try:
                num = env.get_num_regions()
            except Exception:
                num = 0

            if num and num > 0:
                self._supports_regions = hasattr(env, "switch_region") and hasattr(
                    env, "get_current_region"
                )
                self._num_regions = num
                self._region_stats = [{"steps": 0, "spot_steps": 0} for _ in range(num)]
                self._has_spot_attr = hasattr(env, "has_spot")

        self._last_elapsed_seconds = getattr(self.env, "elapsed_seconds", 0.0)
        self._episode_initialized = True

    def _update_progress(self):
        """Incrementally update total completed work based on task_done_time list."""
        task_done = getattr(self, "task_done_time", None)
        if not task_done:
            return

        cur_len = len(task_done)
        if cur_len <= self._last_task_len:
            return

        total_new = 0.0
        for i in range(self._last_task_len, cur_len):
            total_new += float(task_done[i])

        self._progress_done += total_new
        self._last_task_len = cur_len

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
        elapsed = getattr(env, "elapsed_seconds", 0.0)

        # Initialize or reset per-episode state.
        if not getattr(self, "_episode_initialized", False) or elapsed < getattr(
            self, "_last_elapsed_seconds", -1.0
        ):
            self._initialize_episode_state()
        self._last_elapsed_seconds = elapsed

        # Update completed work progress.
        self._update_progress()

        # Remaining work in seconds.
        task_duration = self.task_duration
        work_remaining = task_duration - self._progress_done
        if work_remaining <= 0.0:
            # Task already complete; no need to run more.
            return ClusterType.NONE

        # Core timing parameters (in seconds).
        gap = env.gap_seconds
        restart_overhead = self.restart_overhead
        deadline = self.deadline
        t = elapsed

        # Safety check: can we afford to lose the entire next step with zero progress
        # and then still finish on On-Demand (including a full restart_overhead)?
        if t + gap + restart_overhead + work_remaining > deadline:
            # Cannot risk losing another full step: must use On-Demand now.
            return ClusterType.ON_DEMAND

        # We are in the "safe" zone: it's okay if the next step yields zero progress,
        # since we can still switch to On-Demand after that step and finish by deadline.

        # Multi-region spot scanning is only useful in this safe zone.
        has_spot_current = has_spot

        if self._supports_regions and self._has_spot_attr:
            num_regions = self._num_regions
            try:
                orig_region = env.get_current_region()
            except Exception:
                orig_region = 0

            has_spot_per_region = [False] * num_regions

            # Probe all regions' spot availability for this timestep.
            for r in range(num_regions):
                try:
                    env.switch_region(r)
                except Exception:
                    # If switching fails, leave env as-is.
                    pass
                cur_region_has_spot = bool(getattr(env, "has_spot", False))
                has_spot_per_region[r] = cur_region_has_spot

                # Update simple frequency stats for region reliability.
                stats = self._region_stats[r]
                stats["steps"] += 1
                if cur_region_has_spot:
                    stats["spot_steps"] += 1

            # Select target region: prefer any region with spot now, weighted by
            # historical spot availability; otherwise choose the most reliable region.
            target_region = orig_region
            any_spot_now = False
            best_score = -1.0

            # Prefer regions that currently have spot.
            for r in range(num_regions):
                if not has_spot_per_region[r]:
                    continue
                any_spot_now = True
                stats = self._region_stats[r]
                steps = stats["steps"]
                p = (stats["spot_steps"] / steps) if steps > 0 else 0.0
                score = p
                if r == orig_region:
                    score += 1e-6  # tiny bias to avoid switching when equal
                if score > best_score:
                    best_score = score
                    target_region = r

            if not any_spot_now:
                # No region has spot at this timestep: choose region with highest
                # long-term spot frequency to wait in.
                best_p = -1.0
                for r in range(num_regions):
                    stats = self._region_stats[r]
                    steps = stats["steps"]
                    p = (stats["spot_steps"] / steps) if steps > 0 else 0.0
                    if (
                        p > best_p + 1e-12
                        or (abs(p - best_p) <= 1e-12 and r == orig_region)
                    ):
                        best_p = p
                        target_region = r

            # Switch to chosen region for this step if needed.
            if target_region != orig_region:
                try:
                    env.switch_region(target_region)
                except Exception:
                    pass

            has_spot_current = has_spot_per_region[target_region]

        # Decision: in the safe zone, use Spot wherever available; otherwise wait.
        if has_spot_current:
            return ClusterType.SPOT

        # No spot currently (in best region); we can afford to wait for spot later.
        return ClusterType.NONE