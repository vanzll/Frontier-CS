import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with safe on-demand fallback."""

    NAME = "cbm_multi_safe_v1"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize the solution from spec_path config."""
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Runtime state (initialized on first _step call when env is ready)
        self._runtime_initialized = False
        self._force_on_demand = False
        self._cached_done = 0.0
        self._last_done_len = 0
        return self

    def _initialize_runtime_state(self):
        """Lazy initialization that requires self.env to exist."""
        if self._runtime_initialized:
            return
        num_regions = self.env.get_num_regions()
        self._num_regions = num_regions
        self._region_up = [0] * num_regions
        self._region_total = [0] * num_regions
        self._region_down_streak = [0] * num_regions

        gap = float(getattr(self.env, "gap_seconds", 3600.0))
        # Require about one hour of continuous downtime before switching regions
        self._switch_downtime_steps = max(1, int(3600.0 / gap))

        self._force_on_demand = False
        self._cached_done = 0.0
        self._last_done_len = 0
        self._runtime_initialized = True

    def _update_done_cache(self):
        """Efficiently maintain total completed work."""
        td = self.task_done_time
        cur_len = len(td)
        if cur_len > self._last_done_len:
            # Add only new segments
            self._cached_done += sum(td[self._last_done_len : cur_len])
            self._last_done_len = cur_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Decide next cluster type."""
        self._initialize_runtime_state()

        # Update aggregate work done
        self._update_done_cache()
        done = self._cached_done
        remaining_work = max(0.0, self.task_duration - done)

        # If task is finished, run nothing
        if remaining_work <= 0.0:
            return ClusterType.NONE

        # Update region statistics with current observation
        cur_region = self.env.get_current_region()
        self._region_total[cur_region] += 1
        if has_spot:
            self._region_up[cur_region] += 1
            self._region_down_streak[cur_region] = 0
        else:
            self._region_down_streak[cur_region] += 1

        # Time remaining to deadline
        time_remaining = self.deadline - self.env.elapsed_seconds

        # Safety margin to account for discretization and one more overhead
        gap = float(self.env.gap_seconds)
        safety_margin = gap + self.restart_overhead  # ensures we commit early enough

        # If we are close enough to the deadline that we can *only* safely finish
        # using on-demand from now on (even if spot gives no additional progress),
        # then lock into on-demand.
        if (
            not self._force_on_demand
            and time_remaining
            <= remaining_work + self.restart_overhead + safety_margin
        ):
            self._force_on_demand = True

        # Once we commit to on-demand, stay on it until the task completes.
        if self._force_on_demand or last_cluster_type == ClusterType.ON_DEMAND:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        # Before fallback window: prefer spot when available.
        if has_spot:
            return ClusterType.SPOT

        # Spot unavailable in current region and still outside fallback window.
        # Optionally switch to a historically better region to improve future spot odds.
        if self._num_regions > 1:
            cur_est = (self._region_up[cur_region] + 1.0) / (
                self._region_total[cur_region] + 2.0
            )
            best_idx = cur_region
            best_est = cur_est
            for i in range(self._num_regions):
                if i == cur_region:
                    continue
                est = (self._region_up[i] + 1.0) / (self._region_total[i] + 2.0)
                if est > best_est:
                    best_est = est
                    best_idx = i

            # Switch regions only if:
            # - We've observed a continuous downtime streak here; and
            # - Another region's estimated availability is noticeably better.
            if (
                best_idx != cur_region
                and self._region_down_streak[cur_region]
                >= self._switch_downtime_steps
                and best_est - cur_est > 0.05
            ):
                self.env.switch_region(best_idx)

        # Wait idly this step; we still have slack and a safe fallback.
        return ClusterType.NONE