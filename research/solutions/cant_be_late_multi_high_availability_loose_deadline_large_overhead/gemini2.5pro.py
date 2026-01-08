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

        self.num_regions = self.env.get_num_regions()
        
        # Hyperparameters for the strategy
        self.ema_alpha = 0.3
        self.switch_margin = 0.4
        self.consecutive_no_spot_threshold = 2
        
        # State tracking for each region
        # Optimistic start based on problem hint of high spot availability
        self.region_ema = [0.8] * self.num_regions 
        self.consecutive_no_spot = [0] * self.num_regions

        # Safety buffer: if remaining slack drops below this, force On-Demand.
        # Set to 1 hour, or 5x restart overhead, whichever is larger.
        self.safety_buffer = max(3600.0, self.restart_overhead * 5)
        
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
        current_region = self.env.get_current_region()
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # Update EMA of spot availability for the current region
        current_ema = self.region_ema[current_region]
        self.region_ema[current_region] = \
            self.ema_alpha * int(has_spot) + (1 - self.ema_alpha) * current_ema
        
        # Update consecutive no-spot counter
        if has_spot:
            self.consecutive_no_spot[current_region] = 0
        else:
            self.consecutive_no_spot[current_region] += 1

        # Region Switching Logic (affects the next timestep)
        if not has_spot:
            best_alt_region = -1
            best_alt_score = -1.0
            for r in range(self.num_regions):
                if r == current_region:
                    continue
                if self.region_ema[r] > best_alt_score:
                    best_alt_score = self.region_ema[r]
                    best_alt_region = r

            # Switch if a significantly better region exists and current one has been out of spot for a while
            if (best_alt_region != -1 and
                    best_alt_score > self.region_ema[current_region] + self.switch_margin and
                    self.consecutive_no_spot[current_region] >= self.consecutive_no_spot_threshold):
                
                self.env.switch_region(best_alt_region)
                self.consecutive_no_spot[current_region] = 0

        # Instance Type Choice Logic (for the current timestep)

        # 1. Panic Mode: If deadline is too close, must use On-Demand to guarantee progress.
        if time_to_deadline <= work_remaining + self.safety_buffer:
            return ClusterType.ON_DEMAND

        # 2. Normal Mode:
        if has_spot:
            # Spot is available and cheap; use it greedily as we are not in panic mode.
            return ClusterType.SPOT
        else:
            # No spot available. Decide between On-Demand (make progress) and None (wait).
            # We can afford to wait if doing so won't push us into panic mode in the next step.
            if (time_to_deadline - self.env.gap_seconds) > (work_remaining + self.safety_buffer):
                # Safe to wait for spot to return.
                return ClusterType.NONE
            else:
                # Not safe to wait, must make progress with On-Demand.
                return ClusterType.ON_DEMAND