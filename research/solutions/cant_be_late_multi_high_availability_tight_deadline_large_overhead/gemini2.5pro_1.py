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

        # Custom initialization for the strategy
        self.num_regions = self.env.get_num_regions()
        self.region_stats = [{'seen': 0, 'available': 0} for _ in range(self.num_regions)]
        self.consecutive_failures = [0] * self.num_regions

        # Heuristic parameters
        self.PANIC_BUFFER_FACTOR = 2.0
        self.SWITCH_THRESHOLD = 2
        self.BETA_PRIOR_SUCCESS = 1
        self.BETA_PRIOR_FAILURE = 1
        
        return self

    def _get_availability_estimate(self, region_idx: int) -> float:
        """
        Calculates the estimated spot availability for a region using Bayesian averaging
        with a Beta(1,1) prior to handle regions with little observation data.
        """
        stats = self.region_stats[region_idx]
        s = self.BETA_PRIOR_SUCCESS
        f = self.BETA_PRIOR_FAILURE
        return (stats['available'] + s) / (stats['seen'] + s + f)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Update historical stats for the current region
        current_region = self.env.get_current_region()
        self.region_stats[current_region]['seen'] += 1
        if has_spot:
            self.region_stats[current_region]['available'] += 1
            self.consecutive_failures[current_region] = 0
        else:
            self.consecutive_failures[current_region] += 1

        # 2. Calculate progress and time remaining
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done

        if remaining_work <= 0:
            return ClusterType.NONE

        time_elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - time_elapsed

        # 3. PANIC MODE: Must use On-Demand to guarantee completion before the deadline.
        total_time_needed = remaining_work + self.remaining_restart_overhead
        panic_buffer = self.PANIC_BUFFER_FACTOR * self.restart_overhead
        if time_remaining <= total_time_needed + panic_buffer:
            return ClusterType.ON_DEMAND

        # 4. PRIMARY MODE: Spot is available, so use it for cost savings.
        if has_spot:
            return ClusterType.SPOT

        # 5. FALLBACK MODE: Spot is not available in the current region.
        # 5a. Consider switching to a region with better historical spot availability.
        if self.consecutive_failures[current_region] >= self.SWITCH_THRESHOLD and self.num_regions > 1:
            current_avail = self._get_availability_estimate(current_region)
            
            best_alt_region = -1
            max_alt_avail = -1.0
            
            for i in range(self.num_regions):
                if i == current_region:
                    continue
                
                alt_avail = self._get_availability_estimate(i)
                if alt_avail > max_alt_avail:
                    max_alt_avail = alt_avail
                    best_alt_region = i

            if best_alt_region != -1 and max_alt_avail > current_avail:
                self.env.switch_region(best_alt_region)
                self.consecutive_failures[current_region] = 0  # Reset for the region we are leaving

        # 5b. Decide between On-Demand (to catch up) and None (to wait for spot).
        # This is based on whether we are ahead of or behind a linear progress schedule.
        target_work_done = (time_elapsed / self.deadline) * self.task_duration
        
        if work_done < target_work_done:
            # Behind schedule: make progress, even if expensive.
            return ClusterType.ON_DEMAND
        else:
            # Ahead of schedule: can afford to wait for spot.
            return ClusterType.NONE