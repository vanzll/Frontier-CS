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

        self.is_initialized = False
        self.num_regions = 0
        self.probed_in_cycle = set()
        self.last_spot_info = {}

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
        if not self.is_initialized:
            self.num_regions = self.env.get_num_regions()
            self.is_initialized = True

        current_region = self.env.get_current_region()
        now = self.env.elapsed_seconds
        
        self.last_spot_info[current_region] = (has_spot, now)

        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done

        if work_left <= 0:
            return ClusterType.NONE

        time_left_to_deadline = self.deadline - now
        
        # This is the time required to finish the rest of the job using On-Demand,
        # including a one-time overhead for switching to the reliable instance.
        # This serves as a safety threshold.
        time_needed_for_od_finish = work_left + self.restart_overhead

        if time_left_to_deadline <= time_needed_for_od_finish:
            # "Panic mode": Time is running out. We must use the reliable
            # On-Demand instance to guarantee finishing before the deadline.
            self.probed_in_cycle.clear()
            return ClusterType.ON_DEMAND

        if has_spot:
            # Spot is available and cheap. This is the best case.
            self.probed_in_cycle.clear()
            return ClusterType.SPOT
        
        # Spot is not available in the current region.
        # We start or continue a probing cycle to find a region with Spot.
        self.probed_in_cycle.add(current_region)

        # Calculate if we can afford to spend time searching for Spot.
        slack = time_left_to_deadline - time_needed_for_od_finish
        
        # A probe consists of switching region and returning NONE. This costs
        # one time step of progress and incurs a restart overhead on the next compute step.
        probe_time_cost = self.env.gap_seconds + self.restart_overhead
        
        if slack <= probe_time_cost:
            # Not enough slack time to risk searching. Fallback to On-Demand.
            self.probed_in_cycle.clear()
            return ClusterType.ON_DEMAND

        # We have enough slack. Search for a promising region to switch to.
        # A promising region is one we haven't checked in this cycle and was
        # most recently seen with available Spot instances.
        best_candidate_region = -1
        most_recent_spot_time = -1
        has_unprobed_regions = False

        for r in range(self.num_regions):
            if r in self.probed_in_cycle:
                continue
            
            has_unprobed_regions = True
            
            if r in self.last_spot_info:
                r_has_spot, r_timestamp = self.last_spot_info[r]
                if r_has_spot and r_timestamp > most_recent_spot_time:
                    most_recent_spot_time = r_timestamp
                    best_candidate_region = r

        if best_candidate_region != -1:
            # Switch to the most promising candidate.
            self.env.switch_region(best_candidate_region)
            return ClusterType.NONE
        elif has_unprobed_regions:
            # No known promising regions, but some regions haven't been checked yet.
            # Probe the next available one in a round-robin fashion.
            next_region_to_probe = (current_region + 1) % self.num_regions
            while next_region_to_probe in self.probed_in_cycle:
                next_region_to_probe = (next_region_to_probe + 1) % self.num_regions
            
            self.env.switch_region(next_region_to_probe)
            return ClusterType.NONE
        else:
            # All regions have been probed in this cycle and none have Spot.
            # We must use On-Demand to make progress.
            self.probed_in_cycle.clear()
            return ClusterType.ON_DEMAND