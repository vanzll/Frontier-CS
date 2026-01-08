import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that prioritizes finishing before the deadline
    while minimizing cost. It employs a tiered decision-making process:
    1.  **Deadline Adherence:** A "panic mode" ensures the task will finish by
        switching to reliable On-Demand instances if the remaining time is critical.
    2.  **Cost Minimization:** It aggressively seeks and utilizes cheaper Spot
        instances, scanning all available regions and switching if necessary.
    3.  **Intelligent Waiting:** When no Spot instances are available, it uses a
        safety buffer to decide whether to use expensive On-Demand to make
        progress or to wait for Spot availability, thus saving costs without
        unduly risking the deadline.
    """

    NAME = "my_strategy"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy by loading the problem configuration and
        pre-loading spot availability traces for all regions.
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

        self.spot_availability = []
        if "trace_files" in config:
            for trace_file in config["trace_files"]:
                try:
                    with open(trace_file) as f:
                        content = f.read()
                        trace = [char == '1' for char in content if char in '01']
                        self.spot_availability.append(trace)
                except FileNotFoundError:
                    # Handle missing trace files gracefully
                    self.spot_availability.append([])
        
        # A safety buffer in hours to decide when to switch to On-Demand
        # even if not in absolute "panic mode". This is a key tunable parameter.
        self.safety_buffer_hours = 1.5

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the next cluster type to use for the current time step.
        """
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds

        # 1. PANIC MODE: Absolute priority to meet the deadline.
        # Calculates if we must use On-Demand now to finish in time.
        potential_overhead = 0
        if last_cluster_type != ClusterType.ON_DEMAND:
            potential_overhead = self.restart_overhead
        
        completion_time_if_od_now = current_time + potential_overhead + work_remaining
        if completion_time_if_od_now >= self.deadline:
            return ClusterType.ON_DEMAND

        # 2. SPOT SEEKING: Use the cheapest option if available.
        # a) Check current region first.
        if has_spot:
            return ClusterType.SPOT

        # b) If no spot locally, scan other regions.
        current_timestep = int(current_time / self.env.gap_seconds)
        num_regions = self.env.get_num_regions()

        for r in range(num_regions):
            if r == self.env.get_current_region():
                continue
            
            # Check if trace data for region 'r' shows spot availability
            if r < len(self.spot_availability) and current_timestep < len(self.spot_availability[r]):
                if self.spot_availability[r][current_timestep]:
                    self.env.switch_region(r)
                    return ClusterType.SPOT

        # 3. ON-DEMAND vs. NONE: Trade-off between cost and progress.
        # No spot available anywhere. Decide based on a safety buffer.
        safety_buffer_seconds = self.safety_buffer_hours * 3600
        
        # Project the finish time if we wait one more step and are then forced
        # to use On-Demand for the rest of the task.
        time_after_wait = current_time + self.env.gap_seconds
        od_start_overhead = self.restart_overhead
        
        projected_finish_if_we_wait = time_after_wait + od_start_overhead + work_remaining
        
        # If this projected finish time eats into our safety buffer,
        # we must use On-Demand now to preserve our slack time.
        if projected_finish_if_we_wait + safety_buffer_seconds >= self.deadline:
            return ClusterType.ON_DEMAND
        else:
            # We have enough slack; wait for a spot instance to save costs.
            return ClusterType.NONE