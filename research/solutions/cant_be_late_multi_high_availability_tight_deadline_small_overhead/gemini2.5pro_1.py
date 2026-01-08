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

        self.spot_traces = []
        if "trace_files" in config and config["trace_files"]:
            for trace_file in config['trace_files']:
                with open(trace_file) as f:
                    trace = [int(line.strip()) for line in f if line.strip()]
                    self.spot_traces.append(trace)

        self.num_regions = len(self.spot_traces)
        if self.num_regions > 0:
            self.trace_len = len(self.spot_traces[0])
            self.future_spot_runs = [[0] * self.trace_len for _ in range(self.num_regions)]
            for r in range(self.num_regions):
                count = 0
                for t in range(self.trace_len - 1, -1, -1):
                    if self.spot_traces[r][t] == 1:
                        count += 1
                    else:
                        count = 0
                    self.future_spot_runs[r][t] = count
        else:
            self.trace_len = 0
        
        self.PREEMPTION_BUFFER_COUNT = 3

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
        # If no trace data is available, use a simple fallback strategy
        if self.num_regions == 0:
            remaining_work_no_trace = self.task_duration - sum(self.task_done_time)
            if remaining_work_no_trace <= 0:
                return ClusterType.NONE
            time_needed_od_no_trace = remaining_work_no_trace + self.remaining_restart_overhead
            if self.deadline - self.env.elapsed_seconds <= time_needed_od_no_trace:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        # 1. Calculate current state
        elapsed_seconds = self.env.elapsed_seconds
        current_timestep = int(elapsed_seconds / self.env.gap_seconds)
        
        # Handle cases where simulation runs longer than trace data
        if current_timestep >= self.trace_len:
            current_timestep = self.trace_len - 1

        remaining_work = self.task_duration - sum(self.task_done_time)

        # If task is completed, do nothing to save cost.
        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - elapsed_seconds
        safety_buffer = self.PREEMPTION_BUFFER_COUNT * self.restart_overhead

        # 2. Urgency Check: If deadline is approaching, switch to On-Demand to guarantee completion.
        time_needed_on_demand = remaining_work + self.remaining_restart_overhead
        if time_to_deadline <= time_needed_on_demand + safety_buffer:
            return ClusterType.ON_DEMAND

        # 3. Optimization: If we have enough slack, prioritize cheap Spot instances.
        if has_spot:
            return ClusterType.SPOT

        # 4. No Spot in current region: Check if switching to another region is feasible and beneficial.
        # A switch is only safe if we have enough time to absorb the restart overhead.
        time_needed_after_switch = remaining_work + self.restart_overhead
        if time_to_deadline <= time_needed_after_switch + safety_buffer:
            # Not enough time to safely switch. Use On-Demand in the current region.
            return ClusterType.ON_DEMAND

        # Find the best region to switch to, based on future spot availability.
        best_region_to_switch = -1
        max_future_run = -1
        current_region = self.env.get_current_region()

        search_order = [(current_region + i + 1) % self.num_regions for i in range(self.num_regions - 1)]
        for region_idx in search_order:
            if self.spot_traces[region_idx][current_timestep] == 1:
                future_run = self.future_spot_runs[region_idx][current_timestep]
                if future_run > max_future_run:
                    max_future_run = future_run
                    best_region_to_switch = region_idx

        if best_region_to_switch != -1:
            # Found a better region. Switch and use Spot.
            self.env.switch_region(best_region_to_switch)
            return ClusterType.SPOT
        else:
            # No other region has Spot right now. Fall back to On-Demand.
            return ClusterType.ON_DEMAND