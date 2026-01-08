import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "DeadlineAwareSpotSeeker"

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

        self.avg_availability = []
        for trace_file in config.get("trace_files", []):
            try:
                with open(trace_file) as f:
                    lines = f.readlines()
                    trace_data = [int(line.strip()) for line in lines if line.strip()]
            except (IOError, ValueError):
                trace_data = []
            
            if trace_data:
                self.avg_availability.append(sum(trace_data) / len(trace_data))
            else:
                self.avg_availability.append(0.0)

        if self.avg_availability:
            self.best_region_idx = max(range(len(self.avg_availability)), key=self.avg_availability.__getitem__)
        else:
            self.best_region_idx = 0

        MIN_SPOT_AVAILABILITY_THRESHOLD = 0.20
        self.spot_is_generally_viable = True
        if not self.avg_availability or self.avg_availability[self.best_region_idx] < MIN_SPOT_AVAILABILITY_THRESHOLD:
            self.spot_is_generally_viable = False
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        work_done = sum(self.task_done_time)
        if work_done >= self.task_duration:
            return ClusterType.NONE

        work_remaining = self.task_duration - work_done
        time_left = self.deadline - self.env.elapsed_seconds

        safety_buffer = self.restart_overhead + self.env.gap_seconds
        if time_left <= work_remaining + safety_buffer:
            return ClusterType.ON_DEMAND

        if not self.spot_is_generally_viable:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                current_region = self.env.get_current_region()
                if self.env.get_num_regions() > 1 and current_region != self.best_region_idx:
                    self.env.switch_region(self.best_region_idx)
                
                return ClusterType.NONE