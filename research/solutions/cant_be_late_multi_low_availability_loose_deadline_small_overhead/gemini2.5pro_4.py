import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "slack_aware_switching"

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

        self.avg_spot_availability = []
        for trace_file in config.get('trace_files', []):
            with open(trace_file, 'r') as tf:
                lines = tf.readlines()
                availability_data = [int(line.strip()) for line in lines if line.strip()]
                if not availability_data:
                    avg_avail = 0.0
                else:
                    avg_avail = sum(availability_data) / len(availability_data)
                self.avg_spot_availability.append(avg_avail)

        self.initialized_region = False
        self.time_step_val = None

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.initialized_region:
            self.time_step_val = self.env.gap_seconds
            if self.avg_spot_availability:
                num_regions = self.env.get_num_regions()
                if num_regions > 0 and len(self.avg_spot_availability) == num_regions:
                    best_overall_region = max(range(num_regions), key=lambda i: self.avg_spot_availability[i])
                    if self.env.get_current_region() != best_overall_region:
                        self.env.switch_region(best_overall_region)
            self.initialized_region = True

        progress = sum(self.task_done_time)
        remaining_work = self.task_duration - progress

        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        steps_needed_od = math.ceil(remaining_work / self.time_step_val)
        time_needed_od_guaranteed = steps_needed_od * self.time_step_val + self.restart_overhead

        if time_to_deadline <= time_needed_od_guaranteed:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        if num_regions > 1 and self.avg_spot_availability and len(self.avg_spot_availability) == num_regions:
            alt_regions = (i for i in range(num_regions) if i != current_region)
            best_alt_region_idx = max(alt_regions, key=lambda i: self.avg_spot_availability[i], default=-1)

            switch_margin = 0.05
            if best_alt_region_idx != -1 and self.avg_spot_availability[best_alt_region_idx] > self.avg_spot_availability[current_region] + switch_margin:
                self.env.switch_region(best_alt_region_idx)
                return ClusterType.SPOT
        
        slack = time_to_deadline - time_needed_od_guaranteed
        
        wait_threshold = 2 * self.time_step_val
        if slack > wait_threshold:
            return ClusterType.NONE
        else:
            return ClusterType.ON_DEMAND