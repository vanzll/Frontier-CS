import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Cost-optimized multi-region scheduling strategy."""

    NAME = "cost_optimized_strategy"

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
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        
        Strategy:
        1. Calculate time required to finish using safe On-Demand instances.
        2. If time remaining is close to this required time (Panic Mode), force On-Demand.
        3. If there is slack (Opportunistic Mode):
           - If current region has Spot, use Spot.
           - If current region lacks Spot, switch to next region and wait (NONE) to check availability next step.
        """
        # 1. Gather State
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        done = sum(self.task_done_time)
        remaining_work = self.task_duration - done
        
        # Handle finished task case gracefully
        if remaining_work <= 1e-6:
            return ClusterType.NONE

        time_left = self.deadline - elapsed
        
        # 2. Calculate Time Needed for Safe On-Demand Execution
        # If we are already running OD, we pay remaining overhead.
        # If we switch to OD from Spot/None, we pay full restart overhead.
        
        full_overhead = self.restart_overhead
        curr_overhead = self.remaining_restart_overhead
        
        if last_cluster_type == ClusterType.ON_DEMAND:
            od_startup_cost = curr_overhead
        else:
            od_startup_cost = full_overhead
            
        time_needed_for_od = remaining_work + od_startup_cost
        
        # 3. Panic Check (Deadline Safety)
        # Calculate threshold: time needed + buffer of one time step (gap).
        # We multiply gap by 1.1 to provide a small safety margin against floating point issues.
        # If time_left < time_needed + gap, it means if we waste this step (by waiting or failing Spot),
        # we might not have enough time left to finish even with OD.
        
        safety_threshold = time_needed_for_od + gap * 1.1
        
        if time_left < safety_threshold:
            # Panic Mode: Not enough slack to risk waiting or preemption.
            return ClusterType.ON_DEMAND

        # 4. Opportunistic Optimization (Cost Saving)
        if has_spot:
            # Spot is available and we have slack. Use it to save money.
            return ClusterType.SPOT
        else:
            # Spot unavailable in current region.
            # Switch to next region to search for Spot availability.
            num_regions = self.env.get_num_regions()
            if num_regions > 1:
                current_region = self.env.get_current_region()
                next_region = (current_region + 1) % num_regions
                self.env.switch_region(next_region)
            
            # Return NONE for this step. 
            # We cannot return SPOT immediately after switching because we don't know 
            # if the new region has capacity until the next step.
            # Waiting costs time (burning slack) but 0 money.
            return ClusterType.NONE