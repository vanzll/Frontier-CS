import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that balances the use of cheap Spot instances
    with the reliability of On-Demand instances to minimize cost while ensuring
    task completion before the deadline.
    """

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

        # Custom state for the strategy
        # Counter for consecutive steps without finding an available spot instance.
        self.no_spot_counter = 0
        # The number of regions to search before temporarily giving up and
        # using an On-Demand instance to make progress.
        self.patience = 2

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Calculate current state variables
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # 2. If the task is completed, do nothing to save costs.
        if work_remaining <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        # 3. Urgency Check: Determine if we are running out of time.
        # This is the "point of no return" where we must switch to On-Demand.

        # Calculate the total time required to finish the remaining work using
        # On-Demand, including a potential restart overhead.
        needs_restart_for_od = (last_cluster_type != ClusterType.ON_DEMAND)
        on_demand_finish_duration = work_remaining
        if needs_restart_for_od:
            on_demand_finish_duration += self.restart_overhead

        # Set the urgency threshold with a safety buffer of one time step.
        urgency_threshold = on_demand_finish_duration + self.env.gap_seconds

        if time_to_deadline <= urgency_threshold:
            # Urgent mode: Time is critical. Use On-Demand to guarantee completion.
            return ClusterType.ON_DEMAND

        # 4. Standard Mode: There is enough slack time to try using Spot.
        if has_spot:
            # Spot is available in the current region; use it.
            self.no_spot_counter = 0
            return ClusterType.SPOT
        else:
            # Spot is not available in the current region.
            self.no_spot_counter += 1

            if self.no_spot_counter <= self.patience:
                # Still have "patience" to search for Spot elsewhere.
                # Switch to the next region and return NONE to probe availability.
                current_region = self.env.get_current_region()
                num_regions = self.env.get_num_regions()
                if num_regions > 1:
                    next_region = (current_region + 1) % num_regions
                    self.env.switch_region(next_region)
                return ClusterType.NONE
            else:
                # Patience has run out. Use On-Demand for one step to make
                # guaranteed progress, then reset the counter to search again.
                self.no_spot_counter = 0
                return ClusterType.ON_DEMAND