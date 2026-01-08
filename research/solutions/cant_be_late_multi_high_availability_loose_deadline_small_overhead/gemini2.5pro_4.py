import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"

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

        self.initialized = False
        self.region_probes = []
        self.spot_successes = []
        self.total_probes = 0
        self.num_regions = 0
        
        return self

    def _initialize(self):
        """
        Initialize attributes that depend on the environment.
        """
        self.num_regions = self.env.get_num_regions()
        self.region_probes = [0] * self.num_regions
        self.spot_successes = [0] * self.num_regions
        self.initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.initialized:
            self._initialize()

        current_region = self.env.get_current_region()

        # Update statistics for the current region based on spot availability
        self.total_probes += 1
        self.region_probes[current_region] += 1
        if has_spot:
            self.spot_successes[current_region] += 1

        # Calculate current progress and time remaining
        progress = sum(self.task_done_time)
        remaining_work = self.task_duration - progress

        if remaining_work <= 0:
            return ClusterType.NONE

        remaining_time = self.deadline - self.env.elapsed_seconds
        slack = remaining_time - remaining_work

        # Safety Net: If slack is critically low, use On-Demand to guarantee completion.
        if slack <= self.restart_overhead + self.env.gap_seconds:
            return ClusterType.ON_DEMAND

        if has_spot:
            # If Spot is available, use it.
            return ClusterType.SPOT
        else:
            # Spot is not available. Search for a better region.
            if self.num_regions > 1:
                best_next_region = -1
                max_ucb_score = -1.0
                
                log_total = math.log(self.total_probes) if self.total_probes > 0 else 0

                candidate_regions = [i for i in range(self.num_regions) if i != current_region]
                
                # Prioritize exploring unvisited regions.
                unvisited_candidates = [r for r in candidate_regions if self.region_probes[r] == 0]
                if unvisited_candidates:
                    best_next_region = unvisited_candidates[0]
                else:
                    # If all other regions are visited, use UCB1 to select the best one.
                    for i in candidate_regions:
                        avg_reward = self.spot_successes[i] / self.region_probes[i]
                        exploration_term = math.sqrt(2 * log_total / self.region_probes[i])
                        ucb_score = avg_reward + exploration_term
                        
                        if ucb_score > max_ucb_score:
                            max_ucb_score = ucb_score
                            best_next_region = i
                
                if best_next_region != -1:
                    self.env.switch_region(best_next_region)

            # For the current step, wait to save costs.
            return ClusterType.NONE