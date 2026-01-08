import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that combines a deadline-aware safety net
    with a cost-minimization heuristic.

    The strategy operates based on the following principles:
    1.  **Deadline-Driven "Panic Mode"**: The highest priority is to finish the
        job before the deadline. If the time remaining is less than the time
        required to finish on a reliable on-demand instance (including a buffer
        for one potential restart), the strategy will exclusively use
        on-demand instances to guarantee completion.

    2.  **Opportunistic Spot Usage**: If a spot instance is available in the
        current region, it is always chosen. This is the most cost-effective
        way to make progress.

    3.  **Cost-Benefit Analysis (when Spot is unavailable)**: If spot instances
        are not available, the strategy decides between using a costly
        on-demand instance, waiting for a spot instance, or switching to another
        region.
        - It maintains an estimate of spot availability for each region based on
          historical data.
        - It calculates a "break-even" probability threshold from the relative
          prices of spot and on-demand instances.
        - If the estimated availability in the best-known region is below this
          threshold, the expected cost of waiting for a spot instance is higher
          than using on-demand. In this case, it chooses on-demand.
        - Otherwise, it switches to (or remains in) the region with the highest
          estimated spot availability and waits, choosing `ClusterType.NONE` to
          minimize cost while anticipating future spot availability.
    """

    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from the problem specification file.
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
        self.region_stats = []
        self.num_regions = 0

        # Prices from the problem description
        self.OD_PRICE = 3.06
        self.SPOT_PRICE = 0.9701

        # Pre-calculate the probability threshold for cost-based decisions.
        # If expected spot availability is lower than this, on-demand is cheaper.
        if self.OD_PRICE > 0:
            self.MIN_SPOT_PROB_FOR_WAITING = self.SPOT_PRICE / self.OD_PRICE
        else:
            self.MIN_SPOT_PROB_FOR_WAITING = 1.0  # Should not happen

        return self

    def _initialize(self) -> None:
        """
        Lazy initialization on the first step, once the environment is available.
        """
        self.num_regions = self.env.get_num_regions()
        self.region_stats = [{'total_steps': 0, 'spot_up_steps': 0} for _ in range(self.num_regions)]
        self.initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide the next action based on the current state of the environment.
        """
        if not self.initialized:
            self._initialize()

        # 1. Update historical statistics for the current region
        current_region = self.env.get_current_region()
        self.region_stats[current_region]['total_steps'] += 1
        if has_spot:
            self.region_stats[current_region]['spot_up_steps'] += 1

        # 2. Calculate remaining work and time
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done

        if work_left <= 0:
            return ClusterType.NONE  # Job is finished

        time_left = self.deadline - self.env.elapsed_seconds

        # 3. Panic Mode: If deadline is imminent, use On-Demand to guarantee completion
        if time_left <= work_left + self.restart_overhead:
            return ClusterType.ON_DEMAND

        # 4. Opportunistic Mode: If spot is available, always use it
        if has_spot:
            return ClusterType.SPOT

        # 5. Cost-Based Decision: If spot is unavailable, choose the most cost-effective action
        # Estimate spot availability probabilities for all regions using Laplace smoothing
        probs = []
        for i in range(self.num_regions):
            stats = self.region_stats[i]
            p = (stats['spot_up_steps'] + 1.0) / (stats['total_steps'] + 2.0)
            probs.append(p)

        # Find the best region to attempt to use spot
        best_region_idx = 0
        max_prob = -1.0
        for i, p in enumerate(probs):
            if p > max_prob:
                max_prob = p
                best_region_idx = i

        # If the best region's spot probability is too low, use On-Demand
        if max_prob < self.MIN_SPOT_PROB_FOR_WAITING:
            return ClusterType.ON_DEMAND
        else:
            # Otherwise, switch to the best region (if not already there) and wait
            if current_region != best_region_idx:
                self.env.switch_region(best_region_idx)

            return ClusterType.NONE