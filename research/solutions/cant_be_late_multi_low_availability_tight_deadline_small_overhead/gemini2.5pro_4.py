import json
import collections
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

        self.initialized = False
        return self

    def _initialize(self):
        """
        Lazy initializer, called on the first _step.
        Initializes data structures and tunable parameters.
        """
        num_regions = self.env.get_num_regions()
        
        history_len = 100
        self.spot_history = {i: collections.deque(maxlen=history_len) for i in range(num_regions)}
        
        self.panic_buffer = self.restart_overhead * 1.05
        self.switch_slack_factor = 3.0
        self.switch_score_diff_threshold = 0.25
        self.switch_bad_region_threshold = 0.5
        self.od_slack_ratio_threshold = 0.25

        self.initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.initialized:
            self._initialize()

        current_region = self.env.get_current_region()
        self.spot_history[current_region].append(1 if has_spot else 0)

        progress = sum(self.task_done_time)
        remaining_work = self.task_duration - progress

        if remaining_work <= 0:
            return ClusterType.NONE

        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        # Panic Mode: If finishing on On-Demand is the only way to meet the deadline.
        time_needed_on_demand = remaining_work + self.panic_buffer
        if time_needed_on_demand >= remaining_time:
            return ClusterType.ON_DEMAND

        # Greedy choice: If spot is available, use it.
        if has_spot:
            return ClusterType.SPOT

        # No spot available: Decide between Switch, On-Demand, or None.
        slack = remaining_time - remaining_work

        # Evaluate region switching
        scores = {}
        for r_idx, history in self.spot_history.items():
            if len(history) > 0:
                scores[r_idx] = sum(history) / len(history)
            else:
                # Optimistic score for unexplored regions
                scores[r_idx] = 0.75

        current_score = scores[current_region]
        
        best_score_other = -1.0
        best_region_idx_other = -1
        for r_idx, score in scores.items():
            if r_idx != current_region and score > best_score_other:
                best_score_other = score
                best_region_idx_other = r_idx

        can_afford_switch = slack > (self.restart_overhead * self.switch_slack_factor)
        is_current_region_bad = current_score < self.switch_bad_region_threshold
        is_other_region_better = best_score_other > current_score + self.switch_score_diff_threshold

        if best_region_idx_other != -1 and can_afford_switch and is_current_region_bad and is_other_region_better:
            self.env.switch_region(best_region_idx_other)
            return ClusterType.ON_DEMAND

        # Evaluate On-Demand vs. None
        if (slack / remaining_time) < self.od_slack_ratio_threshold:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE