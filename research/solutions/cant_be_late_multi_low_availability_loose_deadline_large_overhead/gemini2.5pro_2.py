import json
import math
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy using the Upper Confidence Bound (UCB)
    algorithm to balance exploration of new regions and exploitation of known
    good regions.
    """

    NAME = "ucb_scheduler"

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

        num_regions = self.env.get_num_regions()
        
        # Hyperparameters
        self.ucb_c = math.sqrt(2)
        self.ema_alpha = 0.1
        self.wait_slack_factor = 2.0
        self.switch_margin = 0.05
        
        # Stats Tracking
        self.region_stats = [
            {'visits': 0, 'spot_prob_ema': 0.5} 
            for _ in range(num_regions)
        ]
        self.total_visits = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        current_region = self.env.get_current_region()
        stats = self.region_stats[current_region]
        
        stats['visits'] += 1
        self.total_visits += 1
        
        current_observation = 1.0 if has_spot else 0.0
        stats['spot_prob_ema'] = (1 - self.ema_alpha) * stats['spot_prob_ema'] + self.ema_alpha * current_observation

        remaining_work = self.task_duration - sum(self.task_done_time)
        
        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds

        on_demand_guarantee_time = remaining_work + self.restart_overhead
        if time_to_deadline <= on_demand_guarantee_time:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            best_score = -1.0
            best_region_idx = -1
            log_total_visits = math.log(self.total_visits)
            
            scores = []
            for r_idx in range(self.env.get_num_regions()):
                r_stats = self.region_stats[r_idx]
                if r_stats['visits'] == 0:
                    score = float('inf')
                else:
                    exploitation_term = r_stats['spot_prob_ema']
                    exploration_term = self.ucb_c * math.sqrt(log_total_visits / r_stats['visits'])
                    score = exploitation_term + exploration_term
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_region_idx = r_idx

            current_score = scores[current_region]
            
            if best_region_idx != current_region and best_score > current_score + self.switch_margin:
                self.env.switch_region(best_region_idx)
                return ClusterType.SPOT
            else:
                slack = time_to_deadline - on_demand_guarantee_time
                wait_threshold = self.wait_slack_factor * self.restart_overhead
                
                if slack > wait_threshold:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND