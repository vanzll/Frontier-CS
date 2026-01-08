import json
from argparse import Namespace
import heapq
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
        
        # Parameters
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        
        # State tracking
        self.spot_history = {}  # region -> list of spot availability
        self.region_costs = {}  # region -> list of costs
        self.current_plan = []
        self.plan_index = 0
        self.last_decision = None
        self.consecutive_spot_failures = 0
        self.time_since_last_switch = 0
        self.region_spot_availability = {}  # region -> availability score
        
        return self

    def _calculate_remaining_time_budget(self):
        """Calculate remaining time budget accounting for overheads."""
        elapsed = self.env.elapsed_seconds
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # Conservative estimate: assume we'll need at least one more restart
        time_needed = remaining_work + self.restart_overhead
        time_left = self.deadline - elapsed
        
        return time_left - time_needed

    def _get_region_spot_score(self, region_idx):
        """Calculate spot availability score for a region."""
        if region_idx not in self.spot_history:
            return 0.5  # Default if no history
        
        history = self.spot_history[region_idx]
        if not history:
            return 0.5
        
        # Calculate recent availability (weight recent observations more)
        recent_len = min(10, len(history))
        recent = history[-recent_len:]
        return sum(recent) / len(recent)

    def _should_switch_region(self, current_region, has_spot):
        """Determine if we should switch regions."""
        time_left = self.deadline - self.env.elapsed_seconds
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # Don't switch if we're very close to deadline
        if time_left < remaining_work + 2 * self.restart_overhead:
            return False
        
        # Don't switch too frequently
        if self.time_since_last_switch < 5 * self.env.gap_seconds:
            return False
        
        # Switch if current region has poor spot availability
        if not has_spot and self.consecutive_spot_failures > 2:
            return True
        
        # Check other regions for better spot availability
        current_score = self._get_region_spot_score(current_region)
        num_regions = self.env.get_num_regions()
        
        for i in range(num_regions):
            if i != current_region:
                score = self._get_region_spot_score(i)
                if score > current_score + 0.2:  # Significant improvement
                    return True
        
        return False

    def _select_best_region(self, current_region):
        """Select the best region to switch to."""
        best_region = current_region
        best_score = self._get_region_spot_score(current_region)
        
        num_regions = self.env.get_num_regions()
        for i in range(num_regions):
            if i != current_region:
                score = self._get_region_spot_score(i)
                if score > best_score:
                    best_score = score
                    best_region = i
        
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        current_region = self.env.get_current_region()
        
        # Update spot history
        if current_region not in self.spot_history:
            self.spot_history[current_region] = []
        self.spot_history[current_region].append(1 if has_spot else 0)
        
        # Update consecutive spot failures
        if not has_spot and last_cluster_type == ClusterType.SPOT:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = 0
        
        # Update time since last switch
        self.time_since_last_switch += self.env.gap_seconds
        
        # Calculate time pressure
        time_left = self.deadline - self.env.elapsed_seconds
        remaining_work = self.task_duration - sum(self.task_done_time)
        time_pressure = remaining_work / max(time_left, 0.1)
        
        # Check if we should switch region
        if self._should_switch_region(current_region, has_spot):
            best_region = self._select_best_region(current_region)
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.time_since_last_switch = 0
                # When switching, start with on-demand for reliability
                return ClusterType.ON_DEMAND
        
        # Emergency mode: if we're running out of time
        if time_pressure > 0.9 or time_left < remaining_work + self.restart_overhead:
            return ClusterType.ON_DEMAND
        
        # If we have pending restart overhead, wait
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
        
        # Normal decision making
        if has_spot:
            # Use spot if we have good time buffer
            if time_pressure < 0.7:
                return ClusterType.SPOT
            else:
                # Mixed strategy based on time pressure
                spot_prob = 1.0 - time_pressure
                # Use spot with probability decreasing as time pressure increases
                if self.consecutive_spot_failures < 2 and spot_prob > 0.3:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
        else:
            # No spot available
            if time_pressure > 0.6:
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available
                return ClusterType.NONE