import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "efficient_adaptive"
    
    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)
        
        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Initialize strategy parameters
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        
        # State tracking
        self.current_region = 0
        self.region_history = []
        self.spot_streak = 0
        self.switch_cooldown = 0
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Get current state
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = deadline - elapsed
        
        # Emergency mode: must finish
        time_per_work = gap  # Normal case
        if last_cluster_type != ClusterType.NONE:
            effective_gap = min(gap, gap - self.remaining_restart_overhead)
            time_per_work = effective_gap if effective_gap > 0 else gap
        
        # Calculate minimum time needed with on-demand
        min_time_needed = work_remaining * (time_per_work / gap)
        if last_cluster_type != ClusterType.ON_DEMAND:
            min_time_needed += self.restart_overhead
        
        # Emergency: switch to on-demand if behind schedule
        if time_remaining < min_time_needed * 1.1:  # 10% safety margin
            if last_cluster_type != ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND if work_remaining > 0 else ClusterType.NONE
        
        # Calculate risk threshold based on progress and time
        progress_ratio = work_done / self.task_duration
        time_ratio = elapsed / deadline
        
        # Dynamic risk tolerance: more conservative as deadline approaches
        base_risk = 0.7
        if time_ratio > 0.5:
            risk_tolerance = base_risk * (1 - (time_ratio - 0.5) * 0.8)
        else:
            risk_tolerance = base_risk
        
        # Adjust risk based on spot availability streak
        if has_spot:
            self.spot_streak += 1
            streak_bonus = min(0.2, self.spot_streak * 0.02)
            risk_tolerance = min(0.9, risk_tolerance + streak_bonus)
        else:
            self.spot_streak = 0
        
        # Calculate value of switching regions
        if self.switch_cooldown > 0:
            self.switch_cooldown -= 1
        
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        # Consider region switch if spot is unavailable and not in cooldown
        if not has_spot and self.switch_cooldown == 0 and num_regions > 1:
            # Simple round-robin switching
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            self.switch_cooldown = 3  # Cooldown period
            return ClusterType.NONE  # Pause for one step after switch
        
        # Decision logic
        if has_spot:
            # Use spot if risk allows
            if work_remaining > 0:
                # Calculate expected value
                expected_work = gap
                if last_cluster_type != ClusterType.SPOT:
                    expected_work = max(0, gap - self.restart_overhead)
                
                # Only use spot if we can afford potential restart
                safe_time_margin = time_remaining - work_remaining * (gap / expected_work if expected_work > 0 else 1)
                if safe_time_margin > self.restart_overhead * risk_tolerance:
                    return ClusterType.SPOT
        
        # Fallback to on-demand if spot unavailable or too risky
        if work_remaining > 0:
            return ClusterType.ON_DEMAND
        
        return ClusterType.NONE