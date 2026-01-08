import json
from argparse import Namespace
from typing import List
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
        
        # Initialize strategy state
        self.region_count = 0
        self.best_region = 0
        self.spot_availability = []
        self.spot_history = []
        self.spot_score = []
        self.region_switches = 0
        self.last_region_switch_time = 0
        self.consecutive_spot_failures = 0
        self.consecutive_spot_successes = 0
        self.emergency_mode = False
        self.initialized = False
        self.time_step = 3600  # 1 hour in seconds
        
        # Cost parameters
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if not self.initialized:
            self._initialize_state()
        
        # Get current state
        current_region = self.env.get_current_region()
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        task_done = sum(self.task_done_time)
        task_remaining = self.task_duration - task_done
        
        # Check if we're in emergency mode (running out of time)
        time_remaining = deadline - elapsed
        time_needed_with_overhead = task_remaining + self.restart_overhead
        
        # Emergency mode: if we're running out of time, use on-demand
        if time_remaining <= time_needed_with_overhead * 1.5:
            self.emergency_mode = True
        
        # If in emergency mode, use on-demand to guarantee completion
        if self.emergency_mode:
            if last_cluster_type != ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        # Update spot availability history
        self._update_spot_history(current_region, has_spot)
        
        # Calculate urgency factor (0-1, higher = more urgent)
        urgency = 1.0 - (time_remaining / (deadline * 0.8))
        urgency = max(0.0, min(1.0, urgency))
        
        # Calculate region scores
        self._update_region_scores()
        
        # Check if we should switch regions
        should_switch = self._should_switch_region(current_region, urgency)
        
        if should_switch:
            target_region = self._select_best_region(current_region)
            if target_region != current_region:
                self.env.switch_region(target_region)
                self.region_switches += 1
                self.last_region_switch_time = elapsed
                self.consecutive_spot_failures = 0
                # After switching, start with on-demand for stability
                return ClusterType.ON_DEMAND
        
        # Decision logic
        # Rule 1: If spot is available and we've had recent success, use spot
        if has_spot:
            spot_success_rate = self._get_spot_success_rate(current_region)
            spot_confidence = min(1.0, self.consecutive_spot_successes / 5.0)
            
            # Use spot if we have confidence or if cost savings are significant
            if (spot_confidence > 0.6 or 
                (spot_success_rate > 0.7 and not self.emergency_mode)):
                
                # Check if we've had too many recent failures
                if self.consecutive_spot_failures < 3:
                    self.consecutive_spot_failures = 0
                    self.consecutive_spot_successes += 1
                    return ClusterType.SPOT
        
        # Rule 2: If we just switched regions or had spot failures, use on-demand
        time_since_switch = elapsed - self.last_region_switch_time
        if (time_since_switch < self.time_step * 2 or 
            self.consecutive_spot_failures >= 2):
            self.consecutive_spot_failures = 0
            return ClusterType.ON_DEMAND
        
        # Rule 3: If no spot and not urgent, wait (NONE)
        if not has_spot and urgency < 0.7:
            # Only wait if we have time buffer
            buffer_needed = task_remaining * 1.2 + self.restart_overhead * 2
            if time_remaining > buffer_needed:
                return ClusterType.NONE
        
        # Default: use on-demand
        self.consecutive_spot_failures = 0
        return ClusterType.ON_DEMAND
    
    def _initialize_state(self):
        """Initialize the strategy state."""
        self.region_count = self.env.get_num_regions()
        self.spot_availability = [0] * self.region_count
        self.spot_history = [[] for _ in range(self.region_count)]
        self.spot_score = [0.0] * self.region_count
        self.initialized = True
    
    def _update_spot_history(self, region: int, has_spot: bool):
        """Update spot availability history for a region."""
        if len(self.spot_history[region]) >= 20:  # Keep last 20 observations
            self.spot_history[region].pop(0)
        self.spot_history[region].append(1 if has_spot else 0)
        
        # Update consecutive counters
        if has_spot:
            self.consecutive_spot_successes += 1
            self.consecutive_spot_failures = 0
        else:
            self.consecutive_spot_failures += 1
            self.consecutive_spot_successes = 0
    
    def _get_spot_success_rate(self, region: int) -> float:
        """Calculate spot success rate for a region."""
        if not self.spot_history[region]:
            return 0.0
        return sum(self.spot_history[region]) / len(self.spot_history[region])
    
    def _update_region_scores(self):
        """Update scores for each region based on spot history."""
        for i in range(self.region_count):
            success_rate = self._get_spot_success_rate(i)
            # Weight recent history more heavily
            if len(self.spot_history[i]) >= 5:
                recent = self.spot_history[i][-5:]
                recent_rate = sum(recent) / len(recent)
                success_rate = 0.3 * success_rate + 0.7 * recent_rate
            
            self.spot_score[i] = success_rate
    
    def _should_switch_region(self, current_region: int, urgency: float) -> bool:
        """Determine if we should switch regions."""
        # Don't switch too frequently
        time_since_switch = self.env.elapsed_seconds - self.last_region_switch_time
        if time_since_switch < self.time_step * 3:  # At least 3 hours between switches
            return False
        
        # Don't switch if we're in emergency mode
        if self.emergency_mode:
            return False
        
        # Check current region performance
        current_score = self.spot_score[current_region]
        
        # Find best alternative region
        best_alt_score = 0.0
        for i in range(self.region_count):
            if i != current_region:
                best_alt_score = max(best_alt_score, self.spot_score[i])
        
        # Switch if alternative is significantly better
        score_threshold = 0.2 + urgency * 0.3  # More willing to switch when urgent
        return best_alt_score - current_score > score_threshold
    
    def _select_best_region(self, current_region: int) -> int:
        """Select the best region to switch to."""
        best_score = -1.0
        best_region = current_region
        
        for i in range(self.region_count):
            if i == current_region:
                continue
                
            score = self.spot_score[i]
            # Add small random tie-breaker
            import random
            score += random.random() * 0.01
            
            if score > best_score:
                best_score = score
                best_region = i
        
        return best_region