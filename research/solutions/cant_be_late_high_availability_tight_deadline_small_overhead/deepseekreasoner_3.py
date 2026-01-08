import math
import random
from typing import List, Optional, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self._safety_margin = 0.3  # 30% safety margin
        self._switch_cost_factor = 1.5  # Penalty for switching
        self._spot_history = []
        self._consecutive_spot_failures = 0
        self._last_spot_available = False
        self._adaptive_risk_tolerance = 1.0
        self._critical_threshold = 0.85  # Time usage threshold to go conservative
        
    def solve(self, spec_path: str) -> "Solution":
        # Read configuration if needed (optional)
        return self
    
    def _estimate_completion_time(self, remaining_work: float, use_spot: bool, has_spot: bool) -> float:
        """Estimate time to complete remaining work with given strategy."""
        if use_spot and not has_spot:
            # Can't use spot when unavailable
            return float('inf')
        
        # Base completion time
        if use_spot:
            time = remaining_work
        else:
            time = remaining_work
        
        # Add restart overhead if switching
        current_type = self.env.cluster_type
        if (use_spot and current_type != ClusterType.SPOT) or \
           (not use_spot and current_type != ClusterType.ON_DEMAND):
            time += self.restart_overhead
        
        return time
    
    def _calculate_risk_score(self) -> float:
        """Calculate risk score based on time remaining and progress."""
        elapsed = self.env.elapsed_seconds
        time_used = elapsed / self.deadline
        
        total_done = sum(self.task_done_time)
        progress = total_done / self.task_duration if self.task_duration > 0 else 0
        
        # Calculate slack ratio
        time_remaining = self.deadline - elapsed
        work_remaining = self.task_duration - total_done
        required_rate = work_remaining / time_remaining if time_remaining > 0 else float('inf')
        
        # Risk increases as we approach deadline or fall behind
        risk = 0.0
        if time_used > self._critical_threshold:
            risk += (time_used - self._critical_threshold) * 2.0
        if required_rate > 1.0:  # Need faster than real-time
            risk += (required_rate - 1.0) * 1.5
        
        # Adjust based on spot reliability
        if len(self._spot_history) > 10:
            spot_success_rate = sum(self._spot_history[-10:]) / 10
            risk += (1.0 - spot_success_rate) * 0.5
        
        return min(max(risk, 0.0), 3.0)
    
    def _should_use_spot(self, has_spot: bool, risk_score: float) -> bool:
        """Determine if we should use spot instances."""
        if not has_spot:
            return False
        
        # Check if we're in a critical situation
        elapsed = self.env.elapsed_seconds
        total_done = sum(self.task_done_time)
        time_remaining = self.deadline - elapsed
        work_remaining = self.task_duration - total_done
        
        # Calculate if we can afford spot failures
        spot_time_needed = work_remaining
        if self.env.cluster_type != ClusterType.SPOT:
            spot_time_needed += self.restart_overhead
        
        # Add safety margin
        spot_time_needed *= (1.0 + self._safety_margin)
        
        # Conservative estimate: assume some spot unavailability
        conservative_time = spot_time_needed * 1.2
        
        if conservative_time > time_remaining:
            # Not enough time even with conservative estimate
            return False
        
        # Adjust decision based on risk
        spot_probability = 0.7 - (risk_score * 0.2)
        return random.random() < spot_probability
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update spot history
        self._spot_history.append(1 if has_spot else 0)
        if len(self._spot_history) > 100:
            self._spot_history.pop(0)
        
        # Track consecutive spot failures
        if not has_spot and self._last_spot_available:
            self._consecutive_spot_failures += 1
        else:
            self._consecutive_spot_failures = max(0, self._consecutive_spot_failures - 1)
        self._last_spot_available = has_spot
        
        # Calculate progress and risk
        elapsed = self.env.elapsed_seconds
        total_done = sum(self.task_done_time)
        work_remaining = self.task_duration - total_done
        time_remaining = self.deadline - elapsed
        
        # Check if we've completed the task
        if total_done >= self.task_duration:
            return ClusterType.NONE
        
        # Emergency mode: must finish
        if time_remaining <= work_remaining + self.restart_overhead:
            # Not enough time even with on-demand (shouldn't happen with proper margin)
            return ClusterType.ON_DEMAND
        
        risk_score = self._calculate_risk_score()
        
        # Determine if we should use spot
        use_spot = self._should_use_spot(has_spot, risk_score)
        
        # Special cases for NONE
        if time_remaining > work_remaining * 1.5 and risk_score < 0.5:
            # We have plenty of time, can pause to save cost
            if random.random() < 0.1:  # 10% chance to pause
                return ClusterType.NONE
        
        # Make decision
        if use_spot and has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)