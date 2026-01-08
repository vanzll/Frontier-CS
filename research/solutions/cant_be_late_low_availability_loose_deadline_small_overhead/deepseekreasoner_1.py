import json
import math
from enum import Enum
from typing import List, Tuple, Optional
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class SolutionState(Enum):
    SPOT_RUNNING = 1
    ON_DEMAND_RUNNING = 2
    NONE_RUNNING = 3
    OVERHEAD = 4

class Solution(Strategy):
    NAME = "adaptive_threshold"
    
    def __init__(self, args=None):
        super().__init__(args)
        self.state = SolutionState.NONE_RUNNING
        self.time_in_state = 0
        self.overhead_remaining = 0
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.spot_availability_history = []
        self.last_decision = ClusterType.NONE
        
        # Configurable parameters
        self.spot_aggressiveness = 0.7  # Higher = more aggressive with spot
        self.safety_buffer_factor = 1.2  # Buffer for deadline
        self.min_spot_confidence = 3  # Min consecutive spot available to switch to spot
        self.max_overhead_risk = 0.1  # Max fraction of remaining time to risk overhead
        
    def solve(self, spec_path: str) -> "Solution":
        """Optional initialization from spec file"""
        try:
            with open(spec_path, 'r') as f:
                spec = json.load(f)
                # Could extract parameters from spec here
                if 'spot_aggressiveness' in spec:
                    self.spot_aggressiveness = spec['spot_aggressiveness']
        except:
            pass  # Use defaults if spec file not found or invalid
        return self
    
    def _update_state(self, last_cluster_type: ClusterType, has_spot: bool) -> None:
        """Update internal state tracking"""
        # Update spot availability history
        self.spot_availability_history.append(has_spot)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        # Update consecutive counters
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0
        
        # Update overhead tracking
        if self.overhead_remaining > 0:
            self.overhead_remaining = max(0, self.overhead_remaining - self.env.gap_seconds)
        
        # Update state machine
        if last_cluster_type == ClusterType.SPOT:
            if self.overhead_remaining > 0:
                self.state = SolutionState.OVERHEAD
            else:
                self.state = SolutionState.SPOT_RUNNING
        elif last_cluster_type == ClusterType.ON_DEMAND:
            self.state = SolutionState.ON_DEMAND_RUNNING
        else:
            self.state = SolutionState.NONE_RUNNING
    
    def _calculate_metrics(self) -> Tuple[float, float, float]:
        """Calculate critical metrics for decision making"""
        elapsed = self.env.elapsed_seconds
        total_work_done = sum(self.task_done_time) if self.task_done_time else 0
        remaining_work = self.task_duration - total_work_done
        time_to_deadline = self.deadline - elapsed
        
        # Calculate work rate (adjusted for overheads)
        effective_work_rate = 1.0  # Assuming 1 unit per second when running
        
        # Calculate critical threshold - minimum time needed to finish
        min_time_needed = remaining_work / effective_work_rate
        
        # Calculate safe threshold with buffer
        safe_time_needed = min_time_needed * self.safety_buffer_factor
        
        # Calculate slack ratio (how much buffer we have)
        if time_to_deadline > 0:
            slack_ratio = time_to_deadline / safe_time_needed
        else:
            slack_ratio = 0
            
        return remaining_work, time_to_deadline, slack_ratio
    
    def _should_switch_to_on_demand(self, slack_ratio: float, 
                                   has_spot: bool) -> bool:
        """Determine if we should switch to on-demand"""
        if slack_ratio < 1.0:
            return True  # We're behind schedule
        
        # If we're close to deadline and spot is unreliable
        if slack_ratio < 1.5 and self.consecutive_spot_unavailable > 5:
            return True
            
        # If spot has been unavailable for too long
        if not has_spot and self.consecutive_spot_unavailable > 20:
            return True
            
        return False
    
    def _should_switch_to_spot(self, slack_ratio: float, 
                              has_spot: bool) -> bool:
        """Determine if we should switch to spot"""
        if not has_spot:
            return False
            
        # Need minimum confidence in spot availability
        if self.consecutive_spot_available < self.min_spot_confidence:
            return False
        
        # Calculate spot availability probability from history
        if len(self.spot_availability_history) > 10:
            spot_prob = sum(self.spot_availability_history[-20:]) / min(20, len(self.spot_availability_history))
        else:
            spot_prob = 1.0 if has_spot else 0.0
        
        # Only use spot if we have enough slack to handle interruptions
        required_slack_for_spot = (self.restart_overhead * 2) / (self.deadline - self.env.elapsed_seconds)
        
        # More aggressive with spot when we have more slack
        spot_threshold = self.spot_aggressiveness * slack_ratio
        
        return (slack_ratio > 1.5 and 
                spot_prob > 0.3 and 
                slack_ratio > required_slack_for_spot and
                slack_ratio > spot_threshold)
    
    def _should_pause(self, slack_ratio: float) -> bool:
        """Determine if we should pause"""
        # Only pause if we're way ahead of schedule
        if slack_ratio > 3.0 and self.state not in [SolutionState.ON_DEMAND_RUNNING, SolutionState.OVERHEAD]:
            return True
            
        # Pause if we're in overhead and spot isn't available
        if self.state == SolutionState.OVERHEAD and self.consecutive_spot_unavailable > 0:
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Main decision logic for each time step"""
        # Update internal state
        self._update_state(last_cluster_type, has_spot)
        
        # Calculate critical metrics
        remaining_work, time_to_deadline, slack_ratio = self._calculate_metrics()
        
        # Check if we've already finished
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Check for deadline violation risk
        if time_to_deadline <= 0:
            # Emergency mode - use whatever works
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Determine next action based on state machine
        if self.state == SolutionState.OVERHEAD:
            # In overhead period - continue with current type or pause
            if self.overhead_remaining > 0:
                # If we initiated overhead, continue with last decision
                if self.last_decision != ClusterType.NONE:
                    return self.last_decision
                # Otherwise pause during overhead
                return ClusterType.NONE
        
        # Decision logic
        if self._should_switch_to_on_demand(slack_ratio, has_spot):
            decision = ClusterType.ON_DEMAND
        elif self._should_switch_to_spot(slack_ratio, has_spot):
            decision = ClusterType.SPOT
        elif self._should_pause(slack_ratio):
            decision = ClusterType.NONE
        else:
            # Maintain current state if possible
            if last_cluster_type == ClusterType.SPOT and has_spot:
                decision = ClusterType.SPOT
            elif last_cluster_type == ClusterType.ON_DEMAND:
                decision = ClusterType.ON_DEMAND
            else:
                # Default to on-demand if nothing else works
                decision = ClusterType.ON_DEMAND
        
        # Check if this decision triggers a restart overhead
        if (last_cluster_type in [ClusterType.NONE, ClusterType.SPOT] and 
            decision == ClusterType.ON_DEMAND):
            # Switching to on-demand from nothing or spot
            self.overhead_remaining = self.restart_overhead
        elif (last_cluster_type in [ClusterType.NONE, ClusterType.ON_DEMAND] and 
              decision == ClusterType.SPOT):
            # Switching to spot from nothing or on-demand
            self.overhead_remaining = self.restart_overhead
        
        # Store last decision for overhead tracking
        self.last_decision = decision
        
        return decision
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)