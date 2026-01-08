import argparse
import math
from enum import Enum
from typing import List, Tuple

try:
    from sky_spot.strategies.strategy import Strategy
    from sky_spot.utils import ClusterType
except ImportError:
    # For local testing when dependencies aren't available
    class ClusterType(Enum):
        SPOT = "SPOT"
        ON_DEMAND = "ON_DEMAND"
        NONE = "NONE"

    class Strategy:
        def __init__(self):
            self.env = type('obj', (object,), {
                'elapsed_seconds': 0,
                'gap_seconds': 1,
                'cluster_type': ClusterType.NONE
            })()
            self.task_duration = 0
            self.task_done_time = []
            self.deadline = 0
            self.restart_overhead = 0


class Solution(Strategy):
    NAME = "adaptive_safe"

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self._initialize_state()

    def _initialize_state(self):
        """Initialize internal state variables"""
        self.total_work_needed = 0
        self.work_done = 0
        self.remaining_work = 0
        self.time_elapsed = 0
        self.deadline_time = 0
        self.step_size = 0
        self.spot_price = 0.97
        self.on_demand_price = 3.06
        self.spot_available = False
        self.current_cluster = ClusterType.NONE
        self.in_restart = False
        self.restart_timer = 0
        self.consecutive_spot_failures = 0
        self.spot_history = []
        self.spot_availability_rate = 0.5  # Initial estimate
        self.min_safe_time = 0
        self.last_action = ClusterType.NONE
        self.conservative_mode = False
        self.emergency_mode = False

    def solve(self, spec_path: str) -> "Solution":
        """Optional initialization"""
        try:
            # Read spec file if provided (for future compatibility)
            pass
        except:
            pass
        return self

    def _update_state(self, last_cluster_type: ClusterType, has_spot: bool):
        """Update internal state based on environment and inputs"""
        self.time_elapsed = self.env.elapsed_seconds
        self.step_size = self.env.gap_seconds
        self.current_cluster = last_cluster_type
        self.spot_available = has_spot
        
        # Update work progress
        if not self.in_restart and self.current_cluster != ClusterType.NONE:
            work_this_step = self.step_size
            self.work_done += work_this_step
            
        # Update restart timer if applicable
        if self.in_restart:
            self.restart_timer -= self.step_size
            if self.restart_timer <= 0:
                self.in_restart = False
                self.restart_timer = 0
        
        # Update remaining work
        self.remaining_work = self.task_duration - self.work_done
        
        # Update spot availability history
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > 100:
            self.spot_history.pop(0)
        
        # Calculate spot availability rate (weighted recent history)
        if len(self.spot_history) > 0:
            recent_window = min(20, len(self.spot_history))
            recent_availability = sum(self.spot_history[-recent_window:]) / recent_window
            overall_availability = sum(self.spot_history) / len(self.spot_history)
            self.spot_availability_rate = 0.7 * recent_availability + 0.3 * overall_availability
        
        # Update failure tracking
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = max(0, self.consecutive_spot_failures - 1)
        
        # Calculate safety margins
        time_left = self.deadline - self.time_elapsed
        self.min_safe_time = self.remaining_work + self.restart_overhead
        
        # Determine modes
        self.conservative_mode = (time_left < self.min_safe_time * 1.5 or 
                                 self.consecutive_spot_failures >= 3)
        self.emergency_mode = (time_left < self.min_safe_time * 1.2)

    def _should_use_spot(self) -> bool:
        """Determine if we should attempt to use spot instances"""
        if not self.spot_available:
            return False
        
        time_left = self.deadline - self.time_elapsed
        required_time = self.remaining_work + self.restart_overhead
        
        # Emergency mode: only use on-demand
        if self.emergency_mode:
            return False
        
        # Conservative mode: use spot only if we have good availability
        if self.conservative_mode:
            min_availability = 0.6
            if self.spot_availability_rate < min_availability:
                return False
        
        # Calculate risk-adjusted expected time for spot
        spot_success_prob = self.spot_availability_rate ** 2  # Square for consecutive steps
        expected_spot_time = self.remaining_work / spot_success_prob
        expected_spot_time_with_overhead = expected_spot_time + self.restart_overhead * 2
        
        # Only use spot if we have enough time buffer
        time_buffer_needed = expected_spot_time_with_overhead * 1.3
        if time_left > time_buffer_needed:
            return True
        
        return False

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Main decision logic for each time step"""
        # Initialize on first call
        if not hasattr(self, 'total_work_needed'):
            self._initialize_state()
            self.total_work_needed = self.task_duration
            self.deadline_time = self.deadline
        
        # Update internal state
        self._update_state(last_cluster_type, has_spot)
        
        # Check if we're done
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        # Check restart status
        if self.in_restart:
            return ClusterType.NONE
        
        # Handle cluster switching restart overhead
        if (last_cluster_type == ClusterType.NONE and 
            self.last_action != ClusterType.NONE):
            # Starting from NONE, need restart overhead
            self.in_restart = True
            self.restart_timer = self.restart_overhead
            return ClusterType.NONE
        
        # Emergency mode: use on-demand to guarantee completion
        if self.emergency_mode:
            self.last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        # Try to use spot if conditions are favorable
        if self._should_use_spot():
            self.last_action = ClusterType.SPOT
            return ClusterType.SPOT
        
        # Use on-demand if we can't use spot and need to make progress
        time_left = self.deadline - self.time_elapsed
        if time_left < self.remaining_work * 1.5:
            self.last_action = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        # Otherwise pause (only if we have enough time buffer)
        self.last_action = ClusterType.NONE
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


# For local testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    solution = Solution._from_args(parser)
    print(f"Solution name: {solution.NAME}")