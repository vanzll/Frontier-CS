import json
from argparse import Namespace
import heapq
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with lookahead planning."""
    
    NAME = "lookahead_planner"
    
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
        
        # Additional initialization
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.plan = []
        self.current_plan_index = 0
        self.last_region = -1
        self.consecutive_spot_fails = 0
        self.time_awareness_threshold = 4 * 3600  # 4 hours in seconds
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # Get current state
        current_region = self.env.get_current_region()
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        task_done = sum(self.task_done_time)
        remaining_work = self.task_duration - task_done
        
        # Check if we need to replan
        if (not self.plan or 
            self.last_region != current_region or
            elapsed - self.plan_created_at > 3600):  # Replan every hour
            
            self._create_plan(elapsed, remaining_work, deadline, current_region)
            self.current_plan_index = 0
            self.last_region = current_region
        
        # If we have a plan, follow it
        if self.plan and self.current_plan_index < len(self.plan):
            action = self.plan[self.current_plan_index]
            self.current_plan_index += 1
            
            # Validate spot availability
            if action == ClusterType.SPOT and not has_spot:
                # Fallback to on-demand if spot not available
                return ClusterType.ON_DEMAND
            return action
        
        # Fallback: greedy strategy
        return self._fallback_strategy(has_spot, elapsed, remaining_work, deadline)
    
    def _create_plan(self, current_time: float, remaining_work: float, 
                     deadline: float, current_region: int) -> None:
        """
        Create an execution plan for the next few hours.
        """
        self.plan = []
        self.plan_created_at = current_time
        
        # Calculate available time
        time_left = deadline - current_time
        safety_margin = 3600  # 1 hour safety margin
        
        # If we're running out of time, switch to on-demand
        if time_left - safety_margin < remaining_work + self.restart_overhead:
            self.plan = [ClusterType.ON_DEMAND]
            return
        
        # Estimate optimal mix of spot and on-demand
        # Use spot when we have time buffer, on-demand when buffer is small
        spot_ratio = min(0.8, (time_left - remaining_work) / (2 * 3600))  # 2 hour scaling
        
        # Create plan for next 3 hours or until work is done
        plan_horizon = min(3 * 3600, time_left)
        steps = int(plan_horizon / self.env.gap_seconds)
        
        for i in range(steps):
            step_time = current_time + i * self.env.gap_seconds
            time_after_step = step_time + self.env.gap_seconds
            
            # Calculate remaining work at this step
            work_done_by_now = min(remaining_work, i * self.env.gap_seconds)
            remaining_at_step = remaining_work - work_done_by_now
            
            # Calculate time buffer
            time_buffer = deadline - time_after_step - remaining_at_step
            
            # Decide action based on time buffer and spot availability probability
            if time_buffer > self.time_awareness_threshold:
                # Good time buffer, try spot
                self.plan.append(ClusterType.SPOT)
            elif time_buffer > 3600:  # 1 hour buffer
                # Moderate buffer, use spot if we haven't failed too much recently
                if self.consecutive_spot_fails < 3:
                    self.plan.append(ClusterType.SPOT)
                else:
                    self.plan.append(ClusterType.ON_DEMAND)
            else:
                # Low time buffer, use on-demand
                self.plan.append(ClusterType.ON_DEMAND)
            
            # Break if we've scheduled enough work
            if len(self.plan) * self.env.gap_seconds >= remaining_work:
                break
        
        # If plan is empty or we need more aggressive strategy
        if not self.plan and remaining_work > 0:
            self.plan = [ClusterType.ON_DEMAND]
    
    def _fallback_strategy(self, has_spot: bool, elapsed: float, 
                          remaining_work: float, deadline: float) -> ClusterType:
        """
        Fallback greedy strategy when plan is not available.
        """
        time_left = deadline - elapsed
        
        # Check if we're in critical time
        is_critical = time_left < remaining_work + 2 * self.restart_overhead
        
        if is_critical:
            # Critical: use on-demand to ensure completion
            self.consecutive_spot_fails = 0
            return ClusterType.ON_DEMAND
        
        # Check if we should consider switching regions
        current_region = self.env.get_current_region()
        num_regions = self.env.get_num_regions()
        
        if not has_spot and self.consecutive_spot_fails >= 2:
            # Try switching regions to find spot
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            self.consecutive_spot_fails = 0
            # After switching, we need to restart, so use on-demand for this step
            return ClusterType.ON_DEMAND
        
        # Normal operation
        if has_spot:
            self.consecutive_spot_fails = 0
            return ClusterType.SPOT
        else:
            self.consecutive_spot_fails += 1
            return ClusterType.ON_DEMAND
    
    def _estimate_spot_availability(self, region: int, time: float) -> float:
        """
        Simple estimation of spot availability.
        In a real implementation, this would use historical data.
        """
        # Base availability - higher during off-peak hours
        hour_of_day = (time / 3600) % 24
        if 1 <= hour_of_day <= 6:  # Early morning
            base_avail = 0.9
        elif 9 <= hour_of_day <= 17:  # Business hours
            base_avail = 0.7
        else:  # Evening
            base_avail = 0.8
        
        # Add some region variation
        region_factor = 1.0 - (region * 0.05)
        return min(0.95, max(0.5, base_avail * region_factor))