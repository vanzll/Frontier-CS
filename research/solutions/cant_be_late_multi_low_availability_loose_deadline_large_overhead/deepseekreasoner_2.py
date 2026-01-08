import json
from argparse import Namespace
from typing import List, Tuple
import heapq
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""
    
    NAME = "my_strategy"

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
        
        # Initialize state
        self.region_count = None
        self.spot_prices = []
        self.od_price = 3.06  # $/hour
        self.spot_price = 0.9701  # $/hour
        self.step_size = 3600.0  # seconds (1 hour)
        self.overhead_steps = int(math.ceil(self.restart_overhead / self.step_size))
        
        # Planning attributes
        self.plan = []
        self.current_plan_index = 0
        self.last_action = ClusterType.NONE
        self.consecutive_spot_failures = 0
        self.max_spot_failures = 3
        
        return self

    def _load_traces(self) -> List[List[bool]]:
        """Load spot availability traces for all regions."""
        traces = []
        for i in range(self.env.get_num_regions()):
            # In actual implementation, traces would be loaded from config
            # For now, we'll handle via has_spot parameter
            traces.append([])
        return traces

    def _compute_safe_schedule(self) -> List[Tuple[int, ClusterType]]:
        """Compute a safe schedule that guarantees meeting the deadline."""
        total_work_needed = self.task_duration
        deadline_steps = int(self.deadline / self.step_size)
        work_per_step = self.step_size
        
        # Calculate minimum steps needed with on-demand
        min_work_steps = int(math.ceil(total_work_needed / work_per_step))
        if self.overhead_steps > 0:
            min_work_steps += self.overhead_steps  # Add one potential restart
        
        # If we can't finish even with all on-demand, return conservative plan
        if min_work_steps > deadline_steps:
            # Use all on-demand starting immediately
            plan = []
            for step in range(deadline_steps):
                plan.append((self.env.get_current_region(), ClusterType.ON_DEMAND))
            return plan
        
        # Calculate slack time
        slack_steps = deadline_steps - min_work_steps
        
        # Create a mixed plan: use spot when available, fallback to on-demand
        # Start with on-demand for first few steps to build buffer
        buffer_steps = min(3, slack_steps // 2)
        plan = []
        
        current_region = self.env.get_current_region()
        steps_remaining = deadline_steps
        work_remaining = total_work_needed
        buffer_built = False
        
        while steps_remaining > 0 and work_remaining > 0:
            if not buffer_built and buffer_steps > 0:
                # Build buffer with on-demand
                plan.append((current_region, ClusterType.ON_DEMAND))
                work_remaining = max(0, work_remaining - work_per_step)
                buffer_steps -= 1
                if buffer_steps == 0:
                    buffer_built = True
            elif slack_steps > 0 and work_remaining > slack_steps * work_per_step:
                # Need to use spot to save cost while maintaining schedule
                # Try spot first, with region switching if needed
                best_region = current_region
                plan.append((best_region, ClusterType.SPOT))
                # Assume spot succeeds for planning
                work_remaining = max(0, work_remaining - work_per_step)
                slack_steps -= 1
            else:
                # Use on-demand to ensure we finish on time
                plan.append((current_region, ClusterType.ON_DEMAND))
                work_remaining = max(0, work_remaining - work_per_step)
            
            steps_remaining -= 1
            
            # Occasionally switch regions in plan to find better spot availability
            if len(plan) % 4 == 0 and self.env.get_num_regions() > 1:
                current_region = (current_region + 1) % self.env.get_num_regions()
        
        # Fill remaining steps with NONE if work is done
        while len(plan) < deadline_steps:
            plan.append((current_region, ClusterType.NONE))
        
        return plan[:deadline_steps]

    def _should_switch_to_od(self, time_remaining: float, work_remaining: float) -> bool:
        """Determine if we should switch to on-demand based on time constraints."""
        # Calculate time needed with on-demand (no additional overheads)
        od_time_needed = work_remaining + self.restart_overhead
        
        # Calculate time needed if we try spot and fail
        spot_fail_time_needed = work_remaining + 2 * self.restart_overhead
        
        # If we're running out of time, switch to on-demand
        safety_margin = 2 * self.step_size  # 2 hour safety margin
        return time_remaining < od_time_needed + safety_margin

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Initialize on first call
        if self.region_count is None:
            self.region_count = self.env.get_num_regions()
            self.plan = self._compute_safe_schedule()
            self.current_plan_index = 0
        
        # Get current state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        work_done = sum(self.task_done_time) if self.task_done_time else 0
        work_remaining = self.task_duration - work_done
        time_remaining = deadline - elapsed
        
        # Check if we've finished
        if work_remaining <= 0:
            return ClusterType.NONE
        
        # Check if we're out of time
        if time_remaining <= 0:
            return ClusterType.NONE
        
        # Get current region from plan
        if self.current_plan_index < len(self.plan):
            planned_region, planned_action = self.plan[self.current_plan_index]
            current_region = self.env.get_current_region()
            
            # Switch region if needed
            if planned_region != current_region:
                self.env.switch_region(planned_region)
                # Switching causes restart, so we need to account for overhead
                # Return the planned action for the new region
                action = planned_action
            else:
                action = planned_action
            
            # Adjust action based on actual spot availability
            if action == ClusterType.SPOT and not has_spot:
                self.consecutive_spot_failures += 1
                
                # If too many spot failures in a row, try different region or switch to OD
                if self.consecutive_spot_failures >= self.max_spot_failures:
                    # Try a different region
                    if self.region_count > 1:
                        next_region = (planned_region + 1) % self.region_count
                        self.env.switch_region(next_region)
                        action = ClusterType.SPOT
                    else:
                        # No other regions, use on-demand if time is tight
                        if self._should_switch_to_od(time_remaining, work_remaining):
                            action = ClusterType.ON_DEMAND
                        else:
                            # Wait for spot
                            action = ClusterType.NONE
                else:
                    # Wait for spot or use on-demand based on time pressure
                    if self._should_switch_to_od(time_remaining, work_remaining):
                        action = ClusterType.ON_DEMAND
                    else:
                        action = ClusterType.NONE
            elif action == ClusterType.SPOT and has_spot:
                self.consecutive_spot_failures = 0
            else:
                self.consecutive_spot_failures = 0
            
            self.current_plan_index += 1
            self.last_action = action
            return action
        
        # Fallback: if no plan or plan exhausted, use conservative strategy
        if has_spot and not self._should_switch_to_od(time_remaining, work_remaining):
            # Use spot if available and we have time
            self.consecutive_spot_failures = 0
            self.last_action = ClusterType.SPOT
            return ClusterType.SPOT
        else:
            # Use on-demand or wait
            if self._should_switch_to_od(time_remaining, work_remaining):
                self.last_action = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND
            else:
                # Try to find a region with spot
                if self.region_count > 1:
                    current = self.env.get_current_region()
                    next_region = (current + 1) % self.region_count
                    self.env.switch_region(next_region)
                    self.last_action = ClusterType.SPOT
                    return ClusterType.SPOT
                else:
                    self.last_action = ClusterType.NONE
                    return ClusterType.NONE