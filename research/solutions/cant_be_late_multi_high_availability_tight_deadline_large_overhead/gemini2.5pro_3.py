import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "lookahead_optimizer"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.

        The spec file contains:
        - deadline: deadline in hours
        - duration: task duration in hours
        - overhead: restart overhead in hours
        - trace_files: list of trace file paths (one per region)
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

        # Pre-load spot availability traces for fast lookups
        self.spot_availability = []
        if 'trace_files' in config:
            for trace_file in config["trace_files"]:
                with open(trace_file) as f:
                    # Assuming trace file contains '0' or '1' per line
                    region_trace = [line.strip() == '1' for line in f]
                    self.spot_availability.append(region_trace)
        
        # Set up heuristic parameters
        if self.env.gap_seconds > 0:
            # Look ahead a reasonable number of steps to evaluate regions
            self.lookahead_steps = min(
                int(self.task_duration / self.env.gap_seconds), 
                200 
            )
            self.switch_penalty_steps = self.restart_overhead / self.env.gap_seconds
        else: # Fallback
            self.lookahead_steps = 100
            self.switch_penalty_steps = 2
        
        # A safety buffer to switch to on-demand before the deadline is imminent
        self.safety_buffer = self.env.gap_seconds * 2

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Calculate current state variables
        current_time = self.env.elapsed_seconds
        current_progress = sum(self.task_done_time)
        remaining_work = self.task_duration - current_progress
        
        # 2. If task is done, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - current_time
        current_region = self.env.get_current_region()
        current_timestep = int(round(current_time / self.env.gap_seconds))

        # 3. Criticality Check: Switch to ON_DEMAND if deadline is at risk
        # This is the "panic mode" to ensure task completion.
        # It calculates if the job can be finished using only on-demand from now on.
        od_time_needed = remaining_work + self.restart_overhead
        if od_time_needed + self.safety_buffer >= time_to_deadline:
            return ClusterType.ON_DEMAND

        # 4. Region Selection: Decide if switching regions is beneficial
        num_regions = self.env.get_num_regions()
        if not self.spot_availability or num_regions <= 1:
            # No traces loaded or no other regions to switch to
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

        # Score each region based on future spot availability
        scores = []
        for r_idx in range(num_regions):
            num_spot_steps = 0
            if r_idx < len(self.spot_availability):
                trace = self.spot_availability[r_idx]
                max_timestep = len(trace)
                end_idx = min(max_timestep, current_timestep + self.lookahead_steps)
                # Summing a slice is efficient for this calculation
                if current_timestep < end_idx:
                    num_spot_steps = sum(trace[current_timestep:end_idx])

            # Apply a penalty for switching to account for the restart overhead
            penalty = self.switch_penalty_steps if r_idx != current_region else 0
            scores.append(num_spot_steps - penalty)

        best_score = max(scores)
        best_region_idx = scores.index(best_score)
        
        current_score = scores[current_region]

        # 5. Make Decision: Switch or Stay
        if best_region_idx != current_region and best_score > current_score:
            # Switch if another region is better, even after the penalty
            self.env.switch_region(best_region_idx)
            
            # Check for spot availability in the new region for the current timestep
            new_region_has_spot = False
            if best_region_idx < len(self.spot_availability):
                trace = self.spot_availability[best_region_idx]
                if current_timestep < len(trace):
                    new_region_has_spot = trace[current_timestep]

            if new_region_has_spot:
                return ClusterType.SPOT
            else:
                # Switched for future gains, so wait (NONE) this turn
                return ClusterType.NONE
        else:
            # Stay in the current region
            if has_spot:
                return ClusterType.SPOT
            else:
                # Not in critical mode, so wait for spot to become available
                return ClusterType.NONE