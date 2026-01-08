import json
from argparse import Namespace
from typing import List
import heapq

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Multi-region scheduling strategy with cost optimization."""
    
    NAME = "cost_optimized_multi_region"

    def solve(self, spec_path: str) -> "Solution":
        """Initialize strategy from configuration."""
        with open(spec_path) as f:
            config = json.load(f)
        
        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)
        
        # Load trace data
        self.trace_files = config["trace_files"]
        self.traces = []
        for trace_file in self.trace_files:
            with open(trace_file, 'r') as f:
                # Parse trace file: each line contains availability (0/1)
                availability = [int(line.strip()) for line in f if line.strip()]
                self.traces.append(availability)
        
        # Precompute spot availability windows per region
        self.spot_windows = self._compute_spot_windows()
        
        # Constants from problem description
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.time_step = 3600.0  # 1 hour in seconds
        
        return self
    
    def _compute_spot_windows(self) -> List[List[tuple]]:
        """Compute contiguous spot availability windows for each region."""
        windows = []
        for trace in self.traces:
            region_windows = []
            start = None
            for i, avail in enumerate(trace):
                if avail == 1 and start is None:
                    start = i
                elif avail == 0 and start is not None:
                    region_windows.append((start, i-1))
                    start = None
            if start is not None:
                region_windows.append((start, len(trace)-1))
            windows.append(region_windows)
        return windows
    
    def _get_remaining_time(self) -> float:
        """Calculate remaining time until deadline."""
        return self.deadline - self.env.elapsed_seconds
    
    def _get_work_done(self) -> float:
        """Calculate total work completed."""
        return sum(self.task_done_time)
    
    def _get_remaining_work(self) -> float:
        """Calculate remaining work needed."""
        return self.task_duration - self._get_work_done()
    
    def _estimate_min_cost_to_finish(self, current_time: float, remaining_work: float) -> float:
        """Estimate minimum cost to finish remaining work."""
        # Calculate time steps needed
        time_steps_needed = int(remaining_work / self.time_step)
        if remaining_work % self.time_step > 0:
            time_steps_needed += 1
        
        # If we have enough time, we can use all spot
        if current_time + time_steps_needed * self.time_step <= self.deadline:
            return time_steps_needed * self.spot_price * (self.time_step / 3600.0)
        
        # Otherwise, we need to mix spot and on-demand
        spot_time = max(0, self.deadline - current_time - self.restart_overhead)
        ondemand_time = remaining_work - spot_time
        
        if ondemand_time <= 0:
            return spot_time / 3600.0 * self.spot_price
        
        return (spot_time / 3600.0 * self.spot_price + 
                ondemand_time / 3600.0 * self.ondemand_price)
    
    def _find_best_spot_window(self, current_time_step: int, remaining_work: float) -> tuple:
        """Find the best region and spot window to minimize cost."""
        best_region = -1
        best_window = None
        best_score = float('inf')
        
        current_region = self.env.get_current_region()
        remaining_steps = int(remaining_work / self.time_step)
        if remaining_work % self.time_step > 0:
            remaining_steps += 1
        
        for region_idx, windows in enumerate(self.spot_windows):
            for window_start, window_end in windows:
                if window_start < current_time_step:
                    continue
                
                # Calculate available steps in this window
                available_steps = min(window_end - current_time_step + 1, remaining_steps)
                
                # Calculate cost if we use this window
                spot_time = available_steps * self.time_step
                if spot_time >= remaining_work:
                    # Can finish entirely with spot
                    cost = remaining_work / 3600.0 * self.spot_price
                else:
                    # Need on-demand for the rest
                    ondemand_time = remaining_work - spot_time
                    cost = (spot_time / 3600.0 * self.spot_price + 
                           ondemand_time / 3600.0 * self.ondemand_price)
                
                # Add penalty for switching regions
                if region_idx != current_region:
                    cost += self.restart_overhead / 3600.0 * self.spot_price
                
                # Prioritize windows that can complete more work
                score = cost / available_steps
                
                if score < best_score:
                    best_score = score
                    best_region = region_idx
                    best_window = (window_start, window_end)
        
        return best_region, best_window
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Decide next action based on current state."""
        # If we have pending restart overhead, wait
        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE
        
        # Calculate current state
        current_region = self.env.get_current_region()
        current_time = self.env.elapsed_seconds
        current_time_step = int(current_time / self.time_step)
        remaining_work = self._get_remaining_work()
        remaining_time = self._get_remaining_time()
        
        # If no work left, do nothing
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # If we're out of time, use on-demand to try to finish
        if remaining_time <= self.time_step + self.restart_overhead:
            if remaining_work <= remaining_time:
                return ClusterType.ON_DEMAND
            # Can't finish even with on-demand
            return ClusterType.ON_DEMAND
        
        # Calculate if we can afford to wait for spot
        min_time_to_finish = remaining_work + self.restart_overhead
        time_critical = remaining_time - min_time_to_finish < 2 * self.time_step
        
        # If time is critical, use on-demand
        if time_critical:
            return ClusterType.ON_DEMAND
        
        # Find best spot window across all regions
        best_region, best_window = self._find_best_spot_window(
            current_time_step, remaining_work
        )
        
        # If we found a good spot window and need to switch regions
        if best_region != -1 and best_region != current_region:
            # Switch if the window starts soon
            if best_window and best_window[0] - current_time_step <= 2:
                self.env.switch_region(best_region)
                # After switching, spot might not be immediately available
                # Check if spot is available in the new region at current time
                new_region_has_spot = False
                if current_time_step < len(self.traces[best_region]):
                    new_region_has_spot = self.traces[best_region][current_time_step] == 1
                
                if new_region_has_spot:
                    return ClusterType.SPOT
                else:
                    # Wait for spot to become available
                    return ClusterType.NONE
        
        # Use spot if available in current region
        if has_spot:
            return ClusterType.SPOT
        
        # If no spot available, check if we should wait or use on-demand
        # Look for next spot availability in current region
        next_spot_step = -1
        for i in range(current_time_step + 1, min(current_time_step + 5, len(self.traces[current_region]))):
            if self.traces[current_region][i] == 1:
                next_spot_step = i
                break
        
        if next_spot_step != -1:
            # Spot available soon, wait
            return ClusterType.NONE
        else:
            # No spot coming soon, use on-demand if we need to make progress
            if remaining_work > remaining_time - 3 * self.time_step:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE