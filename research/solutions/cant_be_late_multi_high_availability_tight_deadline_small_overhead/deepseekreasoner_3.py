import json
from argparse import Namespace
from typing import List
import math
import heapq

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_spot_strategy"
    
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
        
        # Parse trace files to get historical spot availability
        self.trace_files = config.get("trace_files", [])
        self.spot_availability_history = []
        for trace_file in self.trace_files:
            try:
                with open(trace_file, 'r') as f:
                    # Assuming each line has format: "time,available"
                    availability = []
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            available = int(parts[1].strip())
                            availability.append(available)
                    if availability:
                        self.spot_availability_history.append(availability)
                    else:
                        self.spot_availability_history.append([])
            except:
                self.spot_availability_history.append([])
        
        # Strategy parameters
        self.consecutive_spot_failures = [0] * self.env.get_num_regions()
        self.spot_success_streak = [0] * self.env.get_num_regions()
        self.last_region = 0
        self.ondemand_used = False
        self.emergency_mode = False
        self.current_streak = 0
        self.max_spot_attempts = 3  # Max spot attempts before switching
        self.switch_region_threshold = 2  # Consecutive spot failures to trigger region switch
        
        # Cost parameters
        self.spot_price = 0.9701 / 3600  # $ per second
        self.ondemand_price = 3.06 / 3600  # $ per second
        
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work and time
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        remaining_time = self.deadline - self.env.elapsed_seconds
        
        current_region = self.env.get_current_region()
        
        # Emergency mode: if we're running out of time
        time_ratio = remaining_time / remaining_work if remaining_work > 0 else float('inf')
        if time_ratio < 1.2:  # Less than 20% buffer
            self.emergency_mode = True
        
        # Update statistics
        if last_cluster_type == ClusterType.SPOT:
            if has_spot:
                self.spot_success_streak[current_region] += 1
                self.consecutive_spot_failures[current_region] = 0
            else:
                self.consecutive_spot_failures[current_region] += 1
                self.spot_success_streak[current_region] = 0
        else:
            self.consecutive_spot_failures[current_region] = 0
        
        # Emergency mode: use on-demand if critically low time
        if self.emergency_mode or time_ratio < 1.05:
            if last_cluster_type != ClusterType.ON_DEMAND:
                # Switch to on-demand only if not already on it
                return ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        # Try to switch region if spot is failing in current region
        if (self.consecutive_spot_failures[current_region] >= self.switch_region_threshold and
            not has_spot and
            self.env.get_num_regions() > 1):
            
            # Find region with best historical spot availability
            best_region = current_region
            best_score = -1
            for region in range(self.env.get_num_regions()):
                if region == current_region:
                    continue
                
                # Simple scoring based on recent history
                hist = self.spot_availability_history[region]
                if hist:
                    # Look at recent history
                    lookback = min(10, len(hist))
                    recent_available = sum(hist[-lookback:]) / lookback
                    
                    # Penalize region if we had many failures there
                    penalty = self.consecutive_spot_failures[region] * 0.1
                    score = recent_available - penalty
                    
                    if score > best_score:
                        best_score = score
                        best_region = region
            
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.consecutive_spot_failures[best_region] = 0
                self.last_region = best_region
                return ClusterType.NONE  # Pause for one step after switching
        
        # Main decision logic
        if has_spot:
            # Use spot if available and we have time buffer
            if time_ratio > 1.5 or self.spot_success_streak[current_region] > 2:
                # Good buffer or successful streak - use spot
                return ClusterType.SPOT
            elif time_ratio > 1.2 and self.consecutive_spot_failures[current_region] < 2:
                # Moderate buffer and not too many recent failures
                return ClusterType.SPOT
            else:
                # Conservative: use on-demand when buffer is low
                return ClusterType.ON_DEMAND
        else:
            # Spot not available
            if time_ratio > 2.0:  # Plenty of time, wait for spot
                # Try another region if multiple available
                if (self.env.get_num_regions() > 1 and 
                    self.consecutive_spot_failures[current_region] > 0):
                    next_region = (current_region + 1) % self.env.get_num_regions()
                    self.env.switch_region(next_region)
                    return ClusterType.NONE
                return ClusterType.NONE
            elif time_ratio > 1.3:
                # Some buffer, wait briefly
                if self.consecutive_spot_failures[current_region] < 2:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND
            else:
                # Low buffer, use on-demand
                return ClusterType.ON_DEMAND
    
    def _should_switch_region(self, current_region: int, has_spot: bool) -> bool:
        """Determine if we should switch regions based on historical patterns."""
        if not has_spot and self.env.get_num_regions() > 1:
            # Check if any other region has better spot availability
            hist = self.spot_availability_history
            if len(hist) > current_region and hist[current_region]:
                current_availability = sum(hist[current_region][-5:]) / 5 if len(hist[current_region]) >= 5 else 0
                
                for region in range(self.env.get_num_regions()):
                    if region == current_region:
                        continue
                    if len(hist) > region and hist[region]:
                        other_availability = sum(hist[region][-5:]) / 5 if len(hist[region]) >= 5 else 0
                        if other_availability > current_availability + 0.2:  # 20% better
                            return True
        
        return False