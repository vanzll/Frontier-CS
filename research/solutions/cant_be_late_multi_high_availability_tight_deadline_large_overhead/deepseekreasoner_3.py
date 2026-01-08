import json
from argparse import Namespace
from typing import List, Dict, Tuple
import numpy as np
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

class Solution(MultiRegionStrategy):
    NAME = "adaptive_threshold_strategy"

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
        
        self.spot_price = 0.9701  # $/hr
        self.ondemand_price = 3.06  # $/hr
        self.gap_hours = self.env.gap_seconds / 3600.0
        
        self.region_stats = {}
        self.current_region = 0
        self.consecutive_spot_failures = 0
        self.max_spot_failures = 3
        self.safety_margin = 0.1  # 10% safety margin
        self.min_spot_confidence = 0.7
        
        return self

    def _compute_required_progress(self) -> Tuple[float, float, float]:
        elapsed = self.env.elapsed_seconds
        remaining_time = self.deadline - elapsed
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # Account for pending restart overhead
        effective_time = remaining_time - self.remaining_restart_overhead
        if effective_time < 0:
            effective_time = 0
            
        # Required progress rate (work per second)
        required_rate = work_remaining / effective_time if effective_time > 0 else float('inf')
        
        # Conservative estimate with safety margin
        conservative_required_rate = work_remaining / (effective_time * (1 - self.safety_margin)) if effective_time > 0 else float('inf')
        
        return work_remaining, required_rate, conservative_required_rate

    def _estimate_spot_reliability(self, region_idx: int, has_spot: bool) -> float:
        if region_idx not in self.region_stats:
            self.region_stats[region_idx] = {'total': 0, 'available': 0}
        
        self.region_stats[region_idx]['total'] += 1
        if has_spot:
            self.region_stats[region_idx]['available'] += 1
        
        total = self.region_stats[region_idx]['total']
        available = self.region_stats[region_idx]['available']
        
        if total < 5:
            return 0.5  # Default confidence for early steps
        return available / total

    def _find_best_alternative_region(self, current_idx: int, has_spot: bool) -> int:
        num_regions = self.env.get_num_regions()
        best_region = current_idx
        best_score = -1
        
        for i in range(num_regions):
            if i == current_idx:
                continue
                
            # Calculate score based on estimated reliability
            reliability = self.region_stats.get(i, {'total': 0, 'available': 0})
            total = reliability['total']
            available = reliability['available']
            
            if total == 0:
                score = 0.5  # Neutral score for unknown regions
            else:
                score = available / total
                
            # Prefer regions we have data for
            if total > 10:
                score *= 1.2
                
            if score > best_score:
                best_score = score
                best_region = i
        
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate progress metrics
        work_remaining, required_rate, conservative_rate = self._compute_required_progress()
        
        # Check if we're done
        if work_remaining <= 0:
            return ClusterType.NONE
        
        # Get current region and update stats
        self.current_region = self.env.get_current_region()
        spot_reliability = self._estimate_spot_reliability(self.current_region, has_spot)
        
        # Calculate time pressure
        elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - elapsed
        time_per_work_unit = time_remaining / work_remaining if work_remaining > 0 else float('inf')
        
        # Calculate work progress per step
        effective_work_per_spot_step = self.gap_hours
        effective_work_per_ondemand_step = self.gap_hours
        
        # Adjust for potential restart overhead
        if last_cluster_type != ClusterType.SPOT and has_spot:
            effective_work_per_spot_step -= self.restart_overhead / 3600.0
        if last_cluster_type != ClusterType.ON_DEMAND:
            effective_work_per_ondemand_step -= self.restart_overhead / 3600.0
        
        # Determine strategy based on time pressure and spot reliability
        if time_per_work_unit < 1.5:  # High time pressure
            # Use on-demand when under high time pressure
            if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead == 0:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.ON_DEMAND
        
        elif time_per_work_unit < 2.5:  # Medium time pressure
            # Use spot if reliable, otherwise on-demand
            if has_spot and spot_reliability >= self.min_spot_confidence:
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                elif self.remaining_restart_overhead == 0:
                    return ClusterType.SPOT
                else:
                    return ClusterType.NONE
            else:
                # Consider switching to a better region
                if not has_spot or spot_reliability < 0.5:
                    best_alt = self._find_best_alternative_region(self.current_region, has_spot)
                    if best_alt != self.current_region and self.remaining_restart_overhead == 0:
                        self.env.switch_region(best_alt)
                        return ClusterType.NONE
                
                if last_cluster_type != ClusterType.ON_DEMAND and self.remaining_restart_overhead == 0:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.ON_DEMAND
        
        else:  # Low time pressure - can be aggressive with spot
            if has_spot:
                if last_cluster_type == ClusterType.SPOT:
                    return ClusterType.SPOT
                elif self.remaining_restart_overhead == 0:
                    return ClusterType.SPOT
                else:
                    return ClusterType.NONE
            else:
                # Spot not available - try another region
                self.consecutive_spot_failures += 1
                
                if self.consecutive_spot_failures >= self.max_spot_failures:
                    best_alt = self._find_best_alternative_region(self.current_region, has_spot)
                    if best_alt != self.current_region and self.remaining_restart_overhead == 0:
                        self.env.switch_region(best_alt)
                        self.consecutive_spot_failures = 0
                        return ClusterType.NONE
                
                # Wait for spot to become available or use none
                return ClusterType.NONE