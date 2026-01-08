import argparse
import math
from typing import List, Tuple
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "smart_threshold"

    def __init__(self, args=None):
        super().__init__(args)
        self.config = {}
        self.spot_price = None
        self.ondemand_price = None
        self.time_step = None
        self.work_remaining = None
        self.time_remaining = None
        self.restart_timer = 0
        self.in_progress = False
        self.last_decision = ClusterType.NONE
        self.spot_unavailable_counter = 0
        self.spot_available_counter = 0
        self.consecutive_spot_failures = 0
        self.safety_margin_factor = 1.5

    def solve(self, spec_path: str) -> "Solution":
        # Default configuration - optimized for given problem parameters
        self.config.update({
            'base_spot_usage': True,
            'panic_threshold': 0.25,  # Use on-demand when < 25% of slack remains
            'spot_retry_limit': 3,  # Switch to on-demand after N consecutive spot failures
            'min_spot_confidence': 0.3,  # Minimum spot availability confidence
            'restart_buffer': 2.0,  # Extra buffer for restart overheads
            'cost_ratio_threshold': 0.4,  # Spot/on-demand cost ratio threshold
            'adaptive_safety': True
        })
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update internal state
        self.last_decision = last_cluster_type
        self.time_step = self.env.gap_seconds
        
        # Track spot availability patterns
        if has_spot:
            self.spot_available_counter += 1
            self.consecutive_spot_failures = 0
        else:
            self.spot_unavailable_counter += 1
            self.consecutive_spot_failures += 1
        
        # Calculate remaining work and time
        total_work_done = sum(self.task_done_time)
        self.work_remaining = self.task_duration - total_work_done
        current_time = self.env.elapsed_seconds
        self.time_remaining = self.deadline - current_time
        
        # Check if work is already completed
        if self.work_remaining <= 0:
            return ClusterType.NONE
        
        # Update restart timer
        if self.restart_timer > 0:
            self.restart_timer -= self.time_step
            self.restart_timer = max(0, self.restart_timer)
        
        # Calculate critical metrics
        total_time_needed = self.work_remaining
        if last_cluster_type == ClusterType.NONE:
            total_time_needed += self.restart_overhead
        
        # Emergency check: if we're critically short on time
        if self.time_remaining < total_time_needed * 1.1:
            return ClusterType.ON_DEMAND
        
        # Calculate safety margin based on spot availability history
        spot_availability_rate = 0.5
        total_spot_checks = self.spot_available_counter + self.spot_unavailable_counter
        if total_spot_checks > 0:
            spot_availability_rate = self.spot_available_counter / total_spot_checks
        
        # Adaptive safety margin based on spot availability
        safety_margin = self.restart_overhead * self.safety_margin_factor
        if spot_availability_rate < 0.3:
            safety_margin *= 2.0
        elif spot_availability_rate > 0.6:
            safety_margin *= 0.7
        
        # Check if we should panic and switch to on-demand
        time_needed_with_buffer = total_time_needed + safety_margin
        slack_ratio = self.time_remaining / time_needed_with_buffer
        
        # Too many consecutive spot failures
        if (self.consecutive_spot_failures >= self.config['spot_retry_limit'] and 
            self.work_remaining > self.time_step * 5):
            return ClusterType.ON_DEMAND
        
        # Critical time condition
        if slack_ratio < self.config['panic_threshold']:
            return ClusterType.ON_DEMAND
        
        # Very safe condition - try spot if available
        if slack_ratio > 2.0 and has_spot:
            # We have plenty of time, use spot
            if last_cluster_type != ClusterType.SPOT:
                # Starting new spot instance
                self.restart_timer = self.restart_overhead
            return ClusterType.SPOT
        
        # Calculate expected completion time with different strategies
        expected_spot_time = self.work_remaining
        if last_cluster_type != ClusterType.SPOT:
            expected_spot_time += self.restart_overhead
        
        expected_ondemand_time = self.work_remaining
        if last_cluster_type != ClusterType.ON_DEMAND:
            expected_ondemand_time += self.restart_overhead
        
        # Adjust for spot availability uncertainty
        expected_spot_time *= max(1.5, 1.0 / max(0.2, spot_availability_rate))
        
        # Decision logic
        if has_spot:
            # Check if we have enough time to tolerate potential spot interruptions
            time_buffer = self.time_remaining - expected_spot_time
            
            if time_buffer > safety_margin * 2:
                # Good buffer, use spot
                if last_cluster_type != ClusterType.SPOT:
                    self.restart_timer = self.restart_overhead
                return ClusterType.SPOT
            elif time_buffer > 0 and spot_availability_rate > 0.4:
                # Moderate buffer, use spot if availability is decent
                if last_cluster_type != ClusterType.SPOT:
                    self.restart_timer = self.restart_overhead
                return ClusterType.SPOT
            else:
                # Too tight, use on-demand
                if last_cluster_type != ClusterType.ON_DEMAND:
                    self.restart_timer = self.restart_overhead
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            if self.time_remaining < expected_ondemand_time * 1.2:
                # Running out of time, must use on-demand
                if last_cluster_type != ClusterType.ON_DEMAND:
                    self.restart_timer = self.restart_overhead
                return ClusterType.ON_DEMAND
            else:
                # Can afford to wait for spot
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)