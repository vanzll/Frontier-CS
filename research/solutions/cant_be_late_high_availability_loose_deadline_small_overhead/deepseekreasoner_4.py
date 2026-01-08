import sys
import os
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
import math

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args=None):
        super().__init__(args)
        self.safety_margin = None
        self.consecutive_spot_failures = 0
        self.max_consecutive_failures = 5
        self.spot_availability_history = []
        self.min_spot_availability = 0.3
        self.last_spot_available = False
        self.critical_threshold = 0.15
        self.switch_to_od_threshold = 0.25
        self.remaining_work_history = []
        
    def solve(self, spec_path: str) -> "Solution":
        """Optional initialization"""
        # Read configuration if spec_path exists
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    # Simple configuration reading
                    config = f.read()
                    # Parse simple key=value pairs
                    for line in config.strip().split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            if key == 'safety_margin':
                                self.safety_margin = float(value)
                            elif key == 'max_consecutive_failures':
                                self.max_consecutive_failures = int(value)
                            elif key == 'min_spot_availability':
                                self.min_spot_availability = float(value)
            except:
                pass  # Use defaults if config can't be read
        
        # Set default safety margin if not configured
        if self.safety_margin is None:
            # Conservative margin: ensure we finish with buffer
            self.safety_margin = 0.1  # 10% of task duration as safety margin
        
        return self
    
    def _estimate_completion_time(self, use_ondemand: bool) -> float:
        """Estimate time to complete remaining work"""
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        if remaining_work <= 0:
            return 0
        
        # If using on-demand: continuous work, no interruptions
        if use_ondemand:
            return remaining_work
        
        # If using spot: account for potential interruptions
        # Estimate based on recent spot availability
        if len(self.spot_availability_history) > 0:
            recent_availability = sum(self.spot_availability_history[-20:]) / min(20, len(self.spot_availability_history))
            if recent_availability < 0.1:
                recent_availability = 0.1  # Minimum estimate
        else:
            # Initial estimate based on typical spot availability
            recent_availability = 0.5  # Conservative estimate
        
        # Account for restart overhead (average case)
        avg_restarts = remaining_work / (10 * 3600)  # Estimate restarts based on time
        overhead_time = avg_restarts * self.restart_overhead
        
        effective_work_time = remaining_work / recent_availability
        
        return effective_work_time + overhead_time
    
    def _get_critical_level(self) -> float:
        """Calculate how critical the situation is (0=plenty of time, 1=urgent)"""
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        
        if remaining_work <= 0:
            return 0.0
        
        time_elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - time_elapsed
        
        if time_remaining <= 0:
            return 1.0
        
        # Estimate time needed if we switch to on-demand now
        est_ondemand_time = remaining_work
        
        # Add safety buffer
        safety_buffer = self.safety_margin * self.task_duration
        required_time = est_ondemand_time + safety_buffer
        
        # Critical level: ratio of required time to remaining time
        # Clamped between 0 and 1
        critical = max(0.0, min(1.0, required_time / time_remaining))
        
        return critical
    
    def _should_switch_to_ondemand(self, last_cluster_type: ClusterType, has_spot: bool) -> bool:
        """Determine if we should switch to on-demand"""
        critical = self._get_critical_level()
        
        # Always switch if we're in critical state
        if critical > self.critical_threshold:
            return True
        
        # Switch if spot has been unreliable recently
        if self.consecutive_spot_failures >= self.max_consecutive_failures:
            return True
        
        # Switch if spot availability is too low
        if len(self.spot_availability_history) >= 10:
            recent_availability = sum(self.spot_availability_history[-10:]) / 10
            if recent_availability < self.min_spot_availability:
                return True
        
        # If we're not currently on spot and spot is available, stay flexible
        if last_cluster_type != ClusterType.SPOT and has_spot:
            return False
        
        # If we're on spot and it's available, continue
        if last_cluster_type == ClusterType.SPOT and has_spot:
            return False
        
        # Estimate completion times
        est_spot_time = self._estimate_completion_time(False)
        est_ondemand_time = self._estimate_completion_time(True)
        
        time_elapsed = self.env.elapsed_seconds
        time_remaining = self.deadline - time_elapsed
        
        # If spot estimate suggests we might miss deadline, switch
        if est_spot_time > time_remaining * 0.9:  # 90% of remaining time
            return True
        
        # If on-demand would be significantly faster relative to time remaining
        time_saved = est_spot_time - est_ondemand_time
        if time_saved > time_remaining * self.switch_to_od_threshold:
            return True
        
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Decision logic for each time step"""
        
        # Update spot availability history
        self.spot_availability_history.append(1 if has_spot else 0)
        if len(self.spot_availability_history) > 100:
            self.spot_availability_history.pop(0)
        
        # Update consecutive failures counter
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            self.consecutive_spot_failures += 1
        else:
            self.consecutive_spot_failures = 0
        
        # Track last spot availability
        self.last_spot_available = has_spot
        
        # Check if task is already complete
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        self.remaining_work_history.append(remaining_work)
        if len(self.remaining_work_history) > 50:
            self.remaining_work_history.pop(0)
        
        if remaining_work <= 0:
            return ClusterType.NONE
        
        # Calculate how critical our situation is
        critical = self._get_critical_level()
        
        # If we're very critical, use on-demand
        if critical > 0.8:
            return ClusterType.ON_DEMAND
        
        # Decision logic
        if self._should_switch_to_ondemand(last_cluster_type, has_spot):
            return ClusterType.ON_DEMAND
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                # If spot is not available and we're not switching to on-demand,
                # we can wait (NONE) or use on-demand based on criticality
                if critical > 0.5:
                    return ClusterType.ON_DEMAND
                else:
                    # Check if we've been waiting too long without progress
                    if len(self.remaining_work_history) >= 5:
                        recent_progress = self.remaining_work_history[-5] - remaining_work
                        if recent_progress <= 0:  # No progress in last 5 steps
                            return ClusterType.ON_DEMAND
                    return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)