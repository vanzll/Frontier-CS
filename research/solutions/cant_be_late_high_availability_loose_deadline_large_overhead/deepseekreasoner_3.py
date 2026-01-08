import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self.spot_unavailable_start = None
        self.on_demand_required = False
        self.safe_margin = 0.0
        self.spot_history = []
        self.window_size = 100
        self.availability_threshold = 0.5
        self.min_spot_runs = 3
        self.spot_run_count = 0
        self.consecutive_spot_available = 0
        self.switch_to_od_threshold = 0.7
        self.last_action = ClusterType.NONE
        self.estimated_remaining = 0.0
        self.work_done = 0.0
        
    def solve(self, spec_path: str) -> "Solution":
        return self
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)
    
    def _update_work_progress(self):
        """Update work progress from task_done_time"""
        if self.task_done_time:
            self.work_done = sum(duration for _, duration in self.task_done_time)
        else:
            self.work_done = 0.0
    
    def _estimate_spot_availability(self, has_spot: bool) -> float:
        """Estimate spot availability probability"""
        self.spot_history.append(1 if has_spot else 0)
        if len(self.spot_history) > self.window_size:
            self.spot_history.pop(0)
        
        if len(self.spot_history) < 10:
            return 0.5
        
        available = sum(self.spot_history[-min(20, len(self.spot_history)):])
        return available / min(20, len(self.spot_history))
    
    def _calculate_safe_margin(self, time_left: float, work_left: float) -> float:
        """Calculate safe margin based on remaining time and work"""
        if time_left <= 0 or work_left <= 0:
            return 0.0
        
        time_ratio = time_left / (self.deadline - self.task_duration)
        work_ratio = work_left / self.task_duration
        
        base_margin = self.restart_overhead * 2.0
        urgency_factor = max(0.0, 1.0 - (time_left / (work_left * 1.5)))
        
        return base_margin * (1.0 + urgency_factor * 2.0)
    
    def _should_use_on_demand(self, has_spot: bool, time_left: float, work_left: float) -> bool:
        """Determine if we should use on-demand"""
        self._update_work_progress()
        
        # If no spot available, must use on-demand or pause
        if not has_spot:
            # If we're in a critical situation, use on-demand
            if time_left < work_left + self.restart_overhead:
                return True
            return False
        
        # Calculate critical ratio
        time_needed = work_left
        if time_left < time_needed + self.restart_overhead * 1.5:
            return True
        
        # Check if we've had too many consecutive spot runs
        if self.spot_run_count >= 10 and self.consecutive_spot_available < 5:
            availability = self._estimate_spot_availability(has_spot)
            if availability < self.switch_to_od_threshold:
                return True
        
        # If we're close to deadline, be conservative
        deadline_buffer = self.deadline - self.env.elapsed_seconds
        if deadline_buffer < work_left * 1.3:
            return True
            
        return False
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time
        self._update_work_progress()
        work_left = self.task_duration - self.work_done
        
        # Update spot availability tracking
        if has_spot:
            self.consecutive_spot_available += 1
            if self.spot_unavailable_start is not None:
                self.spot_unavailable_start = None
        else:
            self.consecutive_spot_available = 0
            if self.spot_unavailable_start is None:
                self.spot_unavailable_start = current_time
        
        # Check if we must finish immediately
        if time_left <= 0 or work_left <= 0:
            return ClusterType.NONE
        
        # If no work left, pause
        if work_left <= 0:
            return ClusterType.NONE
        
        # Calculate urgency
        urgency = (work_left / time_left) if time_left > 0 else float('inf')
        
        # Determine if we should use on-demand
        use_on_demand = self._should_use_on_demand(has_spot, time_left, work_left)
        
        if use_on_demand:
            self.spot_run_count = 0
            return ClusterType.ON_DEMAND
        
        # Use spot if available
        if has_spot:
            # Check if we should take a break from spot
            if self.spot_run_count > 0 and self.spot_run_count % 15 == 0:
                # Brief pause to reset
                self.spot_run_count = 0
                return ClusterType.NONE
            
            # Use spot
            self.spot_run_count += 1
            self.last_action = ClusterType.SPOT
            return ClusterType.SPOT
        
        # Spot not available - pause
        self.spot_run_count = 0
        return ClusterType.NONE