import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"
    
    def __init__(self, args):
        super().__init__(args)
        self._init_state = False
        
    def solve(self, spec_path: str) -> "Solution":
        self._init_state = True
        return self
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not hasattr(self, '_initialized'):
            self._initialize()
        
        # Calculate remaining work and time
        total_done = sum(self.task_done_time)
        remaining_work = self.task_duration - total_done
        time_remaining = self.deadline - self.env.elapsed_seconds
        
        # If no work left or time expired, do nothing
        if remaining_work <= 0 or time_remaining <= 0:
            return ClusterType.NONE
        
        # Calculate safe threshold - conservative approach
        # Reserve last 2 hours for on-demand if we're behind schedule
        work_rate = 1.0  # Work done per second when running
        
        # Estimate time needed with current progress
        est_completion = self.env.elapsed_seconds + remaining_work / work_rate
        
        # Calculate how much slack we have
        slack = self.deadline - est_completion
        
        # Check if we're in the final phase (last 3 hours) or behind schedule
        final_phase = time_remaining < 3 * 3600  # Last 3 hours
        behind_schedule = slack < 0.5 * 3600  # Less than 30 minutes slack
        
        # Emergency mode: we must finish, use on-demand
        if time_remaining < remaining_work * 1.1:  # Less than 10% extra time
            if has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND
        
        # Normal decision logic
        if has_spot:
            # Use spot if we have good slack
            if slack > 2 * 3600:  # More than 2 hours slack
                return ClusterType.SPOT
            elif slack > 1 * 3600:  # Moderate slack
                # Alternate between spot and on-demand
                if self.env.elapsed_seconds % (30 * 60) < 15 * 60:  # 50/50 mix
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            else:  # Low slack, prefer on-demand
                return ClusterType.ON_DEMAND
        else:
            # No spot available
            if behind_schedule or final_phase:
                return ClusterType.ON_DEMAND
            else:
                # Wait for spot to become available if we have time
                return ClusterType.NONE
    
    def _initialize(self):
        """Initialize internal state on first call"""
        self._initialized = True
        self._last_decision = ClusterType.NONE
        
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)