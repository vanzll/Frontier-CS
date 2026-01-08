from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Retrieve current environment state
        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        duration = self.task_duration
        done = sum(self.task_done_time)
        overhead = self.restart_overhead
        
        remaining_work = duration - done
        time_left = deadline - elapsed
        
        # Calculate slack: the time buffer we have before we MUST run on On-Demand to finish.
        # We subtract overhead to be conservative (assuming we might need to switch/restart).
        min_time_needed_od = remaining_work + overhead
        slack = time_left - min_time_needed_od
        
        # Define thresholds (in seconds)
        # PANIC_THRESHOLD (30 mins): If slack is below this, safety is paramount. Use On-Demand.
        PANIC_THRESHOLD = 1800 
        
        # HYSTERESIS_THRESHOLD (1 hour): If we are already on On-Demand, don't switch back to Spot
        # unless we have a healthy buffer. This prevents thrashing and paying overheads repeatedly.
        HYSTERESIS_THRESHOLD = 3600
        
        # WAIT_THRESHOLD (2 hours): If Spot is unavailable, we can afford to wait (NONE) 
        # only if we have plenty of slack. Otherwise, fallback to On-Demand.
        WAIT_THRESHOLD = 7200

        # 1. Critical Deadline Protection
        if slack < PANIC_THRESHOLD:
            return ClusterType.ON_DEMAND
            
        # 2. Logic when Spot instances are available
        if has_spot:
            # If we are currently on On-Demand, check if it's safe to switch
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack < HYSTERESIS_THRESHOLD:
                    return ClusterType.ON_DEMAND
            # Default to Spot for cost savings
            return ClusterType.SPOT
            
        # 3. Logic when Spot instances are unavailable
        else:
            # If we have a large buffer, wait (NONE) to save money
            if slack > WAIT_THRESHOLD:
                return ClusterType.NONE
            # If buffer is eroding, switch to On-Demand to guarantee progress
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)