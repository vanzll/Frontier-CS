from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "AdaptiveSlackSolution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate state
        elapsed = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done
        time_left = self.deadline - elapsed
        
        # Slack: The amount of time we can afford to waste (waiting or overheads)
        # before we must run uninterrupted to meet the deadline.
        slack = time_left - remaining_work
        
        overhead = self.restart_overhead
        
        # Strategy Thresholds (in seconds)
        # 1. Critical Threshold (~9 mins): If buffer is tiny, use OD to avoid death by interruption.
        CRITICAL_THRESHOLD = overhead * 3.0
        
        # 2. Safe Switch Threshold (20 mins): If on OD, only switch to Spot if buffer is healthy.
        #    Switching incurs overhead; avoiding thrashing saves the deadline when tight.
        SAFE_SWITCH_THRESHOLD = 1200.0
        
        # 3. Wait Threshold (30 mins): If Spot is gone, wait until buffer drops to this.
        #    Then switch to OD. Balances cost saving vs safety.
        WAIT_THRESHOLD = 1800.0

        if has_spot:
            # Safety override: Use OD if strictly necessary
            if slack < CRITICAL_THRESHOLD:
                return ClusterType.ON_DEMAND
            
            # Hysteresis: Don't switch OD -> Spot if slack is tight
            if last_cluster_type == ClusterType.ON_DEMAND and slack < SAFE_SWITCH_THRESHOLD:
                return ClusterType.ON_DEMAND
            
            return ClusterType.SPOT
        else:
            # Spot unavailable
            if slack > WAIT_THRESHOLD:
                return ClusterType.NONE
            
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)