from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "OptimalSlackStrategy"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Calculate remaining work and time
        # task_done_time contains segments of work completed
        done = sum(self.task_done_time)
        work_rem = self.task_duration - done
        
        # If task is finished, do nothing
        if work_rem <= 0:
            return ClusterType.NONE

        elapsed = self.env.elapsed_seconds
        time_rem = self.deadline - elapsed
        
        # State variables
        R = self.restart_overhead
        gap = self.env.gap_seconds
        
        # Define a safety buffer.
        # We need to account for the discrete time steps (gap) and provide a safety margin
        # against small delays or boundary conditions.
        # 900 seconds (15 mins) or 10 steps, whichever is larger.
        buffer = max(10.0 * gap, 900.0)
        
        # Slack is the time budget we can afford to waste (wait or incur overhead)
        slack = time_rem - work_rem
        
        # Panic Threshold:
        # If we switch to OD now (or soon), we might pay 'R' overhead (if not already on OD).
        # We need to ensure that after paying 'R', we still have a positive buffer.
        # Threshold = R + buffer.
        panic_threshold = R + buffer
        
        # --- Logic ---
        
        # 1. Safety Check (Panic Mode)
        # If slack is critically low, we must use On-Demand to guarantee completion.
        # We cannot risk waiting for Spot or dealing with Spot interruptions.
        if slack < panic_threshold:
            return ClusterType.ON_DEMAND

        # 2. Cost Optimization
        # We have sufficient slack to be flexible.
        if has_spot:
            # Spot is available. Ideally, we want to use it.
            
            # However, if we are currently on ON_DEMAND, switching to SPOT incurs an immediate overhead R.
            # We should only switch if we have "excess" slack.
            # We want to ensure that after paying R, we still have enough buffer to potentially
            # switch back to OD later if Spot fails (costing another R) without hitting panic immediately.
            # So we look for slack > 2*R + buffer.
            if last_cluster_type == ClusterType.ON_DEMAND:
                if slack > 2 * R + buffer:
                    return ClusterType.SPOT
                else:
                    # Not enough excess slack to justify the switch cost/risk
                    return ClusterType.ON_DEMAND
            else:
                # Currently on SPOT or NONE.
                # We have enough slack (checked by panic_threshold), so proceed with Spot.
                return ClusterType.SPOT
        else:
            # Spot is unavailable.
            # Since we are not in panic mode (slack >= panic_threshold), we prefer to wait (NONE)
            # rather than burning money on OD. We consume slack hoping Spot becomes available.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)