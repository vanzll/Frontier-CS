from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        elapsed = self.env.elapsed_seconds
        gap = self.env.gap_seconds
        deadline = self.deadline
        overhead = self.restart_overhead
        total_duration = self.task_duration
        
        # Calculate remaining work based on completed segments
        work_done = sum(self.task_done_time)
        work_rem = total_duration - work_done
        
        if work_rem <= 1e-6:
            return ClusterType.NONE
            
        time_rem = deadline - elapsed
        
        # Calculate the threshold where we MUST use On-Demand to guarantee finishing.
        # We assume worst-case: if we don't start/continue OD now, we might face a restart overhead later.
        # We need (work_rem + overhead) seconds to finish the job via OD from a cold start.
        # We are deciding for the current step (duration `gap`). 
        # If we choose not to ensure progress this step (i.e. not OD), we must guarantee that
        # at the START of the next step, we still have enough time to finish.
        # At next step, time remaining will be (time_rem - gap).
        # So we require: (time_rem - gap) >= (work_rem + overhead).
        # Panic condition: time_rem < work_rem + overhead + gap.
        
        # Add a small margin for float precision and safety
        safety_margin = 1.0
        panic_threshold = work_rem + overhead + gap + safety_margin
        
        if time_rem < panic_threshold:
            return ClusterType.ON_DEMAND
            
        # We have enough slack to prioritize cost (Spot or Pause).
        
        if has_spot:
            # Always prefer Spot if available and safe
            return ClusterType.SPOT
        
        # Spot is unavailable.
        # If we are currently running On-Demand, we have a choice:
        # 1. Continue OD: Costs money, but avoids restart overhead later.
        # 2. Switch to NONE: Saves money now, but incurs overhead when we resume.
        # We should only pause if we have substantial slack to make the overhead worth it.
        if last_cluster_type == ClusterType.ON_DEMAND:
            slack = time_rem - panic_threshold
            # Heuristic: if slack is less than 2x overhead, keep running to avoid thrashing
            if slack < 2.0 * overhead:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE
            
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)