from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Read spec_path for configuration if needed.
        Must return self.
        """
        self.my_pending_overhead = 0.0
        self.last_work_done = 0.0
        self.initial_slack = 0.0
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step. Return which cluster type to use next.
        """
        # One-time initialization of initial_slack, which is the total buffer time.
        if self.initial_slack == 0.0 and self.deadline > self.task_duration:
            self.initial_slack = self.deadline - self.task_duration

        # --- 1. Update internal state based on the outcome of the last step ---
        gap = self.env.gap_seconds
        
        # Decrease pending overhead by the time elapsed in the last step.
        self.my_pending_overhead = max(0.0, self.my_pending_overhead - gap)
        
        # Calculate progress made in the last step to detect preemptions.
        current_work_done = sum(end - start for start, end in self.task_done_time)
        progress_in_last_step = current_work_done - self.last_work_done
        
        # A preemption is detected if SPOT was used but no progress was made.
        if last_cluster_type == ClusterType.SPOT and progress_in_last_step < gap * 0.5:
            # A new restart overhead is incurred, replacing any existing one.
            self.my_pending_overhead = self.restart_overhead
        
        self.last_work_done = current_work_done
        
        # --- 2. Make a decision for the next time step ---
        work_remaining = self.task_duration - current_work_done
        
        if work_remaining <= 0:
            return ClusterType.NONE # Task is finished, do nothing to save cost.

        time_left_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # Calculate the time required to finish if we use on-demand from this point.
        # This is our worst-case fallback plan.
        time_needed_with_on_demand = self.my_pending_overhead + work_remaining
        
        # A safety buffer to avoid running too close to the deadline.
        safety_buffer = 5 * gap
        
        # A) Critical Zone: If our on-demand fallback plan takes longer than the
        # time remaining, we must use on-demand to guarantee progress.
        if time_needed_with_on_demand + safety_buffer >= time_left_to_deadline:
            return ClusterType.ON_DEMAND
            
        # B) Safe Zone: We have a time buffer (slack).
        slack = time_left_to_deadline - (time_needed_with_on_demand + safety_buffer)
        
        if has_spot:
            # Spot is available. We risk it only if our slack can absorb a potential preemption.
            # The time cost of a failed spot attempt is the lost 'gap' plus a new 'restart_overhead'.
            if slack + self.my_pending_overhead > self.restart_overhead + gap:
                return ClusterType.SPOT
            else:
                # The risk is too high; a preemption would put us in the critical zone.
                return ClusterType.ON_DEMAND
        else:
            # Spot is not available. We can either wait (NONE) or make progress (ON_DEMAND).
            # We choose to wait only if we have a very large slack buffer.
            if self.initial_slack > 0 and slack > 0.75 * self.initial_slack:
                return ClusterType.NONE
            else:
                # Otherwise, it's safer to make steady progress with on-demand.
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)