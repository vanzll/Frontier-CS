import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "AdaptiveSafetyBuffer"

    def __init__(self, args):
        super().__init__(args)
        self.spot_session_start_time = None
        self.spot_run_durations = []
        
        # Initial guess for average spot uptime in seconds. This value is
        # updated dynamically based on observed preemption patterns.
        # A moderately pessimistic guess of 2.5 hours is used to start.
        self.t_spot_avg_estimate = 2.5 * 3600
        
        # A constant safety buffer to add on top of the dynamic one,
        # expressed as a factor of the restart overhead.
        self.fixed_safety_buffer_factor = 1.5

    def solve(self, spec_path: str) -> "Solution":
        """
        Optional initialization. Called once before evaluation.
        Must return self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Called at each time step to decide which cluster type to use next.
        """
        current_time = self.env.elapsed_seconds
        
        # 1. Learning: Update our estimate of Spot instance stability.
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            if self.spot_session_start_time is not None:
                run_duration = current_time - self.spot_session_start_time
                if run_duration > 0:
                    self.spot_run_durations.append(run_duration)
                    # Update the average spot instance lifetime using a simple mean.
                    self.t_spot_avg_estimate = sum(self.spot_run_durations) / len(self.spot_run_durations)
            
            self.spot_session_start_time = None

        # 2. State Calculation: Assess current job progress.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        # 3. Decision Logic: Choose the next cluster type.
        current_slack = self.deadline - current_time - work_remaining
        
        if current_slack < 0:
            return ClusterType.ON_DEMAND

        # Estimate future time loss due to potential spot preemptions.
        num_future_preemptions = work_remaining / (self.t_spot_avg_estimate + 1e-9)
        dynamic_safety_buffer = num_future_preemptions * self.restart_overhead
            
        fixed_safety_buffer = self.fixed_safety_buffer_factor * self.restart_overhead
        total_safety_buffer = dynamic_safety_buffer + fixed_safety_buffer

        if current_slack <= total_safety_buffer:
            decision = ClusterType.ON_DEMAND
        else:
            if has_spot:
                decision = ClusterType.SPOT
            else:
                decision = ClusterType.NONE
        
        # 4. State Update: Prepare for the next step.
        if decision == ClusterType.SPOT:
            if self.spot_session_start_time is None:
                self.spot_session_start_time = current_time
        elif last_cluster_type == ClusterType.SPOT:
             self.spot_session_start_time = None
        
        # Final safeguard: The environment forbids choosing SPOT when it's unavailable.
        if not has_spot and decision == ClusterType.SPOT:
            return ClusterType.ON_DEMAND

        return decision

    @classmethod
    def _from_args(cls, parser):
        """
        Instantiates the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)