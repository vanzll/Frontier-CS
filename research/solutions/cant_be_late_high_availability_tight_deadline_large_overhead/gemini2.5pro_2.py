from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy's state variables and hyperparameters.
        """
        # --- Hyperparameters ---
        # Factor to determine the 'danger' threshold based on restart overhead.
        # If slack is below this threshold, we must use ON_DEMAND.
        self.danger_threshold_factor = 1.5

        # Fraction of initial total slack to determine the 'comfortable' threshold.
        # If slack is above this, we can afford to pause and wait for spot.
        self.comfort_threshold_slack_fraction = 0.25

        # --- State Tracking ---
        # Estimate of the pending restart overhead from preemptions.
        self.pending_overhead = 0.0
        
        # Flag for one-time initialization in the first _step call.
        self.is_first_step = True
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Main decision-making function called at each time step.
        Implements a three-tiered, slack-based scheduling policy.
        """
        # --- One-time Initialization ---
        if self.is_first_step:
            # DANGER_THRESHOLD: The minimum slack required to risk using a spot instance.
            # Set to be larger than a single restart overhead.
            self.DANGER_THRESHOLD = self.restart_overhead * self.danger_threshold_factor
            
            # COMFORT_THRESHOLD: The slack level above which we can afford to idle
            # and wait for spot instances to become available.
            initial_slack = self.deadline - self.task_duration
            comfort_threshold_from_slack = initial_slack * self.comfort_threshold_slack_fraction
            
            # Ensure the comfort threshold is meaningfully larger than the danger threshold.
            self.COMFORT_THRESHOLD = max(
                self.DANGER_THRESHOLD * 2.0, comfort_threshold_from_slack
            )
            self.is_first_step = False

        # --- State Update: Infer pending overhead ---
        # A preemption is inferred if we chose SPOT, but the resulting cluster state is not SPOT.
        if last_cluster_type == ClusterType.SPOT and self.env.cluster_type != ClusterType.SPOT:
             # A preemption occurred. The new overhead replaces any old one.
             self.pending_overhead = self.restart_overhead
        elif last_cluster_type in [ClusterType.SPOT, ClusterType.ON_DEMAND]:
            # Successful compute step reduces any pending overhead.
            self.pending_overhead = max(0.0, self.pending_overhead - self.env.gap_seconds)
        
        # --- Decision Logic ---
        work_done_now = sum(self.task_done_time)
        net_work_remaining = self.task_duration - work_done_now
        
        # If the task is finished, do nothing to save costs.
        if net_work_remaining <= 0:
            return ClusterType.NONE
            
        # Calculate the total effective work remaining, including overhead.
        total_work_to_be_done = net_work_remaining + self.pending_overhead
        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # Slack is the time buffer we have before we must use on-demand continuously.
        slack = time_to_deadline - total_work_to_be_done
        
        # 1. DANGER ZONE: Critical slack level.
        # Prioritize finishing on time above all else.
        if slack < self.DANGER_THRESHOLD:
            return ClusterType.ON_DEMAND
        
        # 2. OPPORTUNISTIC SPOT: If not in danger, always prefer Spot when available.
        if has_spot:
            return ClusterType.SPOT
            
        # 3. NO SPOT: Decide whether to wait (NONE) or pay for progress (ON_DEMAND).
        # This decision depends on whether we are in the comfortable or cautious zone.
        if slack >= self.COMFORT_THRESHOLD:
            # COMFORTABLE ZONE: High slack, can afford to wait for cheaper spot.
            return ClusterType.NONE
        else:
            # CAUTIOUS ZONE: Moderate slack, better to make progress with on-demand.
            return ClusterType.ON_DEMAND
            
    @classmethod
    def _from_args(cls, parser):
        """
        Required classmethod to instantiate the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)