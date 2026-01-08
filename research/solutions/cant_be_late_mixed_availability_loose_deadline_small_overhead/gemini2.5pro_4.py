import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_slack_strategy"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy's state and hyperparameters.
        """
        # --- State for Online Learning ---
        self.steps_observed = 0
        self.spot_available_count = 0
        # Start with a neutral assumption about spot availability (~60%)
        self.p_spot_availability = 0.6

        # --- Tunable Hyperparameters ---
        # Number of steps to observe before adapting the strategy
        self.LEARNING_STEPS = 200
        # Threshold to distinguish between high/low availability scenarios
        self.HIGH_AVAIL_THRESHOLD = 0.5
        # Ratio of remaining work to keep as a slack buffer.
        # A smaller buffer is used in high-availability environments.
        self.HIGH_AVAIL_RATIO = 0.20
        # A larger buffer is used in low-availability environments.
        self.LOW_AVAIL_RATIO = 0.40
        # The ratio for the initial learning phase
        self.INITIAL_RATIO = (self.HIGH_AVAIL_RATIO + self.LOW_AVAIL_RATIO) / 2
        # A static buffer to avoid cutting it too close, in terms of restart overheads.
        self.STATIC_BUFFER_IN_OVERHEADS = 10.0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decision logic for each time step.
        """
        # --- 1. Online Learning: Update spot availability estimate ---
        self.steps_observed += 1
        if has_spot:
            self.spot_available_count += 1
        
        # Periodically update our belief about the environment's spot availability
        if self.steps_observed > 0 and self.steps_observed % 50 == 0:
            self.p_spot_availability = self.spot_available_count / self.steps_observed

        # --- 2. Calculate Progress and Key Metrics ---
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is done, switch to NONE to minimize cost.
        if work_remaining <= 0:
            return ClusterType.NONE

        time_remaining_to_deadline = self.deadline - self.env.elapsed_seconds

        # --- 3. Panic Mode: Check if we are out of slack ---
        # If the time left is less than or equal to the work left, we must
        # use a guaranteed resource (On-Demand) to finish by the deadline.
        if time_remaining_to_deadline <= work_remaining:
            return ClusterType.ON_DEMAND

        # --- 4. Main Decision Logic ---
        # If not in panic mode, there's some slack.
        # The cheapest way to make progress is with a Spot instance.
        if has_spot:
            return ClusterType.SPOT

        # If Spot is not available, we decide between expensive progress (ON_DEMAND)
        # or waiting (NONE). The decision is based on our available slack compared
        # to a dynamically calculated "comfort zone" threshold.
        
        slack = time_remaining_to_deadline - work_remaining

        # Determine the appropriate buffer ratio based on our learned availability
        if self.steps_observed < self.LEARNING_STEPS:
            slack_buffer_ratio = self.INITIAL_RATIO
        elif self.p_spot_availability >= self.HIGH_AVAIL_THRESHOLD:
            # High availability: we can be more aggressive (smaller buffer)
            slack_buffer_ratio = self.HIGH_AVAIL_RATIO
        else:
            # Low availability: we must be more conservative (larger buffer)
            slack_buffer_ratio = self.LOW_AVAIL_RATIO

        # The slack threshold combines a dynamic part (proportional to remaining work)
        # and a static part (to handle potential preemption chains).
        static_buffer = self.STATIC_BUFFER_IN_OVERHEADS * self.restart_overhead
        dynamic_buffer = work_remaining * slack_buffer_ratio
        slack_threshold = dynamic_buffer + static_buffer

        if slack < slack_threshold:
            # Our slack has fallen below the desired threshold.
            # Use ON_DEMAND to make guaranteed progress and build slack.
            return ClusterType.ON_DEMAND
        else:
            # We have a comfortable amount of slack.
            # We can afford to wait for a cheaper Spot instance to become available.
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)