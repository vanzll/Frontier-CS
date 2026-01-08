import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    This strategy aims to minimize cost by using Spot instances opportunistically
    while ensuring the job finishes before the deadline. It operates based on a
    dynamic, adaptive schedule.

    Core principles:
    1.  **Panic Mode:** If the remaining time to the deadline is just enough to
        complete the remaining work on a guaranteed On-Demand instance, switch to
        On-Demand permanently to avoid failure.
    2.  **Opportunistic Spot:** If not in "panic mode" and Spot instances are
        available, always use them as they are the cheapest option.
    3.  **Adaptive Scheduling:** When Spot is unavailable, the decision to use
        expensive On-Demand or wait (NONE) is based on a target completion
        schedule.
        - A target finish time is set to be `deadline - safety_buffer`.
        - If the job is behind this target schedule, use On-Demand to catch up.
        - If the job is ahead of schedule, wait for Spot to become available again.
    4.  **Dynamic Safety Buffer:** The safety buffer is not fixed. It starts with
        an initial value and increases every time a Spot preemption is detected.
        This makes the strategy more cautious and switches to On-Demand earlier
        if the Spot environment proves to be unstable.
    """
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the strategy's parameters.
        """
        # --- Hyperparameters ---

        # The initial safety buffer to ensure we finish ahead of the hard deadline.
        # Total slack is 22 hours (79200s). A buffer of 8 hours is a reasonable
        # starting point, leaving 14 hours to absorb spot unavailability and
        # preemption costs.
        self.initial_safety_buffer_seconds: float = 8.0 * 3600.0

        # Multiplier for how much to increase the safety buffer per preemption.
        # A value of 2.0 means for each preemption, we add twice the restart
        # overhead to our buffer, making our schedule more aggressive to
        # compensate for the observed instability.
        self.preemption_cost_multiplier: float = 2.0

        # --- State Tracking ---
        self.spot_preemptions: int = 0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Main decision-making logic called at each timestep.
        """
        # 1. CALCULATE CURRENT STATE
        current_time = self.env.elapsed_seconds
        
        # Calculate total work completed so far.
        work_done = sum(end - start for start, end in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the job is finished, do nothing.
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_to_deadline = self.deadline - current_time

        # 2. DETECT PREEMPTION AND UPDATE STATE
        # A preemption is detected if we chose SPOT in the last step, but the
        # environment reports that a different cluster type (i.e., NONE) was run.
        if last_cluster_type == ClusterType.SPOT and self.env.cluster_type != ClusterType.SPOT:
            self.spot_preemptions += 1

        # 3. PANIC MODE: SWITCH TO ON-DEMAND IF COMPLETION IS AT RISK
        # If the time left is less than or equal to the work remaining, we have no slack.
        # We must use the guaranteed On-Demand instance to finish.
        if time_to_deadline <= work_remaining:
            return ClusterType.ON_DEMAND

        # 4. OPPORTUNISTIC SPOT USAGE
        # If Spot is available and we are not in panic mode, always use it.
        # It's the most cost-effective way to make progress.
        if has_spot:
            return ClusterType.SPOT

        # 5. ADAPTIVE SCHEDULING (WHEN SPOT IS NOT AVAILABLE)
        # Decide between ON_DEMAND (make progress) and NONE (wait for spot).
        
        # The safety buffer grows with each preemption, making the strategy
        # more risk-averse in unstable environments.
        dynamic_safety_buffer = self.initial_safety_buffer_seconds + \
                                self.spot_preemptions * self.restart_overhead * self.preemption_cost_multiplier

        # Calculate a target finish time that is earlier than the deadline.
        target_finish_time = self.deadline - dynamic_safety_buffer

        # If our adaptive schedule dictates we should have already finished,
        # or if the calculation is invalid (e.g., buffer > deadline),
        # we must use On-Demand.
        if current_time >= target_finish_time or target_finish_time <= 0:
            return ClusterType.ON_DEMAND

        # Determine the target work completion based on a linear progress model
        # towards our `target_finish_time`.
        target_rate = self.task_duration / target_finish_time
        work_target = current_time * target_rate

        # If we are behind this adaptive schedule, use On-Demand to catch up.
        # Otherwise, we are ahead and can afford to save money by waiting for Spot.
        if work_done < work_target:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> "Solution":
        """
        Instantiates the solution from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)