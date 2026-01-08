import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "adaptive_threshold_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy's parameters. Called once before evaluation.
        """
        self.ema_alpha: float = 0.005
        self.estimated_availability: float = 0.5
        
        p1, k1 = 0.1, 1.5
        p2, k2 = 0.8, 4.0

        # Create a linear model for our "patience" threshold multiplier `k`
        # based on the estimated spot availability `p`.
        # k = a*p + b
        if p2 - p1 != 0:
            self.k_coeff_a = (k2 - k1) / (p2 - p1)
        else:
            self.k_coeff_a = 0.0
        self.k_coeff_b = k1 - self.k_coeff_a * p1

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step on which cluster type to use.
        """
        # Update our estimate of spot availability using an Exponential Moving Average.
        current_spot_observation = 1.0 if has_spot else 0.0
        self.estimated_availability = (self.ema_alpha * current_spot_observation +
                                       (1 - self.ema_alpha) * self.estimated_availability)

        # Calculate the amount of work remaining.
        work_done = sum(seg[1] - seg[0] for seg in self.task_done_time)
        work_remaining = self.task_duration - work_done

        # If the task is finished, no need to run any instances.
        if work_remaining <= 0:
            return ClusterType.NONE

        # Determine the "point of no return": the latest time we must switch
        # to on-demand to guarantee finishing by the deadline.
        latest_start_for_od_phase = self.deadline - work_remaining

        # If we are past the point of no return, we must use on-demand.
        if self.env.elapsed_seconds >= latest_start_for_od_phase:
            return ClusterType.ON_DEMAND

        # If we have a time buffer (slack), we can be more strategic.
        if has_spot:
            # Spot is available and is the most cost-effective option.
            return ClusterType.SPOT
        else:
            # Spot is not available. Decide between waiting (NONE) or using
            # on-demand to make progress.
            slack = (self.deadline - self.env.elapsed_seconds) - work_remaining
            
            # Compare slack against a dynamic threshold that depends on our
            # estimate of spot availability.
            k = self.k_coeff_a * self.estimated_availability + self.k_coeff_b
            slack_threshold = k * self.restart_overhead

            if slack > slack_threshold:
                # We have sufficient slack to risk waiting for a spot instance.
                return ClusterType.NONE
            else:
                # Slack is too low to risk waiting. Use on-demand to make progress.
                return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required method for the evaluator to instantiate the strategy.
        """
        args, _ = parser.parse_known_args()
        return cls(args)