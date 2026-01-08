import math
from argparse import ArgumentParser

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "safety_cushion"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy. This method is called once before the
        simulation starts.
        """
        # A configurable factor to adjust the patience threshold.
        # A value of 1.0 is a robust baseline, meaning we keep enough
        # slack to absorb one full restart_overhead penalty.
        self.patience_factor = 1.0
        if (hasattr(self, 'args') and
                hasattr(self.args, 'patience_factor') and
                self.args.patience_factor is not None):
            self.patience_factor = self.args.patience_factor

        # Cache for memoizing work_done calculation, as task_done_time
        # can grow, although it's unlikely to be a bottleneck.
        self._work_done_cache = 0.0
        self._work_done_segs_count = 0
        return self

    def _get_work_done(self) -> float:
        """
        Calculates the total work done so far, using a cache to avoid
        re-computation.
        """
        num_segs = len(self.task_done_time)
        if num_segs > self._work_done_segs_count:
            # A new work segment was added (e.g., after a preemption),
            # so we need to re-calculate the total work done.
            self._work_done_cache = sum(
                end - start for start, end in self.task_done_time)
            self._work_done_segs_count = num_segs
        return self._work_done_cache

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Implements the decision logic for each time step.

        The strategy is based on maintaining a "safety cushion" of time.
        It ensures that at any point, there is enough time remaining to
        complete the job using reliable On-Demand instances, even in a
        worst-case scenario of an immediate Spot preemption.

        The logic hierarchy is as follows:
        1.  Check for Terminal State: If the job is done or the deadline
            is missed, do nothing (NONE).
        2.  Check for Danger Zone: If the time required to finish with one
            potential preemption (`work_remaining + restart_overhead`) exceeds
            the time left until the deadline, we MUST use On-Demand to
            guarantee completion.
        3.  Safe Zone (Spot Available): If not in the danger zone and Spot
            instances are available, use them to minimize cost.
        4.  Safe Zone (Spot Unavailable): If Spot is unavailable, we decide
            between waiting (NONE) and using On-Demand. This is based on a
            "patience threshold". If our safety slack (time beyond the
            worst-case requirement) is above this threshold, we wait.
            Otherwise, we use On-Demand to make progress and preserve our
            dwindling slack.
        """
        # --- 1. Calculate current state ---
        current_time = self.env.elapsed_seconds
        work_done = self._get_work_done()
        work_remaining = self.task_duration - work_done

        # --- 2. Handle terminal states ---
        if work_remaining <= 1e-9:
            return ClusterType.NONE

        time_to_deadline = self.deadline - current_time
        if time_to_deadline <= 0:
            return ClusterType.NONE

        # --- 3. Define and check against the "danger zone" ---
        # This is the time needed to finish if we are preempted right now.
        worst_case_finish_duration = work_remaining + self.restart_overhead

        if worst_case_finish_duration >= time_to_deadline:
            return ClusterType.ON_DEMAND

        # --- 4. Handle the "safe zone" ---
        if has_spot:
            return ClusterType.SPOT

        # Spot is not available. Decide whether to wait or use On-Demand.
        # This is based on our "safety_slack", which is the buffer we have
        # beyond the worst-case scenario.
        safety_slack = time_to_deadline - worst_case_finish_duration

        # The patience threshold is how much safety_slack we're willing to burn
        # waiting for Spot. A threshold of `1.0 * restart_overhead` ensures
        # we can always absorb one preemption without entering the danger zone.
        patience_threshold = self.restart_overhead * self.patience_factor

        if safety_slack > patience_threshold:
            # Plenty of slack, it's cost-effective to wait.
            return ClusterType.NONE
        else:
            # Slack is low, use On-Demand to make progress and preserve it.
            return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser: ArgumentParser):
        """
        Required method for the evaluator to instantiate the strategy.
        Allows for adding command-line arguments for hyperparameter tuning.
        """
        parser.add_argument(
            '--patience-factor',
            type=float,
            default=1.0,
            help='Factor to scale the restart_overhead for the patience threshold.'
        )
        args, _ = parser.parse_known_args()
        return cls(args)