import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    This strategy uses a buffered linear schedule to balance cost and deadline risk.

    Core Idea:
    1.  Establish a "target deadline" that is earlier than the hard deadline. This
        creates a safety buffer.
    2.  Define a linear "glide path" of required work progress to meet this
        target deadline.
    3.  At each step, compare actual work done to the expected work on the glide path.

    Decision Logic:
    - If BEHIND schedule: We are at risk. We must make progress. Use the cheapest
      available option that guarantees progress (SPOT if available, otherwise ON_DEMAND).
    - If AHEAD of schedule: We have a time surplus. We can prioritize cost savings.
      Use SPOT if available, otherwise wait (NONE) for a cheap instance to appear,
      effectively "spending" our time surplus to save money.

    This creates a control system that dynamically switches between a progress-focused
    mode and a cost-saving mode based on its performance against a safe schedule.
    """
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy by setting up the buffered schedule.
        """
        # The total slack time is deadline - task_duration (70h - 48h = 22h).
        # We reserve a portion of this slack as a final safety buffer that
        # the strategy will not "spend" by waiting for spot instances.
        # A 5-hour buffer is chosen as a conservative safety net against
        # long periods of low spot availability and multiple preemptions.
        self.deadline_buffer = 5 * 3600  # 5 hours in seconds

        # The target deadline is our internal goal for finishing the task.
        target_deadline = self.deadline - self.deadline_buffer

        # Based on the target deadline, we calculate the required average rate of
        # progress (work-seconds per elapsed-second).
        if target_deadline > 0:
            self.progress_rate = self.task_duration / target_deadline
        else:
            # This is an edge case if the buffer is larger than the time
            # available, which shouldn't happen with the given parameters.
            # Fallback to aiming for the original deadline.
            self.progress_rate = self.task_duration / self.deadline if self.deadline > 0 else 1

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step based on the scheduling strategy.
        """
        # Calculate the total amount of work completed so far.
        work_done = sum(end - start for start, end in self.task_done_time)

        # If the task is already finished, do nothing to avoid further costs.
        if work_done >= self.task_duration:
            return ClusterType.NONE

        # Determine the expected amount of work that should have been completed
        # by the current time to stay on our linear glide path.
        expected_work_done = self.env.elapsed_seconds * self.progress_rate

        # --- Main Decision Logic ---

        # Case 1: We are behind our self-imposed schedule.
        # We must prioritize making progress to catch up.
        if work_done < expected_work_done:
            # Use SPOT if available, as it's the cheapest way to make progress.
            if has_spot:
                return ClusterType.SPOT
            # If SPOT is not available, we must use ON_DEMAND to avoid falling
            # further behind.
            else:
                return ClusterType.ON_DEMAND

        # Case 2: We are ahead of or on schedule.
        # We have a time surplus (buffer) that we can leverage for cost savings.
        else:  # work_done >= expected_work_done
            # Use SPOT if available to continue making cheap progress and extend
            # our lead.
            if has_spot:
                return ClusterType.SPOT
            # If SPOT is not available, we can afford to wait (NONE). This avoids
            # expensive ON_DEMAND costs by "spending" our accumulated time buffer.
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)