import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    This strategy employs a three-zone buffer system to make decisions at each
    timestep. The central idea is to calculate a "buffer" which represents the
    remaining slack time assuming the rest of the task is completed using
    reliable On-Demand instances.

    Buffer = (Time Remaining until Deadline) - (Work Remaining on Task)

    Based on the size of this buffer, the strategy operates in one of three modes:

    1.  Green Zone (Large Buffer): When there is a substantial amount of slack,
        the strategy prioritizes cost savings. It will use cheap SPOT instances
        when available and will prefer to wait (NONE) rather than pay for
        expensive ON_DEMAND instances if SPOT is unavailable.

    2.  Yellow Zone (Medium Buffer): As the slack time diminishes, the strategy
        becomes more cautious. It can no longer afford to wait and risk falling
        behind. It still prefers SPOT instances for their low cost, but will
        switch to ON_DEMAND if SPOT is not available, ensuring continuous
        progress towards completion.

    3.  Red Zone (Critically Low Buffer): When the buffer is very small, there is
        a high risk of missing the deadline. In this "panic mode," the strategy's
        sole priority is to finish the task on time. It will exclusively use
        ON_DEMAND instances for their guaranteed availability and progress.

    The thresholds defining these zones are determined during initialization. The
    Red Zone threshold is dynamically calculated based on the restart overhead to
    provide a cushion against potential preemptions. The Yellow Zone threshold
    is a fixed value tuned to balance risk and cost for the given problem
    parameters, particularly the low Spot availability. A caching mechanism is
    used to efficiently track the work completed.
    """
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        # Factor for the safety buffer, defined in multiples of restart_overhead.
        # This creates a "panic" zone to switch to On-Demand when the slack
        # is low, providing a cushion against a few final preemptions.
        safety_buffer_factor = 7.5

        # Threshold to switch from waiting (NONE) to using On-Demand when Spot
        # is unavailable. This defines the boundary between the Green and Yellow
        # zones. Given the initial 22h of slack, this value allows for a period
        # of aggressive waiting before becoming more conservative.
        wait_buffer_hours = 16.0

        self.safety_buffer_s = safety_buffer_factor * self.restart_overhead
        self.wait_buffer_s = wait_buffer_hours * 3600

        # Cache for work_done calculation to ensure O(1) complexity per step.
        self._work_done_s = 0.0
        self._last_task_done_len = 0

        return self

    def _get_work_done_seconds(self) -> float:
        """Efficiently calculates total work done using a cache."""
        # The task_done_time list is append-only, so we only need to process
        # new segments.
        if len(self.task_done_time) > self._last_task_done_len:
            new_segments = self.task_done_time[self._last_task_done_len:]
            self._work_done_s += sum(end - start for start, end in new_segments)
            self._last_task_done_len = len(self.task_done_time)
        return self._work_done_s

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        work_done_s = self._get_work_done_seconds()
        work_remaining_s = self.task_duration - work_done_s

        # If the task is complete, do nothing to minimize cost.
        if work_remaining_s <= 1e-9:
            return ClusterType.NONE

        time_to_deadline_s = self.deadline - self.env.elapsed_seconds

        # The core metric: slack time if we were to finish on On-Demand from now.
        buffer_s = time_to_deadline_s - work_remaining_s

        # --- Zone-based Decision Logic ---

        # Red Zone ("Panic Mode"): Buffer is critically low. Use On-Demand.
        if buffer_s <= self.safety_buffer_s:
            return ClusterType.ON_DEMAND

        # Yellow Zone ("Cautious Mode"): Buffer is shrinking. Make progress.
        elif buffer_s <= self.wait_buffer_s:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # Green Zone ("Aggressive Cost-Saving"): Buffer is large. Wait for Spot.
        else:  # buffer_s > self.wait_buffer_s
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """Required method for evaluator instantiation."""
        args, _ = parser.parse_known_args()
        return cls(args)