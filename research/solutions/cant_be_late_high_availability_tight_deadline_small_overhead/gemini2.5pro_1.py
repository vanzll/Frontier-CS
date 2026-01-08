import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes the strategy by pre-calculating constants and hyperparameters.
        """
        # --- Hyperparameters for the dynamic thresholds ---

        # The "panic" threshold is based on the restart overhead (RO).
        # It's calculated as: RO * (RO_BASE + RO_SCALE * progress_ratio)
        # RO_BASE: A constant buffer, ensuring we can always withstand this many
        #          preemptions.
        # RO_SCALE: A dynamic buffer that decreases as the job progresses.
        self.RO_BASE = 1.5
        self.RO_SCALE = 4.5

        # The "comfort" threshold is a gap above the panic threshold.
        # This gap also shrinks as the job progresses.
        # Gap = (initial_cushion * GAP_FRACTION) * progress_ratio
        # GAP_FRACTION: The portion of the initial slack to use for the buffer zone
        #               at the start of the job.
        self.GAP_FRACTION = 0.6

        # Pre-calculate constants for efficiency in _step
        self.initial_cushion = self.deadline - self.task_duration
        self.gap_base = self.initial_cushion * self.GAP_FRACTION

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step based on a dynamic three-zone
        cushion-based heuristic.
        """
        work_done = sum(self.task_done_time)
        work_rem = self.task_duration - work_done

        # If the job is finished, do nothing to save costs.
        if work_rem <= 1e-6:
            return ClusterType.NONE

        time_rem = self.deadline - self.env.elapsed_seconds

        # Cushion is the available slack time we have.
        cushion = time_rem - work_rem

        # If cushion is negative, we are guaranteed to be late.
        # Run on-demand to minimize lateness.
        if cushion < 0:
            return ClusterType.ON_DEMAND

        # --- Dynamic Threshold Calculation ---
        # progress_ratio represents the fraction of work remaining.
        progress_ratio = max(0.0, work_rem / self.task_duration)

        # c_thresh_od: The "panic" threshold. Below this cushion, we must use On-Demand.
        c_thresh_od = self.restart_overhead * (self.RO_BASE + self.RO_SCALE * progress_ratio)

        # c_thresh_none: The "comfort" threshold. Above this cushion, we can afford
        # to wait (NONE) if spot is unavailable.
        c_thresh_none = c_thresh_od + self.gap_base * progress_ratio

        # --- Decision Logic with Three Zones ---

        # 1. Panic Zone: cushion < c_thresh_od
        # High risk of missing deadline. Use the most reliable resource.
        if cushion < c_thresh_od:
            return ClusterType.ON_DEMAND

        # 2. Buffer Zone: c_thresh_od <= cushion < c_thresh_none
        # Some slack exists, but not enough to waste. Prioritize progress.
        # Use Spot if available, but fall back to On-Demand to preserve cushion.
        if cushion < c_thresh_none:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # 3. Comfort Zone: cushion >= c_thresh_none
        # Ample slack. Prioritize cost savings.
        # Use Spot if available, otherwise, wait (NONE).
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        """
        Instantiates the strategy from command-line arguments.
        """
        args, _ = parser.parse_known_args()
        return cls(args)