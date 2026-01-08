import argparse

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    """
    An adaptive strategy that projects future completion time to decide between
    using On-Demand instances and waiting for Spot availability.

    The core idea is to maintain a projection of the total time required to
    finish the remaining work. This projection is based on:
    1. An online estimate of the spot instance availability probability (`p_hat`).
    2. The historical ratio of work done on spot vs. on-demand (`current_f_s`).

    The strategy dynamically adjusts its aggressiveness based on this projection:
    - If the projected time to completion (including a safety buffer for
      preemptions) exceeds the time remaining until the deadline, the strategy
      acts aggressively. It will use On-Demand instances if Spot is unavailable
      to ensure progress and bring the projection back into a safe range.
    - If the projection is well within the deadline, the strategy acts
      frugally. It will wait for Spot instances (use NONE) to save costs,
      "spending" its time slack instead of money.

    Key features:
    - **Adaptive Spot Availability Estimation**: Uses an Exponential Moving
      Average (EMA) to estimate `p_hat`, allowing it to adapt to changing
      market conditions.
    - **Dynamic Risk Management**: The decision to use expensive On-Demand
      resources is based on a forward-looking time projection, not a fixed
      slack threshold.
    - **Preemption Buffering**: A safety buffer is added to the time projection
      to account for potential time lost to restart overheads. This buffer
      is proportional to the amount of work projected to be done on Spot.
    - **Edge Case Handling**: Includes specific rules for critical situations,
      such as when the remaining slack is less than the restart overhead,
      guaranteeing a switch to On-Demand to prevent deadline failure due to a
      single preemption.
    """
    NAME = "adaptive_projection_controller"

    # --- Hyperparameters with default values ---
    # These can be overridden via command-line arguments in the evaluator.
    
    # Initial guess for spot availability probability.
    INITIAL_P_HAT: float = 0.2
    # Alpha for the Exponential Moving Average of p_hat.
    EMA_ALPHA: float = 0.01
    # Estimated overhead from preemptions per second of spot work.
    # e.g., 1 preemption (180s) every 5 hours (18000s) of spot work -> 180/18000 = 0.01
    PREEMPTION_OVERHEAD_RATE: float = 0.01
    # Safety factor for the preemption risk rule. Switch to OD if
    # slack < overhead * factor.
    PREEMPTION_RISK_FACTOR: float = 1.5

    def solve(self, spec_path: str) -> "Solution":
        """Initializes the strategy's state variables."""
        # --- State variables ---
        self.total_work_done: float = 0.0
        self.work_done_on_spot: float = 0.0
        self.p_hat: float = self.INITIAL_P_HAT
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Main decision-making logic called at each timestep."""

        # 1. Update state from the previous step's outcome.
        new_work_done = sum(self.task_done_time)
        progress = new_work_done - self.total_work_done

        if last_cluster_type == ClusterType.SPOT and progress > 0:
            self.work_done_on_spot += progress
        self.total_work_done = new_work_done

        # Update spot availability estimate (p_hat) using EMA.
        spot_sample = 1.0 if has_spot else 0.0
        self.p_hat = self.EMA_ALPHA * spot_sample + (1 - self.EMA_ALPHA) * self.p_hat

        # 2. Calculate current key metrics.
        work_rem = self.task_duration - self.total_work_done
        time_rem = self.deadline - self.env.elapsed_seconds

        # 3. Handle terminal and critical cases first.
        if work_rem <= 1e-6:
            return ClusterType.NONE  # Job is done.

        if time_rem <= 1e-6:
            # Past the deadline or no time left, must use OD to finish ASAP.
            return ClusterType.ON_DEMAND

        slack = time_rem - work_rem
        if slack <= 0:
            # No slack left, must use guaranteed OD to make progress.
            return ClusterType.ON_DEMAND

        # Preemption risk rule: If slack is less than the cost of one preemption
        # (with a safety factor), it's too risky to use Spot.
        if slack < self.restart_overhead * self.PREEMPTION_RISK_FACTOR:
            return ClusterType.ON_DEMAND

        # 4. Core controller logic: Project future time needed.
        if self.total_work_done > 1e-6:
            # Ratio of work done on spot so far.
            current_f_s = self.work_done_on_spot / self.total_work_done
        else:
            # At the beginning, optimistically assume we'll do everything on Spot.
            current_f_s = 1.0
        
        # Estimate a dynamic preemption buffer based on projected future spot work.
        work_rem_on_spot_proj = work_rem * current_f_s
        preemption_buffer = work_rem_on_spot_proj * self.PREEMPTION_OVERHEAD_RATE

        # Project the total time needed to finish the remaining work based on
        # the current spot ratio (current_f_s) and availability (p_hat).
        if self.p_hat < 1e-6:
            time_needed_proj = float('inf') if current_f_s > 1e-6 else work_rem
        else:
            # The term (1/p_hat) represents the time expansion factor for work done on Spot.
            time_needed_proj = work_rem * (1.0 + current_f_s * (1.0 / self.p_hat - 1.0))
        
        # 5. Make the decision.
        # If our projected time (with buffer) is more than the actual time
        # remaining, we are at risk of being late.
        if time_needed_proj + preemption_buffer > time_rem:
            # Projected to be late: Be aggressive. Use OD if Spot is unavailable.
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
        else:
            # Projected to be safe: Be frugal. Wait for Spot to save cost.
            return ClusterType.SPOT if has_spot else ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> "Solution":
        """
        Instantiates the strategy from command-line arguments.
        This is required by the evaluator for hyperparameter tuning.
        """
        parser.add_argument('--initial_p_hat', type=float, default=cls.INITIAL_P_HAT)
        parser.add_argument('--ema_alpha', type=float, default=cls.EMA_ALPHA)
        parser.add_argument('--preemption_overhead_rate', type=float, default=cls.PREEMPTION_OVERHEAD_RATE)
        parser.add_argument('--preemption_risk_factor', type=float, default=cls.PREEMPTION_RISK_FACTOR)

        args, _ = parser.parse_known_args()
        
        # Create an instance and override default hyperparameters with parsed args
        instance = cls(args)
        instance.INITIAL_P_HAT = args.initial_p_hat
        instance.EMA_ALPHA = args.ema_alpha
        instance.PREEMPTION_OVERHEAD_RATE = args.preemption_overhead_rate
        instance.PREEMPTION_RISK_FACTOR = args.preemption_risk_factor
        
        return instance