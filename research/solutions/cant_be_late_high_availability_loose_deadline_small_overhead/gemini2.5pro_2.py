import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    This strategy operates on the principle of maintaining a "time slack".
    Slack is defined as the extra time available before the deadline, after
    accounting for the time required to finish the remaining work using
    guaranteed on-demand instances.

    slack = (deadline - current_time) - work_remaining

    The decision at each step is based on this slack value, compared against
    dynamically calculated thresholds. These thresholds represent the potential
    time cost of future risks (like spot preemption) and waiting periods.

    Key principles:
    1.  Dynamic Buffers: The safety buffers for taking risks are not fixed.
        They are proportional to the amount of work remaining. When more work
        is left, there's a higher chance of encountering future preemptions,
        so the strategy requires a larger slack buffer to take risks.

    2.  Adaptive Risk-Taking: The strategy monitors for spot preemptions.
        If a preemption is detected (i.e., being on a spot instance resulted
        in no work progress), the strategy becomes more conservative by
        increasing its safety buffers. This helps it adapt to unstable spot
        markets.

    3.  Decision Logic:
        - If Spot is available: Use SPOT only if the current slack is large
          enough to absorb a potential preemption. The size of this "preemption
          buffer" is dynamic. Otherwise, use the safer ON_DEMAND option.
        - If Spot is unavailable: Choose between waiting (NONE) or making
          progress with ON_DEMAND. Waiting is cheap but consumes slack. The
          strategy will only wait if the slack is comfortably large. The
          "wait buffer" is significantly larger than the preemption buffer,
          reflecting the higher cost of inaction. Otherwise, it uses ON_DEMAND
          to preserve the remaining slack.
    
    This approach aims to aggressively use cheap resources (Spot and None)
    when the deadline is far away, and smoothly transition to safer, more
    expensive options (On-Demand) as the deadline approaches or as market
    instability is detected, ensuring the job finishes on time while minimizing
    cost.
    """
    NAME = "dynamic_adaptive_slack"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initializes hyperparameters and state variables for the strategy.
        """
        # --- Hyperparameters for Dynamic Thresholds ---

        # Base safety factor for using SPOT. We need slack > N * overhead.
        self.N_base = 1.0
        # Divisor for scaling N with work_remaining. A smaller value makes the
        # strategy more cautious. N increases by 1 for every 8 hours of work.
        self.N_wr_div = 8 * 3600.0

        # Base safety factor for waiting (NONE). We need slack > M * overhead.
        self.M_base = 4.0
        # Divisor for scaling M. M increases by 1 for every 2 hours of work.
        self.M_wr_div = 2 * 3600.0

        # --- Adaptive Behavior Parameters ---
        
        # How much to increase the adaptation factor upon detecting a preemption.
        self.adaptation_increment = 0.2

        # --- State Variables ---
        
        # Tracks the total work done at the previous timestep.
        self.last_work_done = 0.0
        # A factor that increases with preemptions, making the strategy more cautious.
        self.adaptation_factor = 1.0
        
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a decision at each time step based on the calculated time slack.
        """
        # 1. Calculate current state
        time_current = self.env.elapsed_seconds
        current_work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - current_work_done

        # If the job is finished, do nothing.
        if work_remaining <= 1e-6:
            self.last_work_done = current_work_done
            return ClusterType.NONE

        time_to_deadline = self.deadline - time_current
        
        # Slack is the core metric: the time buffer we have if we were to
        # finish the rest of the job purely on-demand.
        slack = time_to_deadline - work_remaining

        # 2. Adapt based on previous step's outcome
        work_progress = current_work_done - self.last_work_done
        # A preemption is inferred if we were on SPOT but made negligible progress.
        if last_cluster_type == ClusterType.SPOT and work_progress < self.env.gap_seconds * 0.5:
            self.adaptation_factor += self.adaptation_increment

        # 3. Hard safety check: if slack is non-positive, we are behind schedule.
        # We must use ON_DEMAND to have any chance of finishing.
        if slack <= 0:
            self.last_work_done = current_work_done
            return ClusterType.ON_DEMAND

        # 4. Calculate dynamic thresholds for decision-making
        n_dynamic = self.N_base + work_remaining / self.N_wr_div
        m_dynamic = self.M_base + work_remaining / self.M_wr_div

        # The buffers are scaled by the adaptation_factor to react to preemptions.
        spot_risk_buffer = self.adaptation_factor * n_dynamic * self.restart_overhead
        wait_slack_threshold = self.adaptation_factor * m_dynamic * self.restart_overhead

        # 5. Core Decision Logic
        decision = ClusterType.NONE
        if has_spot:
            # Spot is available. Use it if we have enough slack to absorb a
            # potential preemption, otherwise play it safe with on-demand.
            if slack > spot_risk_buffer:
                decision = ClusterType.SPOT
            else:
                decision = ClusterType.ON_DEMAND
        else:
            # Spot is not available. Wait (NONE) only if we have a very large
            # slack buffer. Otherwise, use on-demand to avoid losing slack.
            if slack > wait_slack_threshold:
                decision = ClusterType.NONE
            else:
                decision = ClusterType.ON_DEMAND
        
        # 6. Update state for the next step and return decision
        self.last_work_done = current_work_done
        return decision

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        """
        Required classmethod for evaluator instantiation.
        """
        args, _ = parser.parse_known_args()
        return cls(args)