from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    """
    A strategy that dynamically adjusts its risk tolerance based on job progress.

    The core idea is to maintain a "cushion time", defined as the slack
    between the deadline and the projected completion time if we were to switch
    to 100% reliable on-demand instances from this point forward.

    cushion_time = (deadline - current_time) - work_remaining

    The strategy uses two thresholds on this cushion_time to make decisions:
    1.  A 'danger' threshold: If the cushion drops below this, we are at high
        risk of missing the deadline. We must use On-Demand to guarantee progress.
    2.  A 'comfort' threshold: If the cushion is above this, we have a large
        amount of slack. We can afford to be patient and wait for the cheaper
        Spot instances to become available, using NONE in the meantime.

    Between these two thresholds, we have a moderate amount of slack. In this
    state, we use Spot if available, but fall back to On-Demand if not, as
    waiting (NONE) would erode our cushion too quickly.

    Crucially, these thresholds are not static. They adapt to the job's progress:
    - Early in the job, we are more aggressive: the danger threshold is low,
      and the comfort threshold is high, encouraging waiting for Spot.
    - Late in the job, we are more conservative: the danger threshold is high,
      and the comfort threshold is low, prioritizing on-time completion over cost.

    This dynamic adjustment allows the strategy to capitalize on cheap Spot
    instances when there is ample time to recover from potential preemptions,
    while becoming progressively more cautious as the deadline approaches.
    """
    NAME = "dynamic_cushion"  # REQUIRED: unique identifier

    def solve(self, spec_path: str) -> "Solution":
        # At the start of the job (progress = 0%), how low can the cushion get
        # before we switch to emergency on-demand? (in hours)
        self.initial_danger_threshold_h = 1.0

        # Near the end of the job (progress = 100%), what is the minimum
        # cushion we want to maintain? (in hours)
        self.final_danger_threshold_h = 3.0

        # At the start of the job, how large must the cushion be for us to be
        # comfortable waiting for Spot (using NONE)? (in hours)
        self.initial_comfort_threshold_h = 18.0

        # Near the end of the job, what cushion is considered large enough
        # to wait for Spot? (in hours)
        self.final_comfort_threshold_h = 5.0

        # Convert hours to seconds for use in _step()
        self.initial_danger_s = self.initial_danger_threshold_h * 3600
        self.final_danger_s = self.final_danger_threshold_h * 3600
        self.initial_comfort_s = self.initial_comfort_threshold_h * 3600
        self.final_comfort_s = self.final_comfort_threshold_h * 3600

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # 1. Calculate current state and progress metrics
        work_done = sum(self.task_done_time)
        work_done = min(work_done, self.task_duration)

        work_remaining = self.task_duration - work_done
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        
        # This is the available slack time if we run on-demand from now on.
        cushion_time = time_to_deadline - work_remaining

        # 2. Calculate dynamic thresholds based on job progress
        progress = work_done / self.task_duration
        
        danger_threshold = self.initial_danger_s * (1 - progress) + \
                           self.final_danger_s * progress
        
        comfort_threshold = self.initial_comfort_s * (1 - progress) + \
                            self.final_comfort_s * progress

        # 3. Apply the three-tiered decision logic
        if cushion_time < danger_threshold:
            # DANGER ZONE: Cushion is critically low. Use On-Demand to guarantee progress.
            return ClusterType.ON_DEMAND
        else:
            if has_spot:
                # If Spot is available and we're safe, always use it.
                return ClusterType.SPOT
            else:
                # Spot is not available. Decide whether to wait or use On-Demand.
                if cushion_time > comfort_threshold:
                    # COMFORT ZONE: Cushion is large. We can afford to wait for Spot.
                    return ClusterType.NONE
                else:
                    # CAUTIOUS ZONE: Cushion is moderate. Use On-Demand to avoid losing ground.
                    return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):  # REQUIRED: For evaluator instantiation
        args, _ = parser.parse_known_args()
        return cls(args)