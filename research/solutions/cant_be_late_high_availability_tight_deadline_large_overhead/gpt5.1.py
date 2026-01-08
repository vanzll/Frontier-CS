from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Internal state initialization
        self._initialized_internal_state = False
        self._progress_done = 0.0
        self._last_task_done_len = 0
        self._force_on_demand = False
        self._allowed_waste = 0.0
        self._catchup_threshold = 0.0
        self._total_slack = 0.0

    def solve(self, spec_path: str) -> "Solution":
        # Optional preprocessing: unused in this strategy.
        return self

    def _initialize_internal_state(self):
        # Compute scheduling-related constants once we have access to env.
        self._total_slack = max(0.0, self.deadline - self.task_duration)
        gap = getattr(self.env, "gap_seconds", 0.0)
        margin = gap + getattr(self, "restart_overhead", 0.0)

        if self._total_slack <= 0.0 or margin >= self._total_slack:
            # No usable slack: disallow any waste from spot/idling.
            self._allowed_waste = 0.0
            self._catchup_threshold = 0.0
        else:
            # Allowed waste is bounded both by a fraction of total slack and by
            # total_slack - margin to keep a safety buffer for the last step
            # and a final restart overhead.
            allowed_by_slack = self._total_slack - margin
            allowed_by_fraction = 0.8 * self._total_slack
            self._allowed_waste = min(allowed_by_slack, allowed_by_fraction)
            if self._allowed_waste < 0.0:
                self._allowed_waste = 0.0

            # Threshold at which we start using on-demand when spot is absent.
            self._catchup_threshold = 0.3 * self._total_slack
            if self._catchup_threshold >= self._allowed_waste:
                # Ensure catchup threshold is strictly below allowed waste.
                self._catchup_threshold = 0.5 * self._allowed_waste

        self._initialized_internal_state = True

    def _update_progress(self):
        # Incrementally track total work done.
        task_segments = self.task_done_time or []
        current_len = len(task_segments)
        if current_len > self._last_task_done_len:
            new_segments = task_segments[self._last_task_done_len:current_len]
            self._progress_done += sum(new_segments)
            self._last_task_done_len = current_len

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized_internal_state:
            self._initialize_internal_state()

        self._update_progress()

        elapsed = self.env.elapsed_seconds
        progress = self._progress_done
        remaining_work = max(0.0, self.task_duration - progress)
        time_to_deadline = self.deadline - elapsed
        wasted_time = elapsed - progress  # overhead + idle so far

        # If task is effectively done or no time remains, avoid further cost.
        if remaining_work <= 0.0 or time_to_deadline <= 0.0:
            return ClusterType.NONE

        # Possibly enter "force on-demand" mode.
        if not self._force_on_demand:
            # If we've consumed our waste budget, or even an all-on-demand
            # schedule from now cannot meet the deadline, force using on-demand.
            if wasted_time >= self._allowed_waste or time_to_deadline <= remaining_work:
                self._force_on_demand = True

        if self._force_on_demand:
            return ClusterType.ON_DEMAND

        # Not yet forced to on-demand everywhere.
        if has_spot:
            # Prefer cheap spot when available in the non-panic phase.
            return ClusterType.SPOT

        # Spot is unavailable.
        gap = self.env.gap_seconds
        # Decide whether we can afford to idle (NONE) for this step.
        # We idle only while we're comfortably within the "early slack" region,
        # and idling one more step still leaves enough time to finish by
        # switching to on-demand afterward.
        can_idle = (
            self._catchup_threshold > 0.0
            and wasted_time < self._catchup_threshold
            and (time_to_deadline - gap) > remaining_work
        )

        if can_idle:
            return ClusterType.NONE

        # Otherwise, use on-demand to maintain progress.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)