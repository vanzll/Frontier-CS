from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "cant_be_late_heuristic_v1"

    def __init__(self, args=None):
        self.args = args
        self._cached_work_done = 0.0
        self._cached_segments_count = 0
        self._have_policy_params = False
        self._S0 = 0.0
        self._S_soft_min = 0.0
        self._S_soft_max = 0.0
        self._step_count = 0
        self._spot_available_count = 0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _init_policy_params(self):
        if self._have_policy_params:
            return
        try:
            slack_initial = max(self.deadline - self.task_duration, 0.0)
        except Exception:
            slack_initial = 0.0
        self._S0 = slack_initial
        if slack_initial <= 0.0:
            self._S_soft_min = 0.0
            self._S_soft_max = 0.0
        else:
            # Use 20%-60% of initial slack as range for soft threshold.
            self._S_soft_min = 0.2 * slack_initial
            self._S_soft_max = 0.6 * slack_initial
        self._have_policy_params = True

    def _get_work_done(self) -> float:
        td = self.task_done_time
        if isinstance(td, (int, float)):
            return float(td)
        try:
            n = len(td)
        except TypeError:
            try:
                return float(td)
            except Exception:
                return 0.0
        if n == 0:
            return 0.0
        if n != self._cached_segments_count:
            new_segments = td[self._cached_segments_count:n]
            self._cached_work_done += float(sum(new_segments))
            self._cached_segments_count = n
        return self._cached_work_done

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_policy_params()

        self._step_count += 1
        if has_spot:
            self._spot_available_count += 1

        work_done = self._get_work_done()
        remaining_work = self.task_duration - work_done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        now = self.env.elapsed_seconds
        time_left = self.deadline - now
        if time_left <= 0.0:
            # Already past deadline; minimize further delay.
            return ClusterType.ON_DEMAND

        gap = self.env.gap_seconds
        if gap <= 0.0:
            gap = 1e-6
        overhead = self.restart_overhead
        if overhead < 0.0:
            overhead = 0.0

        # Conservative risk margin: assume one step plus restart overhead may be lost.
        risk_margin = gap + overhead

        slack = time_left - remaining_work

        # If remaining_work already exceeds time_left, only hope is OD.
        if remaining_work > time_left + 1e-6:
            return ClusterType.ON_DEMAND

        # Hard safety: if slack is below risk margin, we cannot risk SPOT or idling.
        if slack <= risk_margin + 1e-6:
            return ClusterType.ON_DEMAND

        # Compute soft slack threshold based on observed spot availability.
        if self._S0 > 0.0:
            if self._step_count >= 10:
                p_est = self._spot_available_count / float(self._step_count)
            else:
                # Prior belief of moderate availability.
                p_est = 0.3
            if p_est < 0.0:
                p_est = 0.0
            elif p_est > 1.0:
                p_est = 1.0

            p_low = 0.1
            p_high = 0.5
            if p_est <= p_low:
                alpha = 1.0
            elif p_est >= p_high:
                alpha = 0.0
            else:
                alpha = (p_high - p_est) / (p_high - p_low)

            soft_slack = self._S_soft_min + alpha * (self._S_soft_max - self._S_soft_min)

            min_soft = 2.0 * risk_margin
            if soft_slack < min_soft:
                soft_slack = min_soft
            if soft_slack > self._S0:
                soft_slack = self._S0
        else:
            soft_slack = 2.0 * risk_margin

        # Decision logic.
        if has_spot:
            # In safe region, always prefer SPOT to minimize cost.
            return ClusterType.SPOT

        # No spot available.
        # If we have more slack than the soft threshold, we can afford to wait for spot.
        if slack > soft_slack + 1e-6:
            return ClusterType.NONE

        # Slack is getting tight (but still above hard margin): use on-demand.
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)