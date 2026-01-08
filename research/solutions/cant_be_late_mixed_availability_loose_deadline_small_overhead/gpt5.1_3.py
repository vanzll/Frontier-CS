import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "slack_based_cant_be_late"

    def __init__(self, args=None):
        super().__init__(args)
        self._initialized = False
        self._task_duration = 0.0
        self._deadline = 0.0
        self._restart_overhead = 0.0
        self.initial_slack = 0.0
        self.SLACK_IDLE_STOP = 0.0
        self.SLACK_FORCE_OD = 0.0
        self.phase = 0  # 0: idle-ok, 1: no-idle, 2: force-OD

        # Tracking work done efficiently
        self._task_done_sum = 0.0
        self._last_task_done_len = 0

        # Spot availability stats for mild adaptation
        self._steps = 0
        self._spot_available_steps = 0
        self._adapt_steps = None

    def solve(self, spec_path: str) -> "Solution":
        return self

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)

    def _initialize_from_env(self):
        self._initialized = True

        # Core parameters (in seconds)
        try:
            self._task_duration = float(getattr(self, "task_duration", 0.0))
        except (TypeError, ValueError):
            self._task_duration = 0.0

        try:
            self._deadline = float(getattr(self, "deadline", 0.0))
        except (TypeError, ValueError):
            self._deadline = self._task_duration

        try:
            self._restart_overhead = float(getattr(self, "restart_overhead", 0.0))
        except (TypeError, ValueError):
            self._restart_overhead = 0.0

        self.initial_slack = max(0.0, self._deadline - self._task_duration)
        slack = self.initial_slack
        ro = self._restart_overhead

        # Thresholds based on slack and restart overhead
        slack_force_frac = 0.1  # when remaining slack below this, force OD
        slack_idle_frac = 0.4   # allow idling while slack above this

        self.SLACK_FORCE_OD = max(3.0 * ro, slack_force_frac * slack)
        self.SLACK_IDLE_STOP = max(6.0 * ro, slack_idle_frac * slack)

        # Ensure strict ordering between thresholds
        if self.SLACK_IDLE_STOP <= self.SLACK_FORCE_OD:
            self.SLACK_IDLE_STOP = self.SLACK_FORCE_OD + 3.0 * ro

        # Compute when to adapt thresholds based on observed availability (~first 6h)
        gap = getattr(self.env, "gap_seconds", 1.0)
        try:
            gap = float(gap)
        except (TypeError, ValueError):
            gap = 1.0
        if gap <= 0.0:
            gap = 1.0
        self._adapt_steps = max(1, int(6 * 3600.0 / gap))

        self.phase = 0
        self._steps = 0
        self._spot_available_steps = 0
        self._task_done_sum = 0.0
        self._last_task_done_len = 0

    def _get_done_work(self) -> float:
        """Efficiently compute total work done from task_done_time."""
        td = getattr(self, "task_done_time", 0.0)
        if isinstance(td, (list, tuple)):
            n = len(td)
            if n > self._last_task_done_len:
                segment_sum = 0.0
                for v in td[self._last_task_done_len:]:
                    try:
                        segment_sum += float(v)
                    except (TypeError, ValueError):
                        continue
                self._task_done_sum += segment_sum
                self._last_task_done_len = n
            return self._task_done_sum
        try:
            return float(td)
        except (TypeError, ValueError):
            return 0.0

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialize_from_env()

        # Update availability stats
        self._steps += 1
        if has_spot:
            self._spot_available_steps += 1

        # Time left until deadline
        try:
            elapsed = float(getattr(self.env, "elapsed_seconds", 0.0))
        except (TypeError, ValueError):
            elapsed = 0.0
        time_left = max(0.0, self._deadline - elapsed)

        # Remaining work
        done_work = self._get_done_work()
        work_remaining = max(0.0, self._task_duration - done_work)

        # If already finished, do nothing to avoid extra cost
        if work_remaining <= 0.0:
            return ClusterType.NONE

        # Current slack (time buffer) in seconds
        slack = time_left - work_remaining

        # One-time adaptation after early observation window
        if self._adapt_steps is not None and self._steps == self._adapt_steps:
            total = self._steps
            if total > 0:
                avail_ratio = self._spot_available_steps / float(total)
                # If spot availability is very low, reduce idle allowance
                if avail_ratio < 0.2:
                    new_idle_stop = max(
                        self.SLACK_FORCE_OD + 3.0 * self._restart_overhead,
                        0.2 * self.initial_slack,
                    )
                    if new_idle_stop < self.SLACK_IDLE_STOP:
                        self.SLACK_IDLE_STOP = new_idle_stop

        # Determine current phase based on slack; enforce monotonicity
        if slack <= 0.0:
            new_phase = 2
        elif slack <= self.SLACK_FORCE_OD:
            new_phase = 2
        elif slack <= self.SLACK_IDLE_STOP:
            new_phase = 1
        else:
            new_phase = 0

        if new_phase > self.phase:
            self.phase = new_phase

        # Decision logic
        # Phase 2: critical – always run on-demand to eliminate risk
        if self.phase >= 2:
            return ClusterType.ON_DEMAND

        # Phase 1: catch-up – never idle; use spot when present, OD otherwise
        if self.phase == 1:
            if has_spot:
                return ClusterType.SPOT
            else:
                return ClusterType.ON_DEMAND

        # Phase 0: plenty of slack – only use spot; otherwise safely idle
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE