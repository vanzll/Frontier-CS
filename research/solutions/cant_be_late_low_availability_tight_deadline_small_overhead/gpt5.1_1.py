import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType


class Solution(Strategy):
    NAME = "my_solution"

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.committed_to_od = False
        self._policy_initialized = False
        self.initial_slack = 0.0
        self.safety_margin = 0.0
        self.waiting_margin = 0.0
        self.gap_seconds = 60.0
        self._done_total = 0.0
        self._done_seen_len = 0

    def solve(self, spec_path: str) -> "Solution":
        # Optional: could parse spec_path here if needed.
        self.committed_to_od = False
        self._policy_initialized = False
        self._done_total = 0.0
        self._done_seen_len = 0
        return self

    def _initialize_policy(self):
        if self._policy_initialized:
            return

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = 0.0

        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        self.initial_slack = max(deadline - task_duration, 0.0)

        gap = getattr(self.env, "gap_seconds", 60.0) or 60.0
        self.gap_seconds = float(gap)

        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        S = float(self.initial_slack)

        safety_by_slack = 0.2 * S
        safety_by_overhead = 4.0 * overhead
        safety_by_gap = 2.0 * self.gap_seconds
        safety_margin = max(safety_by_slack, safety_by_overhead, safety_by_gap)
        if S > 0.0:
            safety_margin = min(safety_margin, 0.9 * S)
        self.safety_margin = safety_margin

        waiting_by_slack = 0.5 * S
        waiting_by_overhead = 6.0 * overhead
        waiting_by_gap = self.safety_margin + 2.0 * self.gap_seconds
        waiting_margin = max(waiting_by_slack, waiting_by_overhead, waiting_by_gap)
        if S > 0.0:
            waiting_margin = min(waiting_margin, 0.95 * S)
        if waiting_margin < self.safety_margin:
            waiting_margin = self.safety_margin
        self.waiting_margin = waiting_margin

        self._policy_initialized = True

    def _get_total_done(self) -> float:
        td = self.task_done_time

        if isinstance(td, (int, float)):
            return float(td)

        # Incremental summation if list-like
        try:
            n = len(td)
        except TypeError:
            total = 0.0
            try:
                for v in td:
                    total += float(v)
            except Exception:
                return 0.0
            return total

        if n < self._done_seen_len:
            # List reset; recompute
            self._done_total = 0.0
            self._done_seen_len = 0

        try:
            for i in range(self._done_seen_len, n):
                self._done_total += float(td[i])
            self._done_seen_len = n
        except Exception:
            # Fallback: full recompute if elements are not simple numbers
            self._done_total = 0.0
            self._done_seen_len = 0
            try:
                for v in td:
                    self._done_total += float(v)
                    self._done_seen_len += 1
            except Exception:
                self._done_total = 0.0
                self._done_seen_len = 0
        return self._done_total

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._policy_initialized:
            self._initialize_policy()

        now = float(getattr(self.env, "elapsed_seconds", 0.0))
        gap = float(getattr(self.env, "gap_seconds", self.gap_seconds))
        self.gap_seconds = gap

        done = self._get_total_done()
        try:
            task_duration = float(self.task_duration)
        except Exception:
            task_duration = 0.0

        remaining = max(task_duration - done, 0.0)

        if remaining <= 0.0:
            self.committed_to_od = True
            return ClusterType.NONE

        try:
            deadline = float(self.deadline)
        except Exception:
            deadline = now

        time_left = deadline - now

        try:
            overhead = float(self.restart_overhead)
        except Exception:
            overhead = 0.0

        if time_left <= 0.0:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        required_od = remaining + overhead

        if self.committed_to_od:
            return ClusterType.ON_DEMAND

        if time_left <= required_od:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        slack_od = time_left - required_od  # >= 0 here

        safety_margin = max(self.safety_margin, 2.0 * gap, 4.0 * overhead)
        if self.initial_slack > 0.0:
            safety_margin = min(safety_margin, 0.9 * self.initial_slack)
        self.safety_margin = safety_margin

        if slack_od <= self.safety_margin:
            self.committed_to_od = True
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        waiting_margin = max(self.waiting_margin, self.safety_margin + 2.0 * gap, 6.0 * overhead)
        if self.initial_slack > 0.0:
            waiting_margin = min(waiting_margin, 0.95 * self.initial_slack)
        if waiting_margin < self.safety_margin:
            waiting_margin = self.safety_margin
        self.waiting_margin = waiting_margin

        if slack_od > self.waiting_margin:
            return ClusterType.NONE

        self.committed_to_od = True
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)