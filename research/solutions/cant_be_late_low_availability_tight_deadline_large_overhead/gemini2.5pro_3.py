import collections
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "adaptive_buffer"

    HISTORY_WINDOW_SIZE = 120
    INITIAL_P_ESTIMATE = 0.20
    MIN_P_ESTIMATE = 0.01
    MIN_BUFFER_FACTOR = 3.0

    def __init__(self, args):
        super().__init__(args)
        self._initialized = False
        self.spot_availability_history = None
        self.estimated_spot_availability = self.INITIAL_P_ESTIMATE
        self.initial_slack = 0.0
        self.min_safety_buffer = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize(self):
        self.spot_availability_history = collections.deque(maxlen=self.HISTORY_WINDOW_SIZE)
        self.initial_slack = self.deadline - self.task_duration
        self.min_safety_buffer = self.MIN_BUFFER_FACTOR * self.restart_overhead
        self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialize()

        self.spot_availability_history.append(1 if has_spot else 0)
        if len(self.spot_availability_history) > 10:
            self.estimated_spot_availability = (
                sum(self.spot_availability_history) / len(self.spot_availability_history)
            )

        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 0:
            return ClusterType.NONE

        if has_spot:
            return ClusterType.SPOT

        p_safe = max(self.estimated_spot_availability, self.MIN_P_ESTIMATE)
        adaptive_buffer = (1 - p_safe) * self.initial_slack
        buffer = max(adaptive_buffer, self.min_safety_buffer)

        time_to_finish_on_od = work_remaining
        critical_start_time = self.deadline - time_to_finish_on_od - buffer

        if self.env.elapsed_seconds >= critical_start_time:
            return ClusterType.ON_DEMAND
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)