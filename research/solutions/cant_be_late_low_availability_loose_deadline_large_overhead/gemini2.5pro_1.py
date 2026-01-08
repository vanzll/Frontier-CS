import argparse
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "expert_programmer_solution"

    def __init__(self, args):
        super().__init__(args)
        self._initialized: bool = False
        self._risk_buffer: float = 0.0
        self._target_finish_time: float = 0.0
        self._target_rate: float = 0.0

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _initialize_parameters(self):
        self._risk_buffer = self.restart_overhead * 1.5

        initial_slack = self.deadline - self.task_duration
        if initial_slack < 0:
            initial_slack = 0
        
        safety_buffer_slack = initial_slack / 2.0
        self._target_finish_time = self.deadline - safety_buffer_slack

        if self._target_finish_time > 1e-9:
            self._target_rate = self.task_duration / self._target_finish_time
        else:
            self._target_rate = float('inf')
        
        self._initialized = True

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._initialized:
            self._initialize_parameters()

        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done

        if work_remaining <= 1e-9:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_remaining_until_deadline = self.deadline - current_time
        
        slack = time_remaining_until_deadline - work_remaining

        if slack <= self._risk_buffer:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            if current_time < self._target_finish_time:
                target_work_done = self._target_rate * current_time
            else:
                target_work_done = self.task_duration
            
            if work_done < target_work_done:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser):
        args, _ = parser.parse_known_args()
        return cls(args)