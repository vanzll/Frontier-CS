import math
from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "cbl_solution"

    def __init__(self, args):
        super().__init__(args)
        self.spot_available_history = []
        self.remaining_work = 0.0
        self.spot_downtime_start = None
        self.consecutive_spot_available = 0
        self.consecutive_spot_unavailable = 0
        self.restart_pending = False
        self.restart_remaining = 0.0
        self.last_decision = ClusterType.NONE
        self.state = "INITIAL"
        self.spot_streak_threshold = 5
        self.safety_margin_multiplier = 1.5
        self.panic_mode = False
        self.panic_threshold = 0.15

    def solve(self, spec_path: str) -> "Solution":
        return self

    def _update_state_variables(self, last_cluster_type: ClusterType, has_spot: bool):
        # Update availability history
        self.spot_available_history.append(1 if has_spot else 0)
        if len(self.spot_available_history) > 100:
            self.spot_available_history.pop(0)

        # Update consecutive counters
        if has_spot:
            self.consecutive_spot_available += 1
            self.consecutive_spot_unavailable = 0
            if self.spot_downtime_start is not None:
                self.spot_downtime_start = None
        else:
            self.consecutive_spot_unavailable += 1
            self.consecutive_spot_available = 0
            if self.spot_downtime_start is None:
                self.spot_downtime_start = self.env.elapsed_seconds

        # Update restart overhead tracking
        if self.restart_pending:
            self.restart_remaining -= self.env.gap_seconds
            if self.restart_remaining <= 0:
                self.restart_pending = False
                self.restart_remaining = 0.0

        # Calculate remaining work
        completed_work = sum(self.task_done_time)
        self.remaining_work = self.task_duration - completed_work

        # Update panic mode based on deadline
        time_left = self.deadline - self.env.elapsed_seconds
        required_rate = self.remaining_work / max(time_left, 0.001)
        self.panic_mode = required_rate > 0.9 or time_left < self.remaining_work * self.safety_margin_multiplier

    def _calculate_safety_required_od_time(self, time_left: float) -> float:
        if time_left <= 0:
            return 0
        
        # Conservative estimate accounting for potential spot failures
        spot_success_prob = max(0.01, sum(self.spot_available_history) / max(len(self.spot_available_history), 1))
        
        # Expected work per spot time unit
        expected_spot_efficiency = spot_success_prob * 0.95  # 5% overhead for restart
        
        # Required on-demand time to guarantee completion
        if expected_spot_efficiency <= 0:
            return time_left
        
        # Solve: od_time + (time_left - od_time) * expected_spot_efficiency >= remaining_work
        if expected_spot_efficiency >= 1:
            return 0
        
        required_od = (self.remaining_work - time_left * expected_spot_efficiency) / (1 - expected_spot_efficiency)
        return max(0, required_od)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._update_state_variables(last_cluster_type, has_spot)
        
        time_left = self.deadline - self.env.elapsed_seconds
        
        # Emergency: if we can't finish even with 100% on-demand
        if time_left <= 0 or (self.remaining_work > 0 and time_left < self.remaining_work * 0.9):
            return ClusterType.ON_DEMAND if self.remaining_work > 0 else ClusterType.NONE
        
        # If no work left, do nothing
        if self.remaining_work <= 0:
            return ClusterType.NONE
        
        # Calculate conservative required on-demand time
        required_od_time = self._calculate_safety_required_od_time(time_left)
        
        # Calculate current on-demand "budget" based on elapsed time
        elapsed_fraction = self.env.elapsed_seconds / self.deadline
        od_time_used = sum(1 for t in self.task_done_time if t > 0 and self.last_decision == ClusterType.ON_DEMAND) * self.env.gap_seconds
        od_time_budget = required_od_time - od_time_used
        
        # Determine if we should use on-demand now
        use_on_demand = False
        
        # Rule 1: Panic mode - use on-demand if behind schedule
        if self.panic_mode:
            use_on_demand = True
        
        # Rule 2: If we're in restart overhead period and behind schedule
        if self.restart_pending and time_left < self.remaining_work * self.safety_margin_multiplier:
            use_on_demand = True
        
        # Rule 3: If spot has been unavailable for too long
        if self.consecutive_spot_unavailable > 10 and time_left < self.remaining_work * 2:
            use_on_demand = True
        
        # Rule 4: If we need to spend on-demand time to meet safety requirement
        time_until_deadline = self.deadline - self.env.elapsed_seconds
        if od_time_budget > 0 and time_until_deadline < od_time_budget * self.safety_margin_multiplier:
            use_on_demand = True
        
        # Rule 5: If we're very close to deadline
        if time_left < self.remaining_work + self.restart_overhead:
            use_on_demand = True
        
        if use_on_demand:
            self.last_decision = ClusterType.ON_DEMAND
            return ClusterType.ON_DEMAND
        
        # Use spot if available
        if has_spot:
            # Only use spot if we have a reasonable streak or we're not in a risky situation
            if (self.consecutive_spot_available >= self.spot_streak_threshold or 
                time_left > self.remaining_work * 3):
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
        
        # Otherwise wait
        self.last_decision = ClusterType.NONE
        return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)