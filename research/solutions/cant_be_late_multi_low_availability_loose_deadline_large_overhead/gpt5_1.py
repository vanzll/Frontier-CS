import json
from argparse import Namespace
from typing import List, Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        # Runtime state will be initialized in the first _step call
        self._state_initialized = False
        self._locked_on_demand = False
        return self

    def _init_runtime_state(self):
        n = self.env.get_num_regions()
        self._region_total_steps: List[int] = [0 for _ in range(n)]
        self._region_spot_available_steps: List[int] = [0 for _ in range(n)]
        self._region_last_switch_step: int = -1
        self._step_counter: int = 0
        # Prior for Beta distribution (alpha, beta)
        self._beta_prior_alpha: float = 1.0
        self._beta_prior_beta: float = 1.0
        # Hysteresis/cooldown to avoid thrashing regions when idling
        self._region_switch_cooldown_steps: int = 3
        # Minimal score improvement to justify switching region while idling
        self._min_region_score_improvement: float = 0.03
        # Remember if we were previously running SPOT in the same region
        self._last_spot_region: Optional[int] = None

        # Safety buffers
        # Commit buffer ensures we can survive one more failed spot attempt:
        # - one full step lost + up to two restart overheads (preempt + commit)
        self._commit_buffer_seconds: float = self.env.gap_seconds + 2 * self.restart_overhead

        # Guard for switching from OD/NONE to SPOT (requires overhead); ensure ample slack
        self._switch_to_spot_guard_seconds: float = self.env.gap_seconds + 2 * self.restart_overhead

        self._state_initialized = True

    def _estimate_region_score(self, idx: int) -> float:
        # Beta posterior mean with symmetric prior
        succ = self._region_spot_available_steps[idx]
        tot = self._region_total_steps[idx]
        a = self._beta_prior_alpha
        b = self._beta_prior_beta
        return (succ + a) / (tot + a + b) if (tot + a + b) > 0 else a / (a + b)

    def _best_region(self) -> int:
        n = self.env.get_num_regions()
        scores = [self._estimate_region_score(i) for i in range(n)]
        best_idx = 0
        best_score = scores[0]
        for i in range(1, n):
            if scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        return best_idx

    def _should_commit_to_on_demand(self, time_left: float, remaining_work: float) -> bool:
        # Commit when we are within a conservative threshold of deadline
        # ensuring we can withstand one more failed spot attempt (one gap + overheads)
        return time_left <= remaining_work + self._commit_buffer_seconds

    def _safe_to_attempt_spot(self, time_left: float, remaining_work: float) -> bool:
        # Safe to try spot if we have slack over deterministic on-demand completion
        return time_left > remaining_work + self._switch_to_spot_guard_seconds

    def _select_region_while_idling(self):
        # Decide whether to switch region during an idle step to the best predicted one
        current = self.env.get_current_region()
        best = self._best_region()
        if best == current:
            return
        # Switch only if improvement is meaningful and cooldown passed
        cur_score = self._estimate_region_score(current)
        best_score = self._estimate_region_score(best)
        if best_score - cur_score >= self._min_region_score_improvement:
            if self._step_counter - self._region_last_switch_step >= self._region_switch_cooldown_steps:
                self.env.switch_region(best)
                self._region_last_switch_step = self._step_counter

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not self._state_initialized:
            self._init_runtime_state()

        # Update region stats with current observation
        current_region = self.env.get_current_region()
        self._region_total_steps[current_region] += 1
        if has_spot:
            self._region_spot_available_steps[current_region] += 1

        # Compute core timing metrics
        elapsed = self.env.elapsed_seconds
        time_left = self.deadline - elapsed
        remaining_work = self.task_duration - sum(self.task_done_time)
        self._step_counter += 1

        # If already locked into on-demand, never switch back
        if self._locked_on_demand:
            return ClusterType.ON_DEMAND

        # Commit to ON_DEMAND if we might miss the deadline otherwise
        if self._should_commit_to_on_demand(time_left, remaining_work):
            self._locked_on_demand = True
            return ClusterType.ON_DEMAND

        # Not committing yet: try to exploit spot if safe
        if has_spot:
            # If we are already on SPOT in this region, continue to avoid unnecessary overhead
            if last_cluster_type == ClusterType.SPOT:
                self._last_spot_region = current_region
                return ClusterType.SPOT

            # If we are on OD or NONE, consider switching to SPOT only if we have ample slack
            if self._safe_to_attempt_spot(time_left, remaining_work):
                self._last_spot_region = current_region
                return ClusterType.SPOT

            # Otherwise, maintain OD if already running, else idle
            if last_cluster_type == ClusterType.ON_DEMAND:
                self._locked_on_demand = True
                return ClusterType.ON_DEMAND
            else:
                # Idle to preserve safety until we commit threshold
                self._select_region_while_idling()
                return ClusterType.NONE

        # Spot unavailable in current region
        # If we have enough time to wait, idle and possibly switch to a better region
        if self._safe_to_attempt_spot(time_left, remaining_work):
            self._select_region_while_idling()
            return ClusterType.NONE

        # Close to deadline but not within commit threshold yet; prioritize safety.
        # To avoid last-minute risk, start OD now.
        self._locked_on_demand = True
        return ClusterType.ON_DEMAND