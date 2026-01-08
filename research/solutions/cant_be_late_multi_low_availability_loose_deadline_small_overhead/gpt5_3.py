import json
import math
from argparse import Namespace
from typing import List

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multi_region_v1"

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

        # Initialize internal state after env is created by base class
        self._init_internal_state()
        return self

    # ============== Internal State and Helpers ==============

    def _init_internal_state(self):
        self.num_regions = max(1, getattr(self.env, "get_num_regions", lambda: 1)())
        self.region_stats = []
        for _ in range(self.num_regions):
            self.region_stats.append(
                {
                    "obs": 0,            # observations in this region
                    "spot": 0,           # number of spot-available steps
                    "last_avail": None,  # last has_spot value
                    "streak": 0,         # current consecutive availability streak
                    "best_streak": 0,    # best observed availability streak
                }
            )
        self.step_count = 0

        # Runtime mode tracking
        self.last_action = ClusterType.NONE
        self.last_switch_time = 0.0  # elapsed_seconds when last ON_DEMAND was started
        self.last_on_demand_started = None

        # Parameters (tuned heuristics)
        self.beta_prior_alpha = 2.0  # successes prior
        self.beta_prior_beta = 2.0   # failures prior

        # Guard threshold multipliers
        self.wait_safety_multiplier = 1.2
        self.switch_safety_multiplier = 1.1

        # Minimum runtime on on-demand before considering stopping it voluntarily
        # Helps avoid thrashing when availability oscillates quickly.
        self.min_on_demand_runtime_sec = 30 * 60  # 30 minutes

        # Exploration parameter for UCB region selection
        self.ucb_c = 1.5

        # Limit how often we switch regions when waiting (once every few minutes)
        self.wait_region_switch_cooldown_sec = 5 * 60  # 5 minutes
        self.last_region_switch_time = -1e18

        # Remember last region we targeted while waiting, avoids oscillation
        self.last_target_region = None

    def _update_region_stats(self, region_idx: int, has_spot: bool):
        stats = self.region_stats[region_idx]
        stats["obs"] += 1
        if has_spot:
            stats["spot"] += 1
            if stats["last_avail"] is True:
                stats["streak"] += 1
            else:
                stats["streak"] = 1
            if stats["streak"] > stats["best_streak"]:
                stats["best_streak"] = stats["streak"]
        else:
            stats["streak"] = 0
        stats["last_avail"] = has_spot

    def _p_hat(self, region_idx: int) -> float:
        stats = self.region_stats[region_idx]
        succ = stats["spot"] + self.beta_prior_alpha
        fail = (stats["obs"] - stats["spot"]) + self.beta_prior_beta
        return succ / (succ + fail)

    def _ucb_score(self, region_idx: int) -> float:
        p = self._p_hat(region_idx)
        n = max(1, self.region_stats[region_idx]["obs"])
        if self.step_count <= 1:
            bonus = 1.0
        else:
            bonus = self.ucb_c * math.sqrt(2.0 * math.log(self.step_count) / n)
        return min(1.0, p + bonus)

    def _best_region_by_ucb(self) -> int:
        best_idx = 0
        best_score = -1.0
        for i in range(self.num_regions):
            score = self._ucb_score(i)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _best_region_by_mean(self) -> int:
        best_idx = 0
        best_val = -1.0
        for i in range(self.num_regions):
            val = self._p_hat(i)
            if val > best_val:
                best_val = val
                best_idx = i
        return best_idx

    def _expected_wait_time_sec(self, p: float) -> float:
        # Expected additional steps until next success when we are at a failure now:
        # For geometric(p), E[G] = 1/p (number of trials to success), so waiting (excluding current failed step)
        # roughly ~ (1/p). Convert to seconds by multiplying gap.
        p = max(1e-6, min(0.999999, p))
        return self.env.gap_seconds * (1.0 / p)

    def _slack_sec(self, time_left: float, work_left: float) -> float:
        # Slack = time that can be idle and still finish if cluster always available, ignoring overheads
        return time_left - work_left

    def _choose_wait_region_if_needed(self, current_region: int):
        # Choose a region to be in while waiting for spot; prefer region with best UCB or mean.
        now = self.env.elapsed_seconds
        if now - self.last_region_switch_time < self.wait_region_switch_cooldown_sec:
            return  # do not switch too frequently

        target = self._best_region_by_ucb()
        if target != current_region:
            self.env.switch_region(target)
            self.last_region_switch_time = now
            self.last_target_region = target

    def _compute_should_use_on_demand(self, time_left: float, work_left: float, current_region: int) -> bool:
        # Decide whether we need to run on-demand to stay on track.
        # Use predicted expected wait time for best region to guide waiting policy.
        # Slack must cover expected waiting and restart overhead with a safety margin.
        slack = self._slack_sec(time_left, work_left)

        # Conservative: subtract pending restart overhead to be safe
        pending_overhead = getattr(self, "remaining_restart_overhead", 0.0) if hasattr(self, "remaining_restart_overhead") else 0.0
        slack_adjusted = slack - pending_overhead

        # Estimate fastest region to get spot next
        best_region = self._best_region_by_mean()
        p_best = self._p_hat(best_region)
        expected_wait = self._expected_wait_time_sec(p_best)

        safety = self.restart_overhead * self.wait_safety_multiplier + self.env.gap_seconds

        # If slack is insufficient to cover a typical wait plus safety, use on-demand
        if slack_adjusted < expected_wait + safety:
            return True
        return False

    def _should_switch_from_on_demand_to_spot(self, time_left: float, work_left: float) -> bool:
        # Switch from ON_DEMAND to SPOT if we have enough slack to absorb restart overhead.
        # Avoid switching too close to deadline.
        slack = self._slack_sec(time_left, work_left)
        safety = self.restart_overhead * self.switch_safety_multiplier + self.env.gap_seconds
        # Also ensure we have run on-demand for a minimum time before stopping to avoid thrash
        if self.last_on_demand_started is not None:
            if (self.env.elapsed_seconds - self.last_on_demand_started) < self.min_on_demand_runtime_sec:
                return False
        return slack >= (self.restart_overhead + safety)

    def _complete_or_fail_guard(self, time_left: float, work_left: float) -> ClusterType:
        # If done, or cannot finish even with on-demand, handle gracefully.
        if work_left <= 0:
            return ClusterType.NONE
        # If time_left is extremely low, must choose ON_DEMAND to try to finish at max rate
        if time_left <= 0:
            return ClusterType.ON_DEMAND
        return None

    # ============== Core Decision Logic ==============

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Step counter
        self.step_count += 1

        # Update stats for current region
        current_region = self.env.get_current_region() if hasattr(self.env, "get_current_region") else 0
        if 0 <= current_region < self.num_regions:
            self._update_region_stats(current_region, has_spot)

        # Gather timings
        elapsed = self.env.elapsed_seconds
        time_left = max(0.0, self.deadline - elapsed)

        # Work left
        done = sum(self.task_done_time) if isinstance(self.task_done_time, List) else float(self.task_done_time)
        work_left = max(0.0, self.task_duration - done)

        # Early guarding
        guard = self._complete_or_fail_guard(time_left, work_left)
        if guard is not None:
            # Track ON_DEMAND start time if we choose ON_DEMAND
            if guard == ClusterType.ON_DEMAND:
                if last_cluster_type != ClusterType.ON_DEMAND:
                    self.last_on_demand_started = self.env.elapsed_seconds
            return guard

        # If spot is available, prefer SPOT in almost all cases (cheap and progress now)
        if has_spot:
            # If currently ON_DEMAND, consider whether to switch to SPOT (incurs overhead)
            if last_cluster_type == ClusterType.ON_DEMAND:
                if self._should_switch_from_on_demand_to_spot(time_left, work_left):
                    return ClusterType.SPOT
                else:
                    # Continue ON_DEMAND to avoid risking deadline
                    return ClusterType.ON_DEMAND
            # If currently NONE or SPOT, run SPOT
            return ClusterType.SPOT

        # Spot is not available in current region for this step.
        # Decide between waiting or using ON_DEMAND.
        use_on_demand = self._compute_should_use_on_demand(time_left, work_left, current_region)

        if use_on_demand:
            # When choosing ON_DEMAND, switch to region with highest expected spot availability
            # so that we can switch to SPOT soon after if it becomes available.
            target_region = self._best_region_by_mean()
            if target_region != current_region:
                self.env.switch_region(target_region)
            if last_cluster_type != ClusterType.ON_DEMAND:
                self.last_on_demand_started = self.env.elapsed_seconds
            return ClusterType.ON_DEMAND

        # Choose to wait (NONE) and move to the best region for SPOT if necessary
        self._choose_wait_region_if_needed(current_region)
        return ClusterType.NONE