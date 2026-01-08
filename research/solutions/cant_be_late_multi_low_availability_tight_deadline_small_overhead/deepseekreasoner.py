import json
import math
from argparse import Namespace
from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "my_strategy"

    def __init__(self, args=None):
        super().__init__(args)
        self.region_count = 0
        self.gap_seconds = 0
        self.spot_history = {}
        self.region_weights = []
        self.switch_cooldown = {}
        self.consecutive_failures = 0
        self.last_decision = ClusterType.NONE
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        self.min_spot_confidence = 0.6
        self.emergency_threshold = 0.85

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
        return self

    def _update_spot_history(self, region_idx, has_spot):
        if region_idx not in self.spot_history:
            self.spot_history[region_idx] = []
        self.spot_history[region_idx].append(has_spot)
        # Keep recent history only (last 50 entries)
        if len(self.spot_history[region_idx]) > 50:
            self.spot_history[region_idx] = self.spot_history[region_idx][-50:]

    def _get_spot_confidence(self, region_idx):
        if region_idx not in self.spot_history or not self.spot_history[region_idx]:
            return 0.5
        history = self.spot_history[region_idx]
        recent = history[-min(10, len(history)):]
        return sum(1 for x in recent if x) / len(recent)

    def _should_switch_region(self, current_region, has_spot):
        if self.region_count <= 1:
            return False

        remaining_work = self.task_duration - sum(self.task_done_time)
        remaining_time = self.deadline - self.env.elapsed_seconds

        if remaining_time <= 0 or remaining_work <= 0:
            return False

        # Critical time check
        time_per_region = self.gap_seconds * 2
        if remaining_time < time_per_region * 2:
            return False

        # Check if current region has been failing
        if current_region in self.switch_cooldown:
            if self.env.elapsed_seconds - self.switch_cooldown[current_region] < 3600:
                return False

        if not has_spot and self.last_decision == ClusterType.SPOT:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 2:
                return True
        else:
            self.consecutive_failures = 0

        # Check other regions
        best_region = current_region
        best_confidence = self._get_spot_confidence(current_region)
        if not has_spot:
            best_confidence = 0

        for i in range(self.region_count):
            if i == current_region:
                continue
            confidence = self._get_spot_confidence(i)
            if confidence > best_confidence + 0.2:
                best_region = i
                best_confidence = confidence

        return best_region != current_region

    def _select_best_region(self, current_region):
        if self.region_count <= 1:
            return current_region

        best_region = current_region
        best_score = -1

        for i in range(self.region_count):
            if i == current_region:
                continue
            confidence = self._get_spot_confidence(i)
            # Penalize recently switched regions
            penalty = 0
            if i in self.switch_cooldown:
                time_since_switch = self.env.elapsed_seconds - self.switch_cooldown[i]
                if time_since_switch < 7200:
                    penalty = 0.3 * (1 - time_since_switch / 7200)

            score = confidence - penalty
            if score > best_score:
                best_score = score
                best_region = i

        return best_region

    def _calculate_time_pressure(self):
        remaining_work = self.task_duration - sum(self.task_done_time)
        remaining_time = self.deadline - self.env.elapsed_seconds

        if remaining_time <= 0:
            return 1.0

        # Account for potential restart overhead
        safe_time_needed = remaining_work + self.restart_overhead * 2
        if safe_time_needed <= 0:
            return 0.0

        time_pressure = 1 - (remaining_time / (safe_time_needed * 1.2))
        return max(0.0, min(1.0, time_pressure))

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if not hasattr(self, 'gap_seconds') or self.gap_seconds == 0:
            self.gap_seconds = getattr(self.env, 'gap_seconds', 3600.0)

        if self.region_count == 0:
            self.region_count = self.env.get_num_regions()
            self.region_weights = [1.0] * self.region_count

        current_region = self.env.get_current_region()
        self._update_spot_history(current_region, has_spot)

        remaining_work = self.task_duration - sum(self.task_done_time)
        remaining_time = self.deadline - self.env.elapsed_seconds

        # Check if we're done
        if remaining_work <= 0:
            return ClusterType.NONE

        # Check if we've missed deadline
        if remaining_time <= 0:
            return ClusterType.ON_DEMAND

        # Calculate time pressure
        time_pressure = self._calculate_time_pressure()

        # Emergency mode - must finish
        if time_pressure > self.emergency_threshold or remaining_time < remaining_work * 1.5:
            if self.remaining_restart_overhead > 0:
                return ClusterType.NONE
            return ClusterType.ON_DEMAND

        # Check if we should switch region
        if self._should_switch_region(current_region, has_spot):
            best_region = self._select_best_region(current_region)
            if best_region != current_region:
                self.env.switch_region(best_region)
                self.switch_cooldown[current_region] = self.env.elapsed_seconds
                self.consecutive_failures = 0
                # When switching, we may want to be conservative
                if time_pressure > 0.5:
                    return ClusterType.ON_DEMAND
                return ClusterType.NONE

        # Decision logic based on spot availability and time pressure
        if has_spot:
            confidence = self._get_spot_confidence(current_region)
            spot_threshold = self.min_spot_confidence - time_pressure * 0.3

            if confidence >= spot_threshold:
                # Use spot if we have good confidence
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT
            else:
                # Low confidence in spot, use on-demand if we can afford it
                if time_pressure < 0.7:
                    self.last_decision = ClusterType.ON_DEMAND
                    return ClusterType.ON_DEMAND
                else:
                    # In moderate pressure, still try spot
                    self.last_decision = ClusterType.SPOT
                    return ClusterType.SPOT
        else:
            # No spot available
            if time_pressure < 0.6:
                # Not urgent, wait for spot
                return ClusterType.NONE
            else:
                # Getting urgent, use on-demand
                self.last_decision = ClusterType.ON_DEMAND
                return ClusterType.ON_DEMAND