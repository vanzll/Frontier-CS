import json
from argparse import Namespace
from typing import List
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_cost_aware"

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

        self.regions = self.env.get_num_regions()
        self.spot_price = 0.9701
        self.ondemand_price = 3.06
        self.price_ratio = self.spot_price / self.ondemand_price
        self.overhead_seconds = self.restart_overhead
        self.gap_seconds = 3600.0

        self.region_history = [{"spot_available": 0, "total_steps": 0} for _ in range(self.regions)]
        self.consecutive_failures = 0
        self.last_decision = ClusterType.NONE
        self.critical_mode = False
        self.time_since_last_switch = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        current_region = self.env.get_current_region()
        self.region_history[current_region]["total_steps"] += 1
        if has_spot:
            self.region_history[current_region]["spot_available"] += 1

        elapsed = self.env.elapsed_seconds
        deadline = self.deadline
        remaining_time = deadline - elapsed
        work_done = sum(self.task_done_time)
        remaining_work = self.task_duration - work_done

        if remaining_work <= 0:
            return ClusterType.NONE

        self.time_since_last_switch += self.gap_seconds

        if self.remaining_restart_overhead > 0:
            return ClusterType.NONE

        if remaining_time <= 0:
            return ClusterType.NONE

        time_needed = remaining_work
        if remaining_time < time_needed + self.overhead_seconds:
            self.critical_mode = True

        if self.critical_mode:
            if remaining_time < time_needed:
                return ClusterType.ON_DEMAND
            if has_spot and remaining_time > time_needed + self.overhead_seconds * 2:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if has_spot:
            spot_prob = self._estimate_spot_reliability(current_region)
            expected_spot_time = self._expected_spot_time(spot_prob)

            if expected_spot_time > self.overhead_seconds * (1 / self.price_ratio - 1):
                self.consecutive_failures = 0
                self.last_decision = ClusterType.SPOT
                return ClusterType.SPOT

        if remaining_time > time_needed * 1.5:
            best_region = self._find_best_region(current_region)
            if best_region != current_region and self.time_since_last_switch > self.overhead_seconds * 2:
                self.env.switch_region(best_region)
                self.time_since_last_switch = 0
                self.consecutive_failures = 0
                return ClusterType.NONE

            self.consecutive_failures += 1
            if self.consecutive_failures > 2:
                return ClusterType.ON_DEMAND
            return ClusterType.NONE

        if remaining_time < time_needed + self.overhead_seconds * 3:
            return ClusterType.ON_DEMAND

        if has_spot and remaining_time > time_needed * 1.2:
            return ClusterType.SPOT

        return ClusterType.NONE

    def _estimate_spot_reliability(self, region: int) -> float:
        hist = self.region_history[region]
        if hist["total_steps"] == 0:
            return 0.5
        return hist["spot_available"] / hist["total_steps"]

    def _expected_spot_time(self, reliability: float) -> float:
        if reliability == 0:
            return 0
        expected_lifetime = self.gap_seconds / (1 - reliability)
        return min(expected_lifetime, self.gap_seconds * 10)

    def _find_best_region(self, current_region: int) -> int:
        best_region = current_region
        best_reliability = self._estimate_spot_reliability(current_region)

        for region in range(self.regions):
            if region == current_region:
                continue
            reliability = self._estimate_spot_reliability(region)
            if reliability > best_reliability + 0.1:
                best_reliability = reliability
                best_region = region

        return best_region