import json
from argparse import Namespace
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

        # Initialize tracking structures
        self.region_stats = []
        self.current_region = 0
        self.last_action = ClusterType.NONE
        self.consecutive_fails = 0
        self.max_switch_threshold = 2
        self.critical_threshold_multiplier = 2.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        # Update region statistics
        current_region = self.env.get_current_region()
        if len(self.region_stats) <= current_region:
            self.region_stats.append({"total": 0, "available": 0})
        self.region_stats[current_region]["total"] += 1
        if has_spot:
            self.region_stats[current_region]["available"] += 1

        # Calculate remaining work and time
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        time_remaining = self.deadline - self.env.elapsed_seconds

        # Emergency mode: must use on-demand to finish
        min_time_needed = work_remaining
        if last_cluster_type not in [ClusterType.ON_DEMAND, ClusterType.NONE]:
            min_time_needed += self.restart_overhead

        if time_remaining < min_time_needed * self.critical_threshold_multiplier:
            if last_cluster_type != ClusterType.ON_DEMAND and has_spot:
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        # Calculate spot reliability for current region
        reliability = 0.0
        stats = self.region_stats[current_region]
        if stats["total"] > 0:
            reliability = stats["available"] / stats["total"]

        # Try to find a better region
        if (not has_spot or reliability < 0.5) and self.consecutive_fails >= self.max_switch_threshold:
            best_region = current_region
            best_reliability = reliability

            for i in range(self.env.get_num_regions()):
                if i == current_region:
                    continue
                if len(self.region_stats) <= i:
                    region_rel = 0.5  # Default assumption
                else:
                    s = self.region_stats[i]
                    region_rel = s["available"] / max(s["total"], 1)

                if region_rel > best_reliability + 0.1:
                    best_reliability = region_rel
                    best_region = i

            if best_region != current_region:
                self.env.switch_region(best_region)
                self.consecutive_fails = 0
                current_region = best_region
                # After switching, assume spot is available initially
                if best_reliability > 0.3:
                    return ClusterType.SPOT

        # Normal decision making
        if has_spot:
            if reliability > 0.6 or work_remaining > time_remaining * 0.7:
                self.consecutive_fails = 0
                return ClusterType.SPOT
            elif last_cluster_type == ClusterType.SPOT:
                self.consecutive_fails = 0
                return ClusterType.SPOT
            else:
                self.consecutive_fails += 1
                if self.consecutive_fails > 1:
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
        else:
            self.consecutive_fails += 1
            if last_cluster_type == ClusterType.ON_DEMAND:
                return ClusterType.ON_DEMAND
            elif self.consecutive_fails > 1:
                return ClusterType.ON_DEMAND
            else:
                return ClusterType.NONE