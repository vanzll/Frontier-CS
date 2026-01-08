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

        self.trace_files = config["trace_files"]
        self.spot_availability = []
        for trace_file in self.trace_files:
            with open(trace_file) as f:
                lines = f.readlines()
                self.spot_availability.append([bool(int(line.strip())) for line in lines])

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.remaining_restart_overhead > 0 and last_cluster_type != ClusterType.NONE:
            return last_cluster_type

        remaining_work = self.task_duration - sum(self.task_done_time)
        time_left = self.deadline - self.env.elapsed_seconds

        critical_margin = 2 * self.restart_overhead
        if remaining_work > time_left - critical_margin:
            return ClusterType.ON_DEMAND

        step_idx = int(self.env.elapsed_seconds / 3600)
        current_region = self.env.get_current_region()

        if has_spot:
            if last_cluster_type == ClusterType.SPOT:
                return ClusterType.SPOT
            else:
                consecutive = self._consecutive_spot(current_region, step_idx, 10)
                if consecutive >= 2:
                    return ClusterType.SPOT
                else:
                    if last_cluster_type == ClusterType.ON_DEMAND:
                        return ClusterType.ON_DEMAND
                    return ClusterType.NONE
        else:
            best_region, best_streak = self._best_region_for_spot(step_idx, current_region)
            if best_streak >= 2:
                self.env.switch_region(best_region)
                return ClusterType.SPOT
            else:
                if last_cluster_type == ClusterType.ON_DEMAND:
                    return ClusterType.ON_DEMAND
                return ClusterType.NONE

    def _consecutive_spot(self, region: int, start_idx: int, max_look: int) -> int:
        count = 0
        for i in range(max_look):
            idx = start_idx + i
            if idx >= len(self.spot_availability[region]):
                break
            if self.spot_availability[region][idx]:
                count += 1
            else:
                break
        return count

    def _best_region_for_spot(self, step_idx: int, exclude_region: int):
        best_region = exclude_region
        best_streak = 0
        for r in range(self.env.get_num_regions()):
            if r == exclude_region:
                continue
            if step_idx >= len(self.spot_availability[r]):
                continue
            if not self.spot_availability[r][step_idx]:
                continue
            streak = self._consecutive_spot(r, step_idx, 10)
            if streak > best_streak:
                best_streak = streak
                best_region = r
        return best_region, best_streak