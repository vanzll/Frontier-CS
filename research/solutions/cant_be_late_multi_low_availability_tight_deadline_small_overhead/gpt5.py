import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cbm_multi_v1"

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

        # Internal state for multi-region handling
        self._init_done = False
        self._committed_to_od = False
        self._alpha = 3.0  # smoothing for region availability estimates
        return self

    def _lazy_init(self):
        if self._init_done:
            return
        try:
            n = self.env.get_num_regions()
        except Exception:
            n = 1
        self._num_regions = n
        self._spot_obs = [0] * n
        self._total_obs = [0] * n
        self._init_done = True

    def _best_region(self, exclude_current=False) -> int:
        curr = self.env.get_current_region()
        best = curr
        best_score = -1.0
        for i in range(self._num_regions):
            if exclude_current and i == curr:
                continue
            total = self._total_obs[i]
            s = self._spot_obs[i]
            score = (s + self._alpha) / (max(0, total) + 2 * self._alpha)
            if score > best_score + 1e-12:
                best = i
                best_score = score
            elif abs(score - best_score) <= 1e-12:
                # Tie-break: prefer less observed to encourage exploration
                if self._total_obs[i] < self._total_obs[best]:
                    best = i
        return best

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._lazy_init()

        region = self.env.get_current_region()
        # Update region observation stats
        self._total_obs[region] += 1
        if has_spot:
            self._spot_obs[region] += 1

        g = float(self.env.gap_seconds)
        ro = float(self.restart_overhead)
        done = float(sum(self.task_done_time)) if self.task_done_time else 0.0
        remain = float(self.task_duration) - done
        if remain <= 0.0:
            return ClusterType.NONE

        time_remain = float(self.deadline) - float(self.env.elapsed_seconds)
        if time_remain <= 0.0:
            return ClusterType.ON_DEMAND

        # Safety buffer to absorb discretization and overhead effects
        buffer = ro + 2.0 * g

        # If already committed to On-Demand, keep running OD
        if self._committed_to_od:
            return ClusterType.ON_DEMAND

        # Determine if we must commit to On-Demand to guarantee finishing
        time_needed_with_od_now = ro + remain
        if time_remain <= time_needed_with_od_now + buffer:
            self._committed_to_od = True
            return ClusterType.ON_DEMAND

        # Prefer Spot when available and not committed to OD
        if has_spot:
            # If we're currently on OD but not committed, switch back to Spot only if ample slack remains.
            if (last_cluster_type == ClusterType.ON_DEMAND or self.env.cluster_type == ClusterType.ON_DEMAND):
                slack = time_remain - (remain + buffer)
                if slack > (2.0 * ro + 2.0 * g):
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            return ClusterType.SPOT

        # Spot not available; wait if we still have slack, and switch to the best-looking region
        best_region = self._best_region(exclude_current=False)
        if best_region != region:
            self.env.switch_region(best_region)
        return ClusterType.NONE