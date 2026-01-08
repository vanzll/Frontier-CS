import json
import math
from argparse import Namespace
from typing import Any, List, Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "multi_region_safe_spot_v1"

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

        self._initialized = False
        self._force_on_demand = False

        self._cached_done = 0.0
        self._cached_td_len = 0

        self._region_stats = None  # type: Optional[List[List[float]]]
        self._no_spot_steps = 0
        self._last_switch_elapsed = -1e30

        return self

    def _scalar_attr(self, x: Any) -> float:
        if isinstance(x, (list, tuple)):
            if not x:
                return 0.0
            return float(x[0])
        return float(x)

    def _get_task_done_list(self) -> List[float]:
        td = self.task_done_time
        if isinstance(td, (list, tuple)) and td and isinstance(td[0], (list, tuple)):
            td = td[0]
        return td if isinstance(td, list) else list(td)

    def _update_cached_done(self) -> float:
        td = self._get_task_done_list()
        n = len(td)
        if n < self._cached_td_len:
            self._cached_done = float(sum(td))
            self._cached_td_len = n
            return self._cached_done
        if n > self._cached_td_len:
            self._cached_done += float(sum(td[self._cached_td_len : n]))
            self._cached_td_len = n
        return self._cached_done

    def _init_if_needed(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        num_regions = int(self.env.get_num_regions())
        self._region_stats = [[0.0, 0.0] for _ in range(num_regions)]  # [up, total]
        self._cached_done = float(sum(self._get_task_done_list()))
        self._cached_td_len = len(self._get_task_done_list())

    def _choose_switch_region(self, cur_region: int) -> Optional[int]:
        if self._region_stats is None:
            return None
        best = None
        best_key = None
        for r, (up, total) in enumerate(self._region_stats):
            if r == cur_region:
                continue
            if total <= 0.0:
                rate = 0.55
            else:
                rate = up / total
            key = (rate, -total)
            if best is None or key > best_key:
                best = r
                best_key = key
        return best

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._init_if_needed()

        env = self.env
        gap = float(getattr(env, "gap_seconds", 1.0))
        elapsed = float(getattr(env, "elapsed_seconds", 0.0))

        task_duration = self._scalar_attr(self.task_duration)
        deadline = self._scalar_attr(self.deadline)
        restart_overhead = self._scalar_attr(self.restart_overhead)

        done = self._update_cached_done()
        work_remaining = task_duration - done
        if work_remaining <= 0.0:
            return ClusterType.NONE

        time_remaining = deadline - elapsed
        if time_remaining <= 0.0:
            return ClusterType.NONE

        rro = getattr(self, "remaining_restart_overhead", 0.0)
        try:
            pending_overhead = self._scalar_attr(rro)
        except Exception:
            pending_overhead = 0.0
        if pending_overhead < 0.0:
            pending_overhead = 0.0

        cur_region = int(env.get_current_region())
        if self._region_stats is not None and 0 <= cur_region < len(self._region_stats):
            self._region_stats[cur_region][1] += 1.0
            if has_spot:
                self._region_stats[cur_region][0] += 1.0

        safety_margin = max(2.0 * gap, restart_overhead)
        startup_overhead = 0.0 if last_cluster_type == ClusterType.ON_DEMAND else restart_overhead
        worst_overhead_to_start_od = max(pending_overhead, startup_overhead)
        required_time_if_od_now = work_remaining + worst_overhead_to_start_od

        if self._force_on_demand or time_remaining <= required_time_if_od_now + safety_margin:
            self._force_on_demand = True
            return ClusterType.ON_DEMAND

        if has_spot:
            self._no_spot_steps = 0
            return ClusterType.SPOT

        self._no_spot_steps += 1

        patience_seconds = max(5.0 * restart_overhead, 600.0)
        patience_steps = int(math.ceil(patience_seconds / max(gap, 1e-9)))
        if patience_steps < 1:
            patience_steps = 1

        cooldown_seconds = max(2.0 * restart_overhead, 300.0)
        can_switch = (
            int(env.get_num_regions()) > 1
            and pending_overhead <= 0.0
            and (elapsed - self._last_switch_elapsed) >= cooldown_seconds
            and self._no_spot_steps >= patience_steps
        )

        if can_switch:
            target = self._choose_switch_region(cur_region)
            if target is not None and target != cur_region:
                env.switch_region(int(target))
                self._last_switch_elapsed = elapsed
                self._no_spot_steps = 0
                return ClusterType.NONE

        return ClusterType.NONE