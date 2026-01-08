import json
from argparse import Namespace
from typing import Optional

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "cant_be_late_multiregion_v1"

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

        self._inited = False
        self._done_sum = 0.0
        self._done_len = 0

        self._committed_ondemand = False

        return self

    def _ensure_init(self) -> None:
        if self._inited:
            return

        n = int(self.env.get_num_regions())
        self._n_regions = n

        self._avail = [0] * n
        self._total = [0] * n
        self._last_fail_time = [-1e30] * n
        self._last_visit_time = [-1e30] * n

        self._no_spot_streak = 0
        self._spot_up_streak = 0

        self._probes_in_outage = 0
        self._last_switch_time = -1e30
        self._rr_ptr = 0

        gap = float(self.env.gap_seconds)
        ro = float(self.restart_overhead)

        self._gap = gap
        self._ro = ro

        self._commit_slack = gap + 4.0 * ro
        self._wait_slack = gap + 2.0 * ro

        self._switch_after_seconds = max(30.0 * 60.0, 3.0 * gap)
        self._switch_slack_min = max(2.0 * gap + 2.0 * ro, 30.0 * 60.0)
        self._min_switch_interval = max(10.0 * 60.0, 2.0 * gap)
        self._fail_cooldown = max(15.0 * 60.0, 2.0 * gap)

        self._max_probes = 0
        if n > 1:
            self._max_probes = min(3, n - 1)

        self._spot_confirm_steps = 2

        self._inited = True

    def _update_done_sum(self) -> float:
        tdt = self.task_done_time
        l = len(tdt)
        if l != self._done_len:
            self._done_sum += sum(tdt[self._done_len : l])
            self._done_len = l
        return self._done_sum

    def _choose_region(self, current_region: int, now: float) -> int:
        n = self._n_regions
        if n <= 1:
            return current_region

        best_region: Optional[int] = None
        best_score = -1e18

        # First pass: respect fail cooldown.
        for pass_idx in (0, 1):
            for r in range(n):
                if r == current_region:
                    continue
                if pass_idx == 0 and (now - self._last_fail_time[r]) < self._fail_cooldown:
                    continue

                tot = self._total[r]
                if tot <= 0:
                    score = 0.5 + 0.02  # slight bonus to explore unseen regions
                else:
                    score = (self._avail[r] + 1.0) / (tot + 2.0)

                # Prefer regions visited recently with good availability.
                last_visit = self._last_visit_time[r]
                if last_visit > -1e20:
                    age = now - last_visit
                    score += 0.005 * (1.0 / (1.0 + age / (6.0 * 3600.0)))

                # Tie-break with round-robin preference.
                rr_bonus = 0.0
                if n > 0:
                    dist = (r - self._rr_ptr) % n
                    rr_bonus = -1e-6 * dist
                score += rr_bonus

                if score > best_score:
                    best_score = score
                    best_region = r

            if best_region is not None:
                break

        if best_region is None:
            best_region = (current_region + 1) % n

        self._rr_ptr = (best_region + 1) % n
        return best_region

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        self._ensure_init()

        now = float(self.env.elapsed_seconds)
        done = self._update_done_sum()

        remaining_work = float(self.task_duration) - done
        if remaining_work <= 0.0:
            return ClusterType.NONE

        time_left = float(self.deadline) - now
        slack = time_left - remaining_work

        current_region = int(self.env.get_current_region())

        self._total[current_region] += 1
        self._last_visit_time[current_region] = now
        if has_spot:
            self._avail[current_region] += 1
            self._no_spot_streak = 0
            self._spot_up_streak += 1
        else:
            self._last_fail_time[current_region] = now
            self._no_spot_streak += 1
            self._spot_up_streak = 0

        if slack <= self._commit_slack:
            self._committed_ondemand = True

        if self._committed_ondemand:
            return ClusterType.ON_DEMAND

        if last_cluster_type == ClusterType.ON_DEMAND:
            if has_spot and self._spot_up_streak >= self._spot_confirm_steps and slack >= (2.0 * self._gap + 2.0 * self._ro):
                return ClusterType.SPOT
            return ClusterType.ON_DEMAND

        if has_spot:
            self._probes_in_outage = 0
            return ClusterType.SPOT

        if slack <= self._wait_slack:
            self._probes_in_outage = 0
            return ClusterType.ON_DEMAND

        outage_seconds = self._no_spot_streak * self._gap
        can_switch = (
            self._n_regions > 1
            and self._max_probes > 0
            and self._probes_in_outage < self._max_probes
            and outage_seconds >= self._switch_after_seconds
            and slack >= self._switch_slack_min
            and (now - self._last_switch_time) >= self._min_switch_interval
        )

        if can_switch:
            new_region = self._choose_region(current_region, now)
            if new_region != current_region:
                self.env.switch_region(new_region)
                self._last_switch_time = now
                self._probes_in_outage += 1
                self._no_spot_streak = 0

        return ClusterType.NONE