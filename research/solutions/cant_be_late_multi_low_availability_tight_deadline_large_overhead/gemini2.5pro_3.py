import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_ema_slack"

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

        self.num_regions = self.env.get_num_regions()
        
        self.region_stats = [
            {'probes': 0, 'ema': 0.5} for _ in range(self.num_regions)
        ]

        self.cached_work_done = 0.0
        self.cached_task_done_len = 0

        self.ema_alpha = 0.1
        self.probes_before_trusting_ema = 3
        self.switch_ema_threshold = 0.25
        self.switch_slack_factor = 1.5
        self.pause_slack_factor = 2.0
        self.pause_ema_threshold = 0.5
        self.on_demand_safety_margin_seconds = self.env.gap_seconds

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if len(self.task_done_time) > self.cached_task_done_len:
            new_elements = self.task_done_time[self.cached_task_done_len:]
            self.cached_work_done += sum(new_elements)
            self.cached_task_done_len = len(self.task_done_time)

        remaining_work = self.task_duration - self.cached_work_done
        if remaining_work <= 0:
            return ClusterType.NONE

        time_to_deadline = self.deadline - self.env.elapsed_seconds
        current_region = self.env.get_current_region()

        stats = self.region_stats[current_region]
        stats['probes'] += 1
        new_ema = self.ema_alpha * float(has_spot) + (1 - self.ema_alpha) * stats['ema']
        stats['ema'] = new_ema

        time_needed_for_od = remaining_work + self.remaining_restart_overhead
        if time_to_deadline <= time_needed_for_od + self.on_demand_safety_margin_seconds:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT

        slack = time_to_deadline - time_needed_for_od

        if slack > self.restart_overhead * self.switch_slack_factor:
            best_explored_region = -1
            best_explored_ema = -1.0
            unexplored_regions = []

            for r in range(self.num_regions):
                if r == current_region:
                    continue
                r_stats = self.region_stats[r]
                if r_stats['probes'] > 0:
                    if r_stats['ema'] > best_explored_ema:
                        best_explored_ema = r_stats['ema']
                        best_explored_region = r
                else:
                    unexplored_regions.append(r)

            switch_candidate = -1
            current_ema = self.region_stats[current_region]['ema']

            if (best_explored_region != -1 and
                    self.region_stats[best_explored_region]['probes'] >= self.probes_before_trusting_ema and
                    best_explored_ema > current_ema + self.switch_ema_threshold):
                switch_candidate = best_explored_region
            elif unexplored_regions and current_ema < 0.2:
                 switch_candidate = unexplored_regions[0]

            if switch_candidate != -1:
                self.env.switch_region(switch_candidate)
                return ClusterType.NONE

        if (slack > self.restart_overhead * self.pause_slack_factor and
                self.region_stats[current_region]['ema'] > self.pause_ema_threshold):
            return ClusterType.NONE
        
        return ClusterType.ON_DEMAND