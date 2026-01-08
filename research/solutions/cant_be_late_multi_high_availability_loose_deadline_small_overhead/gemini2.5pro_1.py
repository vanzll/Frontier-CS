import json
from argparse import Namespace
import collections
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    NAME = "adaptive_heuristic_strategy"

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

        self.num_regions = len(config['trace_files'])

        self.urgency_threshold = 0.95
        self.wait_slack_factor = 3.0
        self.switch_score_threshold = 0.8
        self.switch_score_gain = 0.2
        self.stability_penalty = 0.5
        self.stability_recovery = 0.01
        self.history_duration_hours = 6

        self.history_window = None
        self.spot_history = [collections.deque() for _ in range(self.num_regions)]
        self.probas = [1.0] * self.num_regions
        self.stability = [1.0] * self.num_regions
        self.observations = [0] * self.num_regions

        self.just_switched = False

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        if self.history_window is None:
            steps_per_hour = 3600.0 / self.env.gap_seconds
            self.history_window = int(self.history_duration_hours * steps_per_hour)
            for i in range(self.num_regions):
                self.spot_history[i] = collections.deque(maxlen=self.history_window)

        remaining_work = self.task_duration - sum(self.task_done_time)
        if remaining_work <= 0:
            return ClusterType.NONE

        current_region = self.env.get_current_region()

        was_preempted = (last_cluster_type == ClusterType.SPOT and
                         self.remaining_restart_overhead > 0 and
                         not self.just_switched)

        if was_preempted:
            self.stability[current_region] *= self.stability_penalty
        elif last_cluster_type != ClusterType.NONE and not self.just_switched:
            self.stability[current_region] = min(1.0, self.stability[current_region] + self.stability_recovery)

        self.just_switched = False

        self.spot_history[current_region].append(1 if has_spot else 0)
        self.observations[current_region] += 1
        self.probas[current_region] = sum(self.spot_history[current_region]) / len(self.spot_history[current_region])

        time_left = self.deadline - self.env.elapsed_seconds
        time_needed_on_demand = remaining_work + self.restart_overhead

        if time_needed_on_demand >= time_left or \
           time_needed_on_demand >= time_left * self.urgency_threshold:
            return ClusterType.ON_DEMAND

        if has_spot:
            return ClusterType.SPOT
        else:
            region_scores = [self.probas[r] * self.stability[r] for r in range(self.num_regions)]
            current_score = region_scores[current_region]

            best_score = -1.0
            best_region = -1
            if self.observations[current_region] >= 3:
                for r in range(self.num_regions):
                    if r == current_region: continue
                    if self.observations[r] >= 3 or self.observations[r] == 0:
                        if region_scores[r] > best_score:
                            best_score = region_scores[r]
                            best_region = r

            if (best_region != -1 and
                    best_score > self.switch_score_threshold and
                    best_score > current_score + self.switch_score_gain):
                self.env.switch_region(best_region)
                self.just_switched = True
                return ClusterType.ON_DEMAND
            else:
                slack = time_left - time_needed_on_demand
                if slack > self.restart_overhead * self.wait_slack_factor:
                    return ClusterType.NONE
                else:
                    return ClusterType.ON_DEMAND