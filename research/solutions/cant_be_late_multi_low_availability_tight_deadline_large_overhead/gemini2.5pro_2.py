import json
from argparse import Namespace
import math

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    A multi-region scheduling strategy that uses lookahead based on spot price traces
    to make decisions.

    The core logic is as follows:
    1. Pre-computation: In `solve`, load all spot availability traces and pre-compute
       the cumulative number of future available spot steps for each region. This
       allows for efficient lookups in the `_step` function.
    2. Panic Mode: At each step, calculate if committing to a reliable On-Demand
       instance is necessary to meet the deadline. If the time required to finish
       on On-Demand (including a potential restart overhead) is greater than or
       equal to the time remaining, it switches to On-Demand and stays in the
       current region to guarantee completion.
    3. Region Selection: If not in panic mode, the strategy identifies the "best"
       region to run in. The best region is defined as the one with the maximum
       number of available spot steps between the current time and the deadline.
       It will switch to this region if not already there.
    4. Cluster Type Selection: In the chosen best region, if a Spot instance is
       available at the current timestep, it is always preferred due to its lower
       cost.
    5. Slack-based Waiting: If Spot is not available in the best region, the
       strategy evaluates its "slack". Slack is the buffer time available before
       the "panic mode" is triggered. If the slack is above a certain threshold,
       it means the system can afford to wait for a Spot instance to become
       available, so it chooses `NONE`. Otherwise, to avoid risking the deadline,
       it falls back to using an `ON_DEMAND` instance to make guaranteed progress.
    """

    NAME = "lookahead_scheduler"

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

        self.spot_traces = []
        try:
            for trace_file in config["trace_files"]:
                with open(trace_file) as f:
                    trace_data = json.load(f)
                    self.spot_traces.append([bool(x) for x in trace_data])
        except (IOError, json.JSONDecodeError):
            self.spot_traces = []

        if not self.spot_traces or not self.spot_traces[0]:
            return self

        num_regions = len(self.spot_traces)
        num_steps = len(self.spot_traces[0])

        self.future_spot_steps = [[0] * (num_steps + 1)
                                  for _ in range(num_regions)]
        for r in range(num_regions):
            for t in range(num_steps - 1, -1, -1):
                self.future_spot_steps[r][t] = self.future_spot_steps[r][
                    t + 1] + int(self.spot_traces[r][t])

        self.slack_threshold = 1.5 * self.restart_overhead

        return self

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        progress = sum(self.task_done_time)
        remaining_progress = self.task_duration - progress

        if remaining_progress <= 0:
            return ClusterType.NONE

        time_now = self.env.elapsed_seconds
        time_to_deadline = self.deadline - time_now

        if time_to_deadline <= 0:
            return ClusterType.ON_DEMAND

        current_step = math.floor(time_now / self.env.gap_seconds)

        if not hasattr(
                self, 'spot_traces'
        ) or not self.spot_traces or not self.spot_traces[
                0] or current_step >= len(self.spot_traces[0]):
            time_needed_od = remaining_progress + (
                self.restart_overhead
                if last_cluster_type != ClusterType.ON_DEMAND else 0)
            if time_needed_od >= time_to_deadline:
                return ClusterType.ON_DEMAND
            return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

        time_if_commit_to_od = remaining_progress
        if last_cluster_type != ClusterType.ON_DEMAND:
            time_if_commit_to_od += self.restart_overhead

        if time_if_commit_to_od >= time_to_deadline:
            return ClusterType.ON_DEMAND

        num_regions = self.env.get_num_regions()

        deadline_step = math.floor(self.deadline / self.env.gap_seconds)
        num_trace_steps = len(self.spot_traces[0])
        effective_deadline_step = min(deadline_step, num_trace_steps)

        best_region_score = -1
        best_region = self.env.get_current_region()

        for r in range(num_regions):
            next_step = current_step + 1
            score = 0
            if next_step < effective_deadline_step:
                score = self.future_spot_steps[r][
                    next_step] - self.future_spot_steps[r][
                        effective_deadline_step]

            if score > best_region_score:
                best_region_score = score
                best_region = r

        if self.env.get_current_region() != best_region:
            self.env.switch_region(best_region)

        final_region = self.env.get_current_region()

        if self.spot_traces[final_region][current_step]:
            return ClusterType.SPOT
        else:
            time_needed_worst_case = remaining_progress + self.restart_overhead
            slack = time_to_deadline - time_needed_worst_case

            if slack > self.slack_threshold:
                return ClusterType.NONE
            else:
                return ClusterType.ON_DEMAND