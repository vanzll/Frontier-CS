import json
from argparse import Namespace
import bisect
import csv

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    An offline heuristic strategy for the Cant-Be-Late Multi-Region Scheduling Problem.

    This strategy works in two phases:
    1. Pre-computation (`solve`): It reads all spot availability traces for all regions
       and builds a unified timeline. It pre-calculates a lookup table to quickly find
       the next available spot instance time for any region at any point in time. This
       allows for efficient decision-making in the execution phase.

    2. Step-by-step decision (`_step`): At each time step, the strategy first performs a
       criticality check. If finishing the task on-time is at risk (i.e., the time
       required to finish with guaranteed On-Demand instances, including worst-case
       overheads, exceeds the time left until the deadline), it will choose On-Demand
       to ensure completion.

       If not in a critical situation, the strategy prioritizes the cheapest option. If
       Spot instances are available in the current region, it uses them. If not, it
       evaluates two alternatives to using expensive On-Demand:
       a) Waiting in the current region for Spot to become available.
       b) Switching to another region where Spot is available sooner.

       It calculates the "time cost" (slack time consumed) for both waiting and
       switching. The option with the lower time cost is chosen, provided there is
       enough slack available to absorb this cost. If neither waiting nor switching is
       affordable within the available slack, it falls back to using On-Demand to make
       progress.
    """

    NAME = "offline_heuristic"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from spec_path config.
        """
        with open(spec_path) as f:
            config = json.load(f)

        args = Namespace(
            deadline_hours=float(config["deadline"]),
            task_duration_hours=[float(config["duration"])],
            restart_overhead_hours=[float(config["overhead"])],
            inter_task_overhead=[0.0],
        )
        super().__init__(args)

        self.num_regions = len(config["trace_files"])
        raw_traces = [[] for _ in range(self.num_regions)]
        all_timestamps = set()

        for i, trace_file in enumerate(config["trace_files"]):
            with open(trace_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    ts = float(row[0])
                    avail = bool(int(row[1]))
                    raw_traces[i].append((ts, avail))
                    all_timestamps.add(ts)
        
        self.sorted_timestamps = sorted(list(all_timestamps))
        time_to_idx = {ts: i for i, ts in enumerate(self.sorted_timestamps)}
        
        num_timesteps = len(self.sorted_timestamps)
        self.spot_availability = [[False] * num_timesteps for _ in range(self.num_regions)]
        
        for r in range(self.num_regions):
            for ts, avail in raw_traces[r]:
                if avail:
                    idx = time_to_idx.get(ts)
                    if idx is not None:
                        self.spot_availability[r][idx] = True
        
        self.next_spot_lookup = [[-1] * num_timesteps for _ in range(self.num_regions)]
        self.infinity_idx = num_timesteps
        for r in range(self.num_regions):
            next_spot = self.infinity_idx
            for k in range(num_timesteps - 1, -1, -1):
                if self.spot_availability[r][k]:
                    next_spot = k
                self.next_spot_lookup[r][k] = next_spot
        
        self.infinity = float('inf')
        self.work_done = 0.0
        self.last_len_task_done_time = 0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        if len(self.task_done_time) > self.last_len_task_done_time:
             self.work_done += sum(self.task_done_time[self.last_len_task_done_time:])
             self.last_len_task_done_time = len(self.task_done_time)
        
        remaining_work = self.task_duration - self.work_done
        if remaining_work <= 1e-9:
            return ClusterType.NONE
            
        t = self.env.elapsed_seconds
        time_to_deadline = self.deadline - t
        
        time_needed_if_preempted = remaining_work + self.restart_overhead
        if time_needed_if_preempted >= time_to_deadline:
             return ClusterType.ON_DEMAND
        
        current_region = self.env.get_current_region()
        
        if has_spot:
            return ClusterType.SPOT
        
        k = bisect.bisect_left(self.sorted_timestamps, t)
        if k >= len(self.sorted_timestamps) or abs(self.sorted_timestamps[k] - t) > 1e-6:
             if k > 0 and abs(self.sorted_timestamps[k-1] - t) < 1e-6:
                 k = k - 1
             else:
                 return ClusterType.ON_DEMAND

        safe_slack = time_to_deadline - time_needed_if_preempted

        k_next_current = self.next_spot_lookup[current_region][k]
        if k_next_current < self.infinity_idx:
            wait_time_cost = self.sorted_timestamps[k_next_current] - t
        else:
            wait_time_cost = self.infinity

        best_switch_region = -1
        min_switch_time_cost = self.infinity
        for r_i in range(self.num_regions):
            if r_i == current_region:
                continue
            
            k_next_ri = self.next_spot_lookup[r_i][k]
            if k_next_ri < self.infinity_idx:
                switch_time_cost = self.restart_overhead + (self.sorted_timestamps[k_next_ri] - t)
                if switch_time_cost < min_switch_time_cost:
                    min_switch_time_cost = switch_time_cost
                    best_switch_region = r_i
        
        min_time_cost = min(wait_time_cost, min_switch_time_cost)
        if min_time_cost > safe_slack or min_time_cost == self.infinity:
            return ClusterType.ON_DEMAND
        
        if wait_time_cost <= min_switch_time_cost:
            return ClusterType.NONE
        else:
            self.env.switch_region(best_switch_region)
            if self.spot_availability[best_switch_region][k]:
                return ClusterType.SPOT
            else:
                return ClusterType.NONE