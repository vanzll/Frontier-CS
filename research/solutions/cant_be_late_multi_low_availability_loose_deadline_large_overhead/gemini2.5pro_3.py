import json
from argparse import Namespace
import numpy as np
import math

try:
    from sky_spot.strategies.multi_strategy import MultiRegionStrategy
    from sky_spot.utils import ClusterType
except ImportError:
    # Dummy classes for local testing if the environment is not available
    class MultiRegionStrategy:
        def __init__(self, args):
            self.task_duration = args.task_duration_hours[0] * 3600.0
            self.deadline = args.deadline_hours * 3600.0
            self.restart_overhead = args.restart_overhead_hours[0] * 3600.0
            self.task_done_time = []
            self.remaining_restart_overhead = 0.0
            class DummyEnv:
                def __init__(self):
                    self.gap_seconds = 3600.0
                    self.elapsed_seconds = 0.0
                    self._current_region = 0
                def get_current_region(self): return self._current_region
                def get_num_regions(self): return 1
                def switch_region(self, idx): self._current_region = idx
            self.env = DummyEnv()
        def solve(self, spec_path: str) -> "MultiRegionStrategy": pass
        def _step(self, last_cluster_type, has_spot) -> "ClusterType": pass

    class ClusterType:
        SPOT = "SPOT"
        ON_DEMAND = "ON_DEMAND"
        NONE = "NONE"


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "my_strategy"

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

        self.avg_avail = []
        if 'trace_files' in config:
            for trace_file in config['trace_files']:
                try:
                    with open(trace_file, 'r') as tf:
                        trace_data = [int(line.strip()) for line in tf if line.strip()]
                        if not trace_data:
                            self.avg_avail.append(0.0)
                        else:
                            self.avg_avail.append(np.mean(trace_data))
                except (IOError, ValueError):
                    self.avg_avail.append(0.0)

        num_regions = len(self.avg_avail)
        if num_regions > 0:
            self.best_region_order = sorted(range(num_regions), key=lambda r: self.avg_avail[r], reverse=True)
            self.best_region = self.best_region_order[0]
        else:
            self.best_region_order = [0]
            self.best_region = 0

        if self.env.gap_seconds > 0:
            t_needed_bailout_initial = math.ceil(
                (self.task_duration + self.restart_overhead) / self.env.gap_seconds
            ) * self.env.gap_seconds
        else:
            t_needed_bailout_initial = float('inf')

        self.initial_slack = self.deadline - t_needed_bailout_initial
        
        if self.initial_slack <= 0:
            self.initial_slack = 1.0

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        w_done = sum(self.task_done_time)
        w_rem = self.task_duration - w_done

        if w_rem <= 0:
            return ClusterType.NONE

        work_plus_overhead = w_rem + self.restart_overhead
        
        if self.env.gap_seconds > 0:
            num_steps_needed = math.ceil(work_plus_overhead / self.env.gap_seconds)
            t_needed_bailout = num_steps_needed * self.env.gap_seconds
        else:
            t_needed_bailout = float('inf')

        is_critical = (self.env.elapsed_seconds + t_needed_bailout >= self.deadline)

        if is_critical:
            return ClusterType.ON_DEMAND
        else:
            if has_spot:
                return ClusterType.SPOT
            else:
                current_region = self.env.get_current_region()
                
                if self.env.get_num_regions() > 1 and current_region != self.best_region:
                    self.env.switch_region(self.best_region)
                    return ClusterType.NONE
                else:
                    slack = self.deadline - (self.env.elapsed_seconds + t_needed_bailout)
                    current_slack_fraction = slack / self.initial_slack
                    
                    if current_slack_fraction > 0.25:
                        return ClusterType.NONE
                    else:
                        return ClusterType.ON_DEMAND