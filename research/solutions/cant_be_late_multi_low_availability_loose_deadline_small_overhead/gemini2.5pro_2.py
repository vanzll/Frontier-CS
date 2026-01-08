import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """
    An adaptive, multi-region scheduling strategy that prioritizes finishing
    before the deadline while minimizing cost.

    The core of the strategy is based on the concept of "slack time", defined as
    the total time remaining until the deadline minus the amount of work left to do.
    The behavior adapts based on the amount of available slack:

    1.  CRITICAL MODE: When slack is critically low (e.g., less than 1 hour),
        the strategy enters a failsafe mode. It exclusively uses On-Demand instances
        to guarantee progress and avoids switching regions to prevent any
        time loss from restart overheads. This minimizes the risk of missing
        the deadline, which incurs a severe penalty.

    2.  LEISURE MODE: When there is a large amount of slack (e.g., more than 18 hours),
        the strategy prioritizes cost savings. It will use cheap Spot instances when
        available. If Spot is unavailable, it will choose to wait (NONE) rather
        than paying for expensive On-Demand instances, as there is ample time to
        complete the task later.

    3.  NORMAL MODE: In the intermediate state, the strategy balances cost and
        progress.
        - It always prefers Spot instances if they are available.
        - If Spot is not available, it uses On-Demand to avoid falling behind schedule.
        - It leverages pre-loaded historical spot-availability traces to score each
          region. If the current region shows poor historical availability and
          another region appears significantly better, it will switch regions to
          improve long-term access to cheaper Spot resources.

    This three-tiered, slack-based approach allows the strategy to be aggressive
    on cost-saving when risk is low and conservative about deadlines when risk is high.
    """

    NAME = "my_strategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialize the solution from the provided spec file. This involves
        setting up task parameters and pre-loading spot availability traces
        for all regions.
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

        self.spot_traces = []
        for trace_file in config.get("trace_files", []):
            try:
                with open(trace_file) as tf:
                    # Load trace data, assuming one integer (0 or 1) per line.
                    trace = [int(line.strip()) for line in tf if line.strip()]
                    self.spot_traces.append(trace)
            except (IOError, ValueError):
                # If a trace file is missing or corrupt, treat it as an empty trace.
                self.spot_traces.append([])

        # --- Strategy Parameters ---
        # These values are tuned for the given problem specifications.

        # Failsafe buffer: Switch to On-Demand-only when slack is below this.
        self.criticality_buffer_seconds = 1 * 3600  # 1 hour

        # Leisure threshold: If slack is above this, wait for Spot instead of using On-Demand.
        self.leisure_slack_seconds = 18 * 3600  # 18 hours

        # Lookahead window for scoring regions based on historical data.
        self.lookahead_window_hours = 1  # 1 hour

        # Thresholds for making a region switch decision.
        self.region_switch_low_score = 0.1
        self.region_switch_high_score = 0.9

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Makes a scheduling decision for the next time step.
        """
        # 1. Calculate current state
        current_time = self.env.elapsed_seconds
        work_done = sum(self.task_done_time)
        work_left = self.task_duration - work_done

        # If the task is done, stop incurring costs.
        if work_left <= 0:
            return ClusterType.NONE

        time_left = self.deadline - current_time

        # 2. Criticality Check: Failsafe to guarantee deadline completion.
        if time_left <= work_left + self.criticality_buffer_seconds:
            return ClusterType.ON_DEMAND

        # 3. Region Selection
        current_step = int(round(current_time / self.env.gap_seconds))
        num_regions = self.env.get_num_regions()
        current_region = self.env.get_current_region()

        if num_regions > 1 and self.spot_traces:
            lookahead_steps = int(round(self.lookahead_window_hours * 3600 / self.env.gap_seconds))
            if lookahead_steps == 0: lookahead_steps = 1

            scores = []
            for r in range(num_regions):
                trace = self.spot_traces[r]
                if current_step >= len(trace):
                    scores.append(0.0)
                    continue

                end_step = min(current_step + lookahead_steps, len(trace))
                trace_slice = trace[current_step:end_step]
                
                score = sum(trace_slice) / len(trace_slice) if trace_slice else 0.0
                scores.append(score)
            
            if scores:
                best_region = max(range(num_regions), key=scores.__getitem__)

                if (scores[current_region] < self.region_switch_low_score and
                    scores[best_region] > self.region_switch_high_score and
                    best_region != current_region):
                    self.env.switch_region(best_region)
                    current_region = best_region

        # 4. Cluster Type Selection
        # Use our pre-loaded trace data to check for spot availability, as it allows
        # us to make a decision for the new region immediately after switching.
        spot_available_now = False
        if (self.spot_traces and 
            0 <= current_region < len(self.spot_traces) and 
            current_step < len(self.spot_traces[current_region])):
             spot_available_now = bool(self.spot_traces[current_region][current_step])

        if spot_available_now:
            return ClusterType.SPOT

        # Spot is not available. Decide between On-Demand and None based on slack.
        slack_seconds = time_left - work_left
        if slack_seconds > self.leisure_slack_seconds:
            # High slack: save money and wait for Spot.
            return ClusterType.NONE
        else:
            # Lower slack: spend money to make progress.
            return ClusterType.ON_DEMAND