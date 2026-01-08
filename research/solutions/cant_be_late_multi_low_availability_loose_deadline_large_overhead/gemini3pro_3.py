import json
from argparse import Namespace

from sky_spot.strategies.multi_strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType


class Solution(MultiRegionStrategy):
    """Your multi-region scheduling strategy."""

    NAME = "robust_hybrid_strategy"

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

        # Attempt to load and parse trace files for smarter decisions
        self.traces = []
        self.trace_loaded = False
        try:
            trace_paths = config.get("trace_files", [])
            for p in trace_paths:
                # Heuristic parsing to handle common trace formats (CSV, single column, etc.)
                data = []
                with open(p, 'r') as tf:
                    for line in tf:
                        line_clean = line.strip().lower()
                        # Skip comments or obvious headers
                        if not line_clean or line_clean.startswith('#') or line_clean.startswith('time'):
                            continue
                        
                        # Check for boolean text
                        if 'true' in line_clean:
                            data.append(True)
                        elif 'false' in line_clean:
                            data.append(False)
                        else:
                            # Check for numeric indicators (1 or 0)
                            # Handle CSV by replacing commas with spaces
                            parts = line_clean.replace(',', ' ').split()
                            val_found = False
                            for part in parts:
                                if part == '1':
                                    data.append(True)
                                    val_found = True
                                    break
                                elif part == '0':
                                    data.append(False)
                                    val_found = True
                                    break
                            
                            # If no clear boolean found, we skip the line (might be just timestamp)
                            pass
                self.traces.append(data)
            
            # Verify we loaded something for each region
            if self.traces and len(self.traces) > 0:
                self.trace_loaded = True
        except Exception:
            self.trace_loaded = False

        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decide next action based on current state.
        """
        # 1. Calculate Current State Variables
        elapsed = self.env.elapsed_seconds
        done = sum(self.task_done_time)
        total_needed = self.task_duration
        remaining_work = total_needed - done
        
        deadline = self.deadline
        time_left = deadline - elapsed
        
        # Slack: The amount of time we can afford to waste (wait or overhead)
        slack = time_left - remaining_work
        
        overhead = self.restart_overhead
        
        # 2. Panic Mode (Safety Net)
        # If slack is critically low, we must switch to On-Demand to guarantee completion.
        # Threshold: Enough to cover one restart overhead plus a safety buffer.
        # If slack drops below this, we stop gambling with Spot.
        if slack < 2.0 * overhead:
            return ClusterType.ON_DEMAND

        # 3. Trace Consistency Validation
        # If we rely on traces, verify they match the current reality.
        current_region = self.env.get_current_region()
        step_idx = int(elapsed // self.env.gap_seconds)
        
        if self.trace_loaded:
            if current_region < len(self.traces) and step_idx < len(self.traces[current_region]):
                predicted_spot = self.traces[current_region][step_idx]
                if predicted_spot != has_spot:
                    # Prediction failed, traces are unreliable. Fallback to probing.
                    self.trace_loaded = False
            else:
                # Index out of bounds, disable traces.
                self.trace_loaded = False

        # 4. Decision Logic
        
        # Priority 1: If current region has Spot, use it.
        # This is the most efficient action (no region switch overhead).
        if has_spot:
            return ClusterType.SPOT
            
        # Priority 2: Current region has NO Spot. Find a better region.
        
        if self.trace_loaded:
            # Use traces to find a region that currently has Spot available
            best_region = -1
            
            for r_idx, trace in enumerate(self.traces):
                if r_idx == current_region:
                    continue
                if step_idx < len(trace) and trace[step_idx]:
                    best_region = r_idx
                    break
            
            if best_region != -1:
                # Switch to the region we know is available
                self.env.switch_region(best_region)
                # Since we trust the trace (verified above), we return SPOT immediately.
                return ClusterType.SPOT
            else:
                # Traces say NO region has spot.
                # Since we have slack (Panic Check passed), we wait.
                # Returning NONE avoids paying OD cost.
                return ClusterType.NONE
                
        else:
            # Fallback: Probing Strategy
            # We don't know which region has spot.
            # We switch to the next region and return NONE to "probe" it in the next step.
            # We cannot return SPOT here because we aren't sure, and that would error.
            num_regions = self.env.get_num_regions()
            next_region = (current_region + 1) % num_regions
            self.env.switch_region(next_region)
            return ClusterType.NONE