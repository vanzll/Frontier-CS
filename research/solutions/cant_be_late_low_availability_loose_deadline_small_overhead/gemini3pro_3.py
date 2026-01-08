from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType

class Solution(Strategy):
    NAME = "SlackAwareStrategy"

    def solve(self, spec_path: str) -> "Solution":
        """
        Initialization step. Returns self.
        """
        return self

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """
        Decides the next cluster type based on slack and spot availability.
        Implements a Least-Slack-First approach with cost optimization.
        """
        # 1. Calculate current state
        # Sum of all completed segments
        work_done = sum(self.task_done_time)
        work_remaining = self.task_duration - work_done
        
        # If work is effectively done, we can stop
        if work_remaining <= 1e-6:
            return ClusterType.NONE

        current_time = self.env.elapsed_seconds
        time_left = self.deadline - current_time
        
        overhead = self.restart_overhead
        gap = self.env.gap_seconds
        
        # 2. Calculate Slack
        # Calculate the time required to finish if we commit to On-Demand immediately.
        # If we are already on OD, we continue seamlessly.
        # If we are not on OD, we incur the restart overhead to switch/start.
        time_to_finish_od = work_remaining
        if last_cluster_type != ClusterType.ON_DEMAND:
            time_to_finish_od += overhead
            
        slack = time_left - time_to_finish_od
        
        # 3. Define Safety Thresholds
        # safety_buffer: Ensure we have enough margin for discrete time steps.
        # If slack < safety_buffer, we are in the 'Danger Zone' and must use OD.
        safety_buffer = 2.0 * gap
        
        # switch_threshold: The slack required to justify leaving OD or not starting OD.
        # If we leave OD (or don't start it), we risk paying 'overhead' later to start it.
        # We need enough slack to absorb that potential future penalty.
        switch_threshold = overhead + safety_buffer
        
        # 4. Decision Logic
        if has_spot:
            # Spot instances are available
            if last_cluster_type == ClusterType.ON_DEMAND:
                # We are currently running OD.
                # Only switch to Spot if we have enough slack to afford the switch overhead
                # and potential switch-back cost.
                if slack > switch_threshold:
                    return ClusterType.SPOT
                else:
                    # Risk of missing deadline is too high to switch context
                    return ClusterType.ON_DEMAND
            else:
                # We are currently on SPOT or NONE
                # Prefer Spot, unless we are critically close to the deadline.
                # If slack is critical, OD is safer because Spot can be interrupted,
                # wasting the step and reducing slack further.
                if slack < safety_buffer:
                    return ClusterType.ON_DEMAND
                return ClusterType.SPOT
        else:
            # Spot instances are NOT available
            if last_cluster_type == ClusterType.ON_DEMAND:
                # We are currently running OD.
                # If we have plenty of slack, we can stop and wait for Spot to return (Wait Strategy).
                # This saves money but burns slack.
                if slack > switch_threshold:
                    return ClusterType.NONE
                else:
                    # Slack is tight, must keep running OD
                    return ClusterType.ON_DEMAND
            else:
                # We are currently on SPOT (failed) or NONE
                # If slack is critical, forced to start OD.
                # Otherwise, wait for Spot (NONE).
                if slack < safety_buffer:
                    return ClusterType.ON_DEMAND
                else:
                    return ClusterType.NONE

    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)