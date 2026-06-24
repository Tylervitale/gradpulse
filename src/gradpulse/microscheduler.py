"""Micro-scheduling module."""
from __future__ import annotations

import collections
import copy
from typing import Dict, List, Optional, Tuple

from gradpulse.scheduling import DependencyGraph, OperationNode


class Microscheduler:
    """
    Schedules a directed acyclic graph of operations onto a continuous time grid.

    Tightly packs analog pulses onto a hardware timeline, accounting for pulse
    ring-down times, buffer constraints, and channel limitations.
    Uses an As-Soon-As-Possible (ASAP) algorithm with constraints.
    """

    def __init__(self, dt_ns: float = 1.0):
        """
        Args:
            dt_ns: The resolution of the time grid in nanoseconds.
        """
        self.dt_ns = dt_ns

        # Channel constraint margins
        # E.g., {'q0_drive': 2.0} means channels 'q0_drive' must be idle for 2ns between operations
        self.channel_margins_ns: Dict[str, float] = collections.defaultdict(float)

    def add_channel_margin(self, channel: str, margin_ns: float):
        """
        Add a required buffer margin after operations on a specific channel.

        Args:
            channel: The channel name.
            margin_ns: The margin in nanoseconds.
        """
        self.channel_margins_ns[channel] = margin_ns

    def schedule(self, graph: DependencyGraph) -> Dict[str, float]:
        """
        Schedule the operations in the graph.

        Args:
            graph: The DependencyGraph containing operations to schedule.

        Returns:
            A dictionary mapping operation IDs to their scheduled start times in nanoseconds.
        """
        # Get topological order from the graph
        order = graph.get_topological_order()

        # Track the time when each channel is free again
        channel_free_time: Dict[str, float] = collections.defaultdict(float)

        # Track the completion time of each operation
        op_completion_time: Dict[str, float] = {}

        # Track the start time of each operation
        schedule: Dict[str, float] = {}

        for op_id in order:
            node = graph.nodes[op_id]

            # 1. Dependency constraints: Must start after all predecessors finish
            earliest_start_deps = 0.0

            # Since graph.edges is A -> B, to find predecessors of B we have to check all A.
            # Alternatively, we could compute predecessors on the fly.
            predecessors = [from_id for from_id, to_ids in graph.edges.items() if op_id in to_ids]

            for pred_id in predecessors:
                if pred_id in op_completion_time:
                    earliest_start_deps = max(earliest_start_deps, op_completion_time[pred_id])

            # 2. Channel constraints: Must start after all required channels are free
            earliest_start_channels = 0.0
            for channel in node.channels:
                earliest_start_channels = max(earliest_start_channels, channel_free_time[channel])

            # Start time is the maximum of dependency constraints and channel constraints
            start_time = max(earliest_start_deps, earliest_start_channels)

            # Align start time to grid dt_ns (round up to nearest multiple of dt_ns)
            # Add small epsilon to handle float precision issues
            grid_steps = int(start_time / self.dt_ns + 1e-9)
            if grid_steps * self.dt_ns < start_time - 1e-9:
                grid_steps += 1
            start_time = grid_steps * self.dt_ns

            schedule[op_id] = start_time

            # Compute completion time
            end_time = start_time + node.duration_ns
            op_completion_time[op_id] = end_time

            # Update channel free times, including required margins
            for channel in node.channels:
                margin = self.channel_margins_ns[channel]
                channel_free_time[channel] = end_time + margin

        return schedule
