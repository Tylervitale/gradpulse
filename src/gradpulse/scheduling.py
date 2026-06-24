from __future__ import annotations

import collections

def color_graph(edges: list[tuple[int, int]], nodes: list[int] | None = None) -> dict[int, int]:
    """
    Perform graph coloring to enable staggered parallel tuning.

    Assigns a 'color' (integer) to each node such that no two adjacent nodes
    share the same color. This is useful for scheduling calibration pulses
    where neighboring qubits cannot be tuned simultaneously to avoid crosstalk.

    Args:
        edges: List of edges representing the coupling graph, e.g., [(0, 1), (1, 2)].
        nodes: Optional list of all nodes. If None, nodes are inferred from edges.
               Useful if there are disconnected nodes.

    Returns:
        A dictionary mapping node ID to its assigned color (0, 1, 2, ...).
    """
    adj = collections.defaultdict(list)

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if nodes is None:
        all_nodes = set()
        for u, v in edges:
            all_nodes.add(u)
            all_nodes.add(v)
        nodes_list = sorted(list(all_nodes))
    else:
        nodes_list = sorted(list(nodes))

    colors = {}

    for node in nodes_list:
        # Find colors of neighbors
        neighbor_colors = {colors[neighbor] for neighbor in adj[node] if neighbor in colors}

        # Find the lowest available color
        color = 0
        while color in neighbor_colors:
            color += 1

        colors[node] = color

    return colors

def get_staggered_tuning_sets(edges: list[tuple[int, int]], nodes: list[int] | None = None) -> list[list[int]]:
    """
    Groups qubits into independent sets for staggered parallel tuning.

    Args:
        edges: List of edges representing the coupling graph.
        nodes: Optional list of all nodes.

    Returns:
        A list of lists, where each inner list contains node IDs that can be
        tuned simultaneously (i.e., they have the same color).
    """
    colors = color_graph(edges, nodes)

    sets_by_color = collections.defaultdict(list)
    for node, color in colors.items():
        sets_by_color[color].append(node)

    # Return as a list of lists, ordered by color
    return [sets_by_color[color] for color in sorted(sets_by_color.keys())]
