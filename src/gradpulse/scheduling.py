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

import numpy as np

def check_commutation(op1_matrix: np.ndarray | None, op1_qubits: list[int],
                      op2_matrix: np.ndarray | None, op2_qubits: list[int],
                      tol: float = 1e-8) -> bool:
    """
    Check if two operations commute.
    If they operate on disjoint qubit sets, they commute.
    If matrices are provided and they intersect, check if their commutator is zero.
    """
    set1 = set(op1_qubits)
    set2 = set(op2_qubits)

    if set1.isdisjoint(set2):
        return True

    if op1_matrix is None or op2_matrix is None:
        # If they intersect but we don't have matrices to check, assume they don't commute
        return False

    joint_qubits = sorted(list(set1 | set2))
    n_joint = len(joint_qubits)
    dim = 2 # Assuming 2-level qubits

    def expand(matrix, qubits):
        if set(qubits) == set(joint_qubits):
            if qubits == joint_qubits:
                return matrix

        perm = [joint_qubits.index(q) for q in qubits] + [i for i in range(n_joint) if joint_qubits[i] not in qubits]
        I_rest = np.eye(dim ** (n_joint - len(qubits)), dtype=complex)
        op_kron_I = np.kron(matrix, I_rest)
        tensor = op_kron_I.reshape([dim] * (2 * n_joint))
        inv_perm = np.argsort(perm)
        full_perm = list(inv_perm) + [p + n_joint for p in inv_perm]
        permuted_tensor = np.transpose(tensor, full_perm)
        return permuted_tensor.reshape(dim**n_joint, dim**n_joint)

    exp1 = expand(op1_matrix, op1_qubits)
    exp2 = expand(op2_matrix, op2_qubits)

    commutator = exp1 @ exp2 - exp2 @ exp1
    return np.linalg.norm(commutator) < tol

class OperationNode:
    """Represents a quantum operation (pulse or gate) in the dependency graph."""
    def __init__(self, op_id: str, qubits: list[int], matrix: np.ndarray | None = None):
        self.op_id = op_id
        self.qubits = qubits
        self.matrix = matrix

    def __repr__(self):
        return f"OperationNode({self.op_id}, qubits={self.qubits})"

class DependencyGraph:
    """
    Directed Acyclic Graph (DAG) for scheduling quantum operations.
    Nodes are operations, edges represent causal dependencies (must execute before).
    """
    def __init__(self):
        self.nodes: dict[str, OperationNode] = {}
        self.edges: collections.defaultdict[str, list[str]] = collections.defaultdict(list)
        self.in_degree: collections.defaultdict[str, int] = collections.defaultdict(int)

    def add_node(self, node: OperationNode):
        if node.op_id not in self.nodes:
            self.nodes[node.op_id] = node
            self.in_degree[node.op_id] = 0

    def add_edge(self, from_id: str, to_id: str):
        """Add a dependency: from_id must finish before to_id can start."""
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
            self.in_degree[to_id] += 1

    def build_dependencies(self, sequence: list[OperationNode]):
        """
        Build the graph from a sequence of operations.
        Adds an edge from an earlier operation to a later one if they do not commute.
        """
        for node in sequence:
            self.add_node(node)

        n = len(sequence)
        for i in range(n):
            for j in range(i + 1, n):
                node_i = sequence[i]
                node_j = sequence[j]

                # If they do not commute, node_j depends on node_i
                if not check_commutation(node_i.matrix, node_i.qubits, node_j.matrix, node_j.qubits):
                    self.add_edge(node_i.op_id, node_j.op_id)

    def get_topological_order(self) -> list[str]:
        """Returns a valid execution order for the operations."""
        queue = collections.deque([node_id for node_id in self.nodes if self.in_degree[node_id] == 0])
        order = []

        in_deg = self.in_degree.copy()

        while queue:
            curr = queue.popleft()
            order.append(curr)

            for neighbor in self.edges[curr]:
                in_deg[neighbor] -= 1
                if in_deg[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle!")

        return order
