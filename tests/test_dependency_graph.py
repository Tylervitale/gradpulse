import numpy as np
import pytest
from gradpulse.scheduling import DependencyGraph, OperationNode, check_commutation

def test_check_commutation_disjoint():
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Disjoint qubits -> always commute
    assert check_commutation(X, [0], Z, [1]) == True

def test_check_commutation_same_qubits():
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    # X and Z anticommute
    assert check_commutation(X, [0], Z, [0]) == False

    # X and X commute
    assert check_commutation(X, [0], X, [0]) == True

def test_check_commutation_intersecting():
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    # CNOT on 0, 1
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])

    # X on 0 and CNOT on 0, 1 do not commute
    assert check_commutation(X, [0], CNOT, [0, 1]) == False

    # X on 1 and CNOT on 0, 1 (control on 0, target on 1).
    # CNOT is |0><0| I + |1><1| X. X commutes with both I and X, so X on 1 commutes with CNOT.
    assert check_commutation(X, [1], CNOT, [0, 1]) == True

def test_dependency_graph():
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])

    op1 = OperationNode("op1", [0], X)
    op2 = OperationNode("op2", [1], Z)
    op3 = OperationNode("op3", [0], Z)

    graph = DependencyGraph()
    graph.build_dependencies([op1, op2, op3])

    # op1 and op2 commute (disjoint) -> no edge
    # op2 and op3 commute (disjoint) -> no edge
    # op1 and op3 DO NOT commute -> edge from op1 to op3

    assert "op3" in graph.edges["op1"]
    assert "op2" not in graph.edges["op1"]
    assert "op3" not in graph.edges["op2"]

    order = graph.get_topological_order()
    # op1 must come before op3
    assert order.index("op1") < order.index("op3")
