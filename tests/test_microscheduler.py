import pytest
import numpy as np

from gradpulse.scheduling import DependencyGraph, OperationNode
from gradpulse.microscheduler import Microscheduler

def test_microscheduler_basic_sequential():
    graph = DependencyGraph()
    # op1 takes 10ns on channel 0
    node1 = OperationNode("op1", [0], duration_ns=10.0, channels=["q0_drive"])
    # op2 takes 20ns on channel 0
    node2 = OperationNode("op2", [0], duration_ns=20.0, channels=["q0_drive"])

    graph.build_dependencies([node1, node2])

    scheduler = Microscheduler(dt_ns=1.0)
    schedule = scheduler.schedule(graph)

    assert schedule["op1"] == 0.0
    assert schedule["op2"] == 10.0

def test_microscheduler_parallel():
    graph = DependencyGraph()
    # op1 takes 10ns on q0
    node1 = OperationNode("op1", [0], duration_ns=10.0, channels=["q0_drive"])
    # op2 takes 20ns on q1 - no dependency with op1
    node2 = OperationNode("op2", [1], duration_ns=20.0, channels=["q1_drive"])

    graph.build_dependencies([node1, node2])

    scheduler = Microscheduler(dt_ns=1.0)
    schedule = scheduler.schedule(graph)

    assert schedule["op1"] == 0.0
    assert schedule["op2"] == 0.0

def test_microscheduler_channel_margin():
    graph = DependencyGraph()
    # two independent operations but on the same channel
    node1 = OperationNode("op1", [0], duration_ns=10.0, channels=["shared_channel"])
    node2 = OperationNode("op2", [1], duration_ns=20.0, channels=["shared_channel"])

    # We do not build dependencies automatically to test pure channel constraint
    graph.add_node(node1)
    graph.add_node(node2)

    scheduler = Microscheduler(dt_ns=1.0)
    scheduler.add_channel_margin("shared_channel", 2.0)
    schedule = scheduler.schedule(graph)

    # They should be scheduled one after another with 2.0ns margin
    # op1 starts at 0, ends at 10. channel is free at 12.
    # op2 starts at 12.
    assert schedule["op1"] == 0.0
    assert schedule["op2"] == 12.0

def test_microscheduler_grid_alignment():
    graph = DependencyGraph()
    node1 = OperationNode("op1", [0], duration_ns=10.1, channels=["q0_drive"])
    node2 = OperationNode("op2", [0], duration_ns=20.0, channels=["q0_drive"])

    graph.build_dependencies([node1, node2])

    # Grid is 1.0ns.
    # op1 ends at 10.1. op2 must start at nearest grid point >= 10.1, which is 11.0
    scheduler = Microscheduler(dt_ns=1.0)
    schedule = scheduler.schedule(graph)

    assert schedule["op1"] == 0.0
    assert schedule["op2"] == 11.0
