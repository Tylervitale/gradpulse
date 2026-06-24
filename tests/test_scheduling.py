import pytest
from gradpulse.scheduling import color_graph, get_staggered_tuning_sets

def test_color_graph_line():
    edges = [(0, 1), (1, 2), (2, 3)]
    colors = color_graph(edges)
    assert colors[0] != colors[1]
    assert colors[1] != colors[2]
    assert colors[2] != colors[3]
    # Two colors should be sufficient for a line graph
    assert set(colors.values()) == {0, 1}

def test_color_graph_star():
    # Node 0 connected to 1, 2, 3, 4
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    colors = color_graph(edges)
    assert colors[0] != colors[1]
    assert colors[0] != colors[2]
    assert colors[0] != colors[3]
    assert colors[0] != colors[4]

    # 1, 2, 3, 4 can all share the same color since they aren't connected
    assert colors[1] == colors[2] == colors[3] == colors[4]

def test_color_graph_grid_checkerboard():
    # 2x2 grid
    # 0 - 1
    # |   |
    # 2 - 3
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    colors = color_graph(edges)

    # Adjacencies
    assert colors[0] != colors[1]
    assert colors[0] != colors[2]
    assert colors[1] != colors[3]
    assert colors[2] != colors[3]

    # Diagonals can share colors (checkerboard pattern)
    assert colors[0] == colors[3]
    assert colors[1] == colors[2]

def test_get_staggered_tuning_sets():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    sets = get_staggered_tuning_sets(edges)

    # Should result in 2 sets for a bipartite graph
    assert len(sets) == 2

    # Each set shouldn't have connected nodes
    for tuning_set in sets:
        for u, v in edges:
            assert not (u in tuning_set and v in tuning_set)

def test_disconnected_nodes():
    edges = [(0, 1)]
    nodes = [0, 1, 2] # Node 2 is disconnected
    colors = color_graph(edges, nodes)

    assert colors[0] != colors[1]
    assert 2 in colors

    sets = get_staggered_tuning_sets(edges, nodes)
    # Node 2 can be grouped with either 0 or 1's color group (color 0)
    assert 2 in sets[0]

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
