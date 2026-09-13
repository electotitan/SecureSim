import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from pipeline import routing_dijkstra as rd


def test_default_topology_has_at_least_6_nodes_8_edges():
    g = rd.default_topology()
    assert len(g.nodes()) >= 6
    assert len(g.edges()) >= 8


def test_simple_known_shortest_path():
    g = rd.Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", 1)
    g.add_edge("A", "C", 5)
    path, cost, _ = rd.dijkstra(g, "A", "C")
    assert path == ["A", "B", "C"]
    assert cost == 2


def test_direct_edge_beats_longer_path_when_cheaper():
    g = rd.Graph()
    g.add_edge("A", "B", 10)
    g.add_edge("A", "C", 1)
    g.add_edge("C", "B", 1)
    path, cost, _ = rd.dijkstra(g, "A", "B")
    assert path == ["A", "C", "B"]
    assert cost == 2


def test_source_equals_target():
    g = rd.default_topology()
    path, cost, _ = rd.dijkstra(g, "A", "A")
    assert path == ["A"]
    assert cost == 0


def test_unreachable_node_returns_empty_path():
    g = rd.Graph()
    g.add_edge("A", "B", 1)
    g.add_node("Z")  # isolated node
    path, cost, _ = rd.dijkstra(g, "A", "Z")
    assert path == []
    assert cost == float("inf")


def test_default_topology_known_shortest_path():
    g = rd.default_topology()
    # A-C=2, C-B=1 => A-C-B = 3, vs A-B direct = 4
    path, cost, _ = rd.dijkstra(g, "A", "B")
    assert path == ["A", "C", "B"]
    assert cost == 3


def test_relaxation_steps_are_recorded():
    g = rd.default_topology()
    _, _, steps = rd.dijkstra(g, "A", "F")
    assert len(steps) > 0
    for step in steps:
        assert "edge" in step and "improved" in step


def test_remove_edge_and_node():
    g = rd.Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("B", "C", 1)
    g.remove_edge("A", "B")
    assert "B" not in g.adjacency["A"]
    g.remove_node("C")
    assert "C" not in g.nodes()


def test_negative_weight_rejected():
    g = rd.Graph()
    try:
        g.add_edge("A", "B", -1)
        assert False, "expected ValueError for negative weight"
    except ValueError:
        pass
