"""
routing_dijkstra.py — Network topology graph + Dijkstra's shortest-path
algorithm (Graph Theory).

Mathematical background
------------------------
A computer network's routers/links are modeled as a weighted,
undirected graph G = (V, E), where each edge weight represents a cost
(e.g. latency or hop cost). Dijkstra's algorithm finds the minimum-
total-weight path from a source vertex to every other vertex (we
extract just the source->target path), using the *greedy* + *relaxation*
strategy:

  1. Maintain tentative distances dist[v] (initially infinity, except
     dist[source]=0) and a min-priority-queue keyed by dist.
  2. Repeatedly extract the unvisited vertex u with smallest dist[u]
     (this greedy choice is provably safe for graphs with
     non-negative weights: once u has the smallest tentative distance
     among all unvisited vertices, no path through another unvisited
     vertex could reach it more cheaply).
  3. "Relax" every edge (u, v): if dist[u] + weight(u,v) < dist[v],
     improve dist[v] and record u as v's predecessor.
  4. Stop once the target is popped from the queue (or the queue is
     empty).

Implemented here with Python's heapq as the min-priority queue, giving
O((V+E) log V) time.
"""

from __future__ import annotations
import heapq


class Graph:
    """Weighted, undirected graph via adjacency list, user-editable."""

    def __init__(self):
        self.adjacency: dict[str, dict[str, float]] = {}

    def add_node(self, node: str) -> None:
        self.adjacency.setdefault(node, {})

    def remove_node(self, node: str) -> None:
        if node in self.adjacency:
            del self.adjacency[node]
        for neighbors in self.adjacency.values():
            neighbors.pop(node, None)

    def add_edge(self, u: str, v: str, weight: float) -> None:
        if weight < 0:
            raise ValueError("Dijkstra's algorithm requires non-negative edge weights")
        self.add_node(u)
        self.add_node(v)
        self.adjacency[u][v] = weight
        self.adjacency[v][u] = weight  # undirected

    def remove_edge(self, u: str, v: str) -> None:
        self.adjacency.get(u, {}).pop(v, None)
        self.adjacency.get(v, {}).pop(u, None)

    def nodes(self) -> list[str]:
        return list(self.adjacency.keys())

    def edges(self) -> list[dict]:
        seen = set()
        result = []
        for u, neighbors in self.adjacency.items():
            for v, w in neighbors.items():
                key = tuple(sorted((u, v)))
                if key not in seen:
                    seen.add(key)
                    result.append({"source": u, "target": v, "weight": w})
        return result

    def to_dict(self) -> dict:
        return {"nodes": self.nodes(), "edges": self.edges()}


def default_topology() -> Graph:
    """A default topology with 6 nodes and 8 edges (multiple candidate paths exist)."""
    g = Graph()
    edges = [
        ("A", "B", 4), ("A", "C", 2),
        ("B", "C", 1), ("B", "D", 5),
        ("C", "D", 8), ("C", "E", 10),
        ("D", "E", 2), ("D", "F", 6),
        ("E", "F", 3),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def dijkstra(graph: Graph, source: str, target: str) -> tuple[list[str], float, list[dict]]:
    """
    Compute the shortest path from `source` to `target`.

    Returns (path, total_weight, relaxation_steps), where
    relaxation_steps is an ordered log of every edge relaxation
    attempted (for frontend visualization of the algorithm's
    progress): each entry records which edge was examined, whether it
    improved a distance, and the distances before/after.
    """
    if source not in graph.adjacency:
        raise ValueError(f"unknown source node: {source}")
    if target not in graph.adjacency:
        raise ValueError(f"unknown target node: {target}")

    dist: dict[str, float] = {node: float("inf") for node in graph.adjacency}
    dist[source] = 0
    prev: dict[str, str | None] = {node: None for node in graph.adjacency}
    visited: set[str] = set()
    pq: list[tuple[float, str]] = [(0, source)]
    relaxation_steps: list[dict] = []

    while pq:
        d_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break

        for v, weight in graph.adjacency[u].items():
            if v in visited:
                continue
            candidate = d_u + weight
            improved = candidate < dist[v]
            relaxation_steps.append({
                "edge": [u, v],
                "weight": weight,
                "candidate_distance": candidate,
                "previous_distance": dist[v] if dist[v] != float("inf") else None,
                "improved": improved,
            })
            if improved:
                dist[v] = candidate
                prev[v] = u
                heapq.heappush(pq, (candidate, v))

    if dist[target] == float("inf"):
        return [], float("inf"), relaxation_steps

    # Reconstruct path by walking predecessors backward from target to source.
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path, dist[target], relaxation_steps


def shortest_path_report(graph: Graph, source: str, target: str) -> dict:
    path, total_cost, steps = dijkstra(graph, source, target)
    return {
        "path": path,
        "total_cost": total_cost if total_cost != float("inf") else None,
        "relaxation_steps": steps,
        "reachable": len(path) > 0,
    }
