"""
core/graph_routing.py

A small simulated router network plus Dijkstra's algorithm, implemented
manually (explicit distance table + relaxation loop, no library
shortest-path call). This stage does not transform packet payloads --
it annotates each packet with the path it takes across the network,
which the visualizer can animate as packets moving node-to-node.

Math concept demonstrated: graph theory, shortest-path algorithms.
"""

import heapq

from core.pipeline import Stage, Trace

DEFAULT_GRAPH = {
    "A": {"B": 2, "C": 5},
    "B": {"A": 2, "C": 1, "D": 4},
    "C": {"A": 5, "B": 1, "D": 2},
    "D": {"B": 4, "C": 2},
}


def dijkstra(graph: dict, source: str, target: str, trace: Trace | None = None):
    """
    Standard Dijkstra with an explicit distance table and a min-heap for
    priority selection. Returns (path as list of node ids, total cost).
    """
    dist = {node: float("inf") for node in graph}
    dist[source] = 0
    prev: dict[str, str] = {}
    visited: set[str] = set()
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if trace is not None:
            trace.log_step(action="visit_node", node=u, distance=d)
        if u == target:
            break
        for v, weight in graph.get(u, {}).items():
            nd = d + weight
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
                if trace is not None:
                    trace.log_step(action="relax_edge", frm=u, to=v, new_dist=nd)

    if target not in dist or dist[target] == float("inf"):
        return None, float("inf")

    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path, dist[target]


class GraphRoutingStage(Stage):
    name = "Graph Routing (Dijkstra)"

    def __init__(self, graph: dict | None = None, source: str = "A", target: str = "D"):
        self.graph = graph or DEFAULT_GRAPH
        self.source = source
        self.target = target

    def run(self, data: dict, trace: Trace) -> dict:
        path, cost = dijkstra(self.graph, self.source, self.target, trace)
        trace.metadata["path"] = path
        trace.metadata["path_cost"] = cost

        packets = [{**p, "path": path} for p in data["packets"]]
        return {"packets": packets, "meta": data["meta"]}
