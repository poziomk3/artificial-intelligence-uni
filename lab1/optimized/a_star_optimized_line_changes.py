import heapq
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional, Callable

from lab1.optimized.algo_commons import init_algo
from lab1.optimized.utils import print_info, print_result


@print_result
@print_info(algo_name="A* with line changes (optimized)")
def a_star_optimized_line_changes(
        graph: nx.DiGraph,
        start: str,
        end: str,
        start_time: pd.Timestamp,
        heuristic_func: Callable[[dict, dict], float]
) -> Optional[Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], int, int]]:
    if not (graph.has_node(start) and graph.has_node(end)):
        return None

    start_time = start_time.replace(year=1900, month=1, day=1)
    init_algo(graph, start)

    # Priority queue: (f_cost, line_changes, node, arrival_time, path, prev_line)
    pq = [(0, 0, start, start_time, [], None)]
    visited = {}  # (node, prev_line) -> line_changes
    checked_nodes = 0

    while pq:
        f_cost, line_changes, current, arrival_time, path, prev_line = heapq.heappop(pq)
        checked_nodes += 1

        if (current, prev_line) in visited and visited[(current, prev_line)] <= line_changes:
            continue
        visited[(current, prev_line)] = line_changes

        if current == end:
            return path, line_changes, checked_nodes

        for neighbor in graph.neighbors(current):
            route_options = graph[current][neighbor]["data"]
            valid_routes = [r for r in route_options if r.departure_time >= arrival_time]
            if not valid_routes:
                continue

            best_route = min(valid_routes, key=lambda r: r.departure_time)

            new_line_changes = line_changes + (1 if prev_line and prev_line != best_route.line else 0)
            h = heuristic_func(graph.nodes[neighbor], graph.nodes[end])
            f = new_line_changes + h

            new_path = path + [(current, neighbor, best_route.line,
                                best_route.departure_time, best_route.arrival_time)]

            heapq.heappush(pq, (f, new_line_changes, neighbor, best_route.arrival_time, new_path, best_route.line))

    return None
