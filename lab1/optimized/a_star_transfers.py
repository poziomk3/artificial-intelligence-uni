import heapq
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional

from lab1.optimized.algo_commons import init_algo
from lab1.optimized.utils import log_route_info, format_algo_result


@format_algo_result
@log_route_info(algo_name="A* with line changes")
def a_star_minimize_line_changes(
        graph: nx.DiGraph,
        start: str,
        end: str,
        start_time: pd.Timestamp,
        heuristic_func
) -> Optional[Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], int]]:
    if not (graph.has_node(start) and graph.has_node(end)):
        return None

    start_time = start_time.replace(year=1900, month=1, day=1)
    init_algo(graph, start)
    pq = []
    heapq.heappush(pq,
                   (0, 0, start, start_time, [], None))

    open_set = {start}
    closed_set = set()

    while pq:
        f_cost, line_changes, current_stop, arrival_time, path, prev_line = heapq.heappop(pq)
        open_set.discard(current_stop)
        closed_set.add(current_stop)

        if current_stop == end:
            return path, line_changes

        for neighbor in graph.neighbors(current_stop):
            if neighbor in closed_set:
                continue  # Skip already processed nodes

            route_options = graph[current_stop][neighbor]["data"]

            valid_routes = [r for r in route_options if r.departure_time >= arrival_time]

            if not valid_routes:
                continue

            best_route = min(valid_routes, key=lambda r: r.departure_time)
            new_line_changes = line_changes + (1 if prev_line and prev_line != best_route.line else 0)

            h_new = heuristic_func(graph.nodes[neighbor], graph.nodes[end])
            f_new = new_line_changes + h_new  # Cost function considers line changes and heuristic

            if graph.nodes[neighbor]["cost"] > new_line_changes:
                graph.nodes[neighbor]["cost"] = new_line_changes
                graph.nodes[neighbor]["predecessor"] = current_stop

                new_path = path + [
                    (current_stop, neighbor, best_route.line, best_route.departure_time, best_route.arrival_time)]
                heapq.heappush(pq,
                               (f_new, new_line_changes, neighbor, best_route.arrival_time, new_path, best_route.line))
                open_set.add(neighbor)

    return None
