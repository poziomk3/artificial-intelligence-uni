import heapq
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional, Callable

from lab1.optimized.algo_commons import init_algo
from lab1.optimized.utils import print_info, print_result


@print_result
@print_info(algo_name="A* with time")
def a_star_fastest_route(
        graph: nx.DiGraph,
        start: str,
        end: str,
        start_time: pd.Timestamp,
        heuristic_func: Callable[[str, str], float]
) -> Optional[Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], pd.Timedelta, int]]:
    if not (graph.has_node(start) and graph.has_node(end)):
        return None

    start_time = start_time.replace(year=1900, month=1, day=1)
    init_algo(graph, start)

    pq = []
    heapq.heappush(pq, (0, 0, start, start_time, []))  # (f_cost, g_cost, stop, arrival_time, path)

    open_set = {start}
    closed_set = set()

    while pq:
        f_cost, g_cost, current_stop, arrival_time, path = heapq.heappop(pq)
        open_set.discard(current_stop)
        closed_set.add(current_stop)

        if current_stop == end:
            return path, arrival_time - start_time, len(closed_set)

        for neighbor in graph.neighbors(current_stop):
            if neighbor in closed_set:
                continue

            route_options = graph[current_stop][neighbor]["data"]
            valid_routes = [r for r in route_options if r.departure_time >= arrival_time]

            if not valid_routes:
                continue

            best_route = min(valid_routes, key=lambda r: r.departure_time)
            waiting_time = (best_route.departure_time - arrival_time).total_seconds() / 60
            travel_time = best_route.time_delta.total_seconds() / 60
            g_new = g_cost + waiting_time + travel_time

            h_new = heuristic_func(graph.nodes[neighbor], graph.nodes[end])
            f_new = g_new + h_new
            if graph.nodes[neighbor]["cost"] > g_new:
                graph.nodes[neighbor]["cost"] = g_new
                graph.nodes[neighbor]["predecessor"] = current_stop
                graph.nodes[neighbor]["arrival"] = best_route.arrival_time

                new_path = path + [
                    (current_stop, neighbor, best_route.line, best_route.departure_time, best_route.arrival_time)]
                heapq.heappush(pq, (f_new, g_new, neighbor, best_route.arrival_time, new_path))
                open_set.add(neighbor)

    return None
