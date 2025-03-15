import heapq
import networkx as nx
import pandas as pd
import logging
from typing import Tuple, List, Optional

from lab1.optimized.algo_commons import init_algo


def dijkstra_fastest_route(graph: nx.DiGraph, start: str, end: str, start_time: pd.Timestamp) -> Optional[
    Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], pd.Timedelta]]:
    if not (graph.has_node(start) and graph.has_node(end)):
        return None

    start_time = start_time.replace(year=1900, month=1, day=1)

    init_algo(graph, start)

    pq = []
    heapq.heappush(pq, (0, start, start_time, []))

    while pq:
        current_cost, current_stop, arrival_time, path = heapq.heappop(pq)

        if current_stop == end:
            return path, arrival_time - start_time

        graph.nodes[current_stop]["visited"] = True

        for neighbor in graph.neighbors(current_stop):
            route_options = graph[current_stop][neighbor]["data"]

            valid_routes = [r for r in route_options if r.departure_time >= arrival_time]

            if not valid_routes:
                continue

            best_route = min(valid_routes, key=lambda r: r.departure_time)

            waiting_time = (best_route.departure_time - arrival_time).total_seconds() / 60
            travel_time = best_route.time_delta.total_seconds() / 60
            total_cost = current_cost + waiting_time + travel_time

            if graph.nodes[neighbor]["cost"] > total_cost:
                graph.nodes[neighbor]["cost"] = total_cost
                graph.nodes[neighbor]["predecessor"] = current_stop
                graph.nodes[neighbor]["arrival"] = best_route.arrival_time

                new_path = path + [
                    (current_stop, neighbor, best_route.line, best_route.departure_time, best_route.arrival_time)]
                heapq.heappush(pq, (total_cost, neighbor, best_route.arrival_time, new_path))

    return None
