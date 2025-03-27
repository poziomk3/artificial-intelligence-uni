from lab1.optimized.utils import print_result, print_info
import heapq
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional, Callable

from lab1.optimized.algo_commons import init_algo
from lab1.optimized.utils import print_info, print_result


import heapq
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional, Callable
from itertools import count

from lab1.optimized.algo_commons import init_algo
from lab1.optimized.utils import print_info, print_result


@print_result
@print_info(algo_name="A* with time (optimized)")
def a_star_fastest_route_optimized(
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
    counter = count()
    heapq.heappush(pq, (0, 0, next(counter), start, start_time, []))

    best_arrival_time = None
    visited_times = {}

    while pq:
        f_cost, g_cost, _, current_stop, arrival_time, path = heapq.heappop(pq)

        # Odwiedzony wcześniej z lepszym czasem – pomiń
        if current_stop in visited_times and arrival_time >= visited_times[current_stop]:
            continue

        visited_times[current_stop] = arrival_time

        # Jeśli osiągnięto cel – sprawdź, czy to najlepsze rozwiązanie
        if current_stop == end:
            if best_arrival_time is None or arrival_time < best_arrival_time:
                best_arrival_time = arrival_time
            return path, arrival_time - start_time, len(visited_times)

        for neighbor in graph.neighbors(current_stop):
            route_options = graph[current_stop][neighbor]["data"]
            valid_routes = [r for r in route_options if r.departure_time >= arrival_time]
            if not valid_routes:
                continue

            best_route = min(valid_routes, key=lambda r: r.departure_time)
            new_arrival = best_route.arrival_time

            # Pruning: jeśli dotarcie zajmie dużo więcej niż najlepsze znalezione – pomiń
            if best_arrival_time and new_arrival > best_arrival_time + pd.Timedelta(minutes=5):
                continue

            # Koszty
            waiting_time = (best_route.departure_time - arrival_time).total_seconds() / 60
            travel_time = best_route.time_delta.total_seconds() / 60
            g_new = g_cost + waiting_time + travel_time

            h_new = heuristic_func(graph.nodes[neighbor], graph.nodes[end])
            h_scaled = 2.0 * h_new  # skalowanie heurystyki, można dostosować
            f_new = g_new + h_scaled

            # Jeśli koszt lepszy – zaktualizuj i dodaj do kolejki
            if graph.nodes[neighbor]["cost"] > g_new:
                graph.nodes[neighbor]["cost"] = g_new
                graph.nodes[neighbor]["predecessor"] = current_stop
                graph.nodes[neighbor]["arrival"] = new_arrival

                new_path = path + [
                    (current_stop, neighbor, best_route.line, best_route.departure_time, best_route.arrival_time)
                ]

                heapq.heappush(pq, (f_new, g_new, next(counter), neighbor, new_arrival, new_path))

    return None
