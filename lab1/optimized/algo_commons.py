from typing import Tuple, Optional, List

import networkx as nx
import pandas as pd


def init_algo(graph: nx.DiGraph, start_stop: str):
    for stop in graph.nodes:
        graph.nodes[stop]["visited"] = False
        graph.nodes[stop]["predecessor"] = None
        graph.nodes[stop]["arrival"] = None
        graph.nodes[stop]["cost"] = float("inf")

    graph.nodes[start_stop]["cost"] = 0


def print_algo_result(result: Optional[Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], pd.Timedelta]]):
    """
    Prints the route result from either Dijkstra or A* algorithm.

    :param result: The output of a shortest-path algorithm, containing a list of edges and total travel time.
    """
    if result is None:
        print("⚠️ No valid route found.")
        return

    path_edges, cost = result
    print("\n✅ Fastest route:\n")

    if not path_edges:
        print("⚠️ No valid path found.")
        return

    previous_line = None

    for edge in path_edges:
        from_stop, to_stop, line, departure, arrival = edge

        # Print when entering a new line
        if line != previous_line:
            if previous_line is not None:
                print(f"🚏 Change at: {from_stop} at {departure.strftime('%H:%M:%S')}\n")
            print(f"🚌 Line {line} | Board at {from_stop} at {departure.strftime('%H:%M:%S')}")

        # Print the next stop in sequence
        print(f"   → {to_stop} (Arrives at {arrival.strftime('%H:%M:%S')})")

        previous_line = line  # Update previous line

    # Final stop arrival
    print(f"\n⏳ Arrived at {to_stop} at {arrival.strftime('%H:%M:%S')}")

    # Print total travel time
    print(f"\n🕒 Total cost: {cost}")


def manhattan_distance(start_stop, end_stop):
    return abs(start_stop["lat"] - end_stop["lat"]) + abs(start_stop["lon"] - end_stop["lon"])


def euclidean_distance(start_stop, end_stop):
    return ((start_stop["lat"] - end_stop["lat"]) ** 2 + (start_stop["lon"] - end_stop["lon"]) ** 2) ** 0.5
