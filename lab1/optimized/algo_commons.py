from typing import Tuple, Optional, List

import networkx as nx
import pandas as pd
import math


def init_algo(graph: nx.DiGraph, start_stop: str):
    for stop in graph.nodes:
        graph.nodes[stop]["visited"] = False
        graph.nodes[stop]["predecessor"] = None
        graph.nodes[stop]["arrival"] = None
        graph.nodes[stop]["cost"] = float("inf")

    graph.nodes[start_stop]["cost"] = 0


def print_algo_result(result: Optional[Tuple[List[Tuple[str, str, str, pd.Timestamp, pd.Timestamp]], pd.Timedelta]]):
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

        if line != previous_line:
            if previous_line is not None:
                print(f"🚏 Change at: {from_stop} at {departure.strftime('%H:%M:%S')}\n")
            print(f"🚌 Line {line} | Board at {from_stop} at {departure.strftime('%H:%M:%S')}")

        print(f"   → {to_stop} (Arrives at {arrival.strftime('%H:%M:%S')})")

        previous_line = line  # Update previous line

    print(f"\n⏳ Arrived at {to_stop} at {arrival.strftime('%H:%M:%S')}")

    print(f"\n🕒 Total cost: {cost}")


def manhattan_distance(start_stop, end_stop):
    return abs(start_stop["lat"] - end_stop["lat"]) + abs(start_stop["lon"] - end_stop["lon"])


def euclidean_distance(start_stop, end_stop):
    return ((start_stop["lat"] - end_stop["lat"]) ** 2 + (start_stop["lon"] - end_stop["lon"]) ** 2) ** 0.5


EARTH_RADIUS_KM = 6371.0
MAX_TRANSIT_SPEED_KMPH = 40


def haversine_time_estimate(start_stop, end_stop):
    """
    Estimate travel time (in minutes) between two stops using Haversine distance and max speed.
    """
    lat1, lon1 = math.radians(start_stop["lat"]), math.radians(start_stop["lon"])
    lat2, lon2 = math.radians(end_stop["lat"]), math.radians(end_stop["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = EARTH_RADIUS_KM * c

    estimated_time_minutes = (distance_km / MAX_TRANSIT_SPEED_KMPH) * 60
    return estimated_time_minutes


AVG_LINE_COVERAGE_KM = 3.0


def haversine_line_change_estimate(start_stop, end_stop):
    lat1, lon1 = math.radians(start_stop["lat"]), math.radians(start_stop["lon"])
    lat2, lon2 = math.radians(end_stop["lat"]), math.radians(end_stop["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = EARTH_RADIUS_KM * c

    estimated_changes = distance_km / AVG_LINE_COVERAGE_KM
    return estimated_changes
