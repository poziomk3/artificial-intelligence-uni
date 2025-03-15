import networkx as nx
import pickle
from typing import List, Optional
from graph_commons import *
import os


class GraphLoadError(Exception):
    """Custom exception for graph loading or creation failures."""
    pass


def read_graph(graph_file: str, file_path: str):
    """
    Loads a graph from a pickle file if it exists. Otherwise, creates a new graph.

    :param graph_file: Path to the saved graph file.
    :param file_path: Path to the CSV file to create the graph if it doesn't exist.
    :return: A NetworkX DiGraph if successful.
    :raises GraphLoadError: If the graph cannot be loaded or created.
    """
    G = None

    if os.path.exists(graph_file):
        print(f"📌 Graph file '{graph_file}' found. Loading graph...")
        G = load_graph(graph_file)
    else:
        print(f"📌 Graph file '{graph_file}' not found. Creating a new graph...")
        G = create_and_save_graph(file_path)

    if G:
        print(f"✅ Graph is ready to use! Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    else:
        raise GraphLoadError("❌ Failed to load or create the graph.")


def create_and_save_graph(file_path):
    dtype_dict = {
        "start_stop": "str",
        "end_stop": "str",
        "line": "str",
        "start_stop_lat": "float32",
        "start_stop_lon": "float32",
        "end_stop_lat": "float32",
        "end_stop_lon": "float32",
    }

    df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
    print("CSV Loaded Successfully.")

    df["departure_time"] = df["departure_time"].astype(str).apply(fix_invalid_time)
    df["arrival_time"] = df["arrival_time"].astype(str).apply(fix_invalid_time)

    df["departure_time"] = pd.to_datetime(df["departure_time"], format="%H:%M:%S", errors="coerce")
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], format="%H:%M:%S", errors="coerce")

    df.dropna(subset=["departure_time", "arrival_time"], inplace=True)

    G = nx.DiGraph()

    routes: List[RouteData] = []

    for row in df.itertuples(index=False):
        route = RouteData(
            start_stop=row.start_stop,
            end_stop=row.end_stop,
            line=row.line,
            start_stop_lat=row.start_stop_lat,
            start_stop_lon=row.start_stop_lon,
            end_stop_lat=row.end_stop_lat,
            end_stop_lon=row.end_stop_lon,
            departure_time=row.departure_time,
            arrival_time=row.arrival_time,
        )
        routes.append(route)

        if not G.has_node(route.start_stop):
            G.add_node(route.start_stop, lat=route.start_stop_lat, lon=route.start_stop_lon)
        if not G.has_node(route.end_stop):
            G.add_node(route.end_stop, lat=route.end_stop_lat, lon=route.end_stop_lon)

        if not G.has_edge(route.start_stop, route.end_stop):
            G.add_edge(route.start_stop, route.end_stop, data=[])

        route_piece = RoutePiece(
            line=route.line,
            departure_time=route.departure_time,
            arrival_time=route.arrival_time,
            time_delta=calculate_time_delta(route.departure_time, route.arrival_time),
        )

        edge_data = G[route.start_stop][route.end_stop]["data"]
        edge_data.append(route_piece)
        edge_data.sort(key=lambda x: x.arrival_time)

    with open("graph.gpickle", "wb") as f:
        pickle.dump(G, f)

    return G


def load_graph(file_path: str) -> Optional[nx.DiGraph]:
    """
    Loads a directed graph from a pickle file.

    :param file_path: Path to the .gpickle file containing the graph.
    :return: A NetworkX DiGraph if loading is successful, otherwise None.
    """
    try:
        with open(file_path, "rb") as f:
            graph = pickle.load(f)

        if not isinstance(graph, nx.DiGraph):
            raise TypeError("Loaded object is not a NetworkX DiGraph.")

        return graph

    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading graph: {e}")
        return None


def print_all_nodes(graph: nx.DiGraph):
    """
    Prints all nodes in the graph along with their attributes.

    :param graph: A NetworkX directed graph.
    """
    print("\n📌 List of All Nodes in the Graph:")

    for node, data in graph.nodes(data=True):
        lat = data.get("lat", "N/A")
        lon = data.get("lon", "N/A")
        print(f"🔹 Node: {node} | Latitude: {lat}, Longitude: {lon}")

    print(f"\n✅ Total Nodes: {graph.number_of_nodes()}")


def print_all_edges(graph: nx.DiGraph, edge_limit: int = 10):
    """
    Prints a limited number of edges in the graph along with their attributes.

    :param graph: A NetworkX directed graph.
    :param edge_limit: Maximum number of edges to display.
    """
    print("\n📌 List of Edges in the Graph:")

    # Convert edges to a list and slice it correctly
    edges = list(graph.edges(data=True))[:edge_limit]

    for start, end, data in edges:
        print(f"\n🔹 Edge: {start} → {end}")

        # Check if edge has 'data' key with route details
        if "data" in data:
            for route_piece in data["data"]:
                print(f"   🚌 Line: {route_piece.line} | Departure: {route_piece.departure_time.strftime('%H:%M:%S')} | "
                      f"Arrival: {route_piece.arrival_time.strftime('%H:%M:%S')} | Duration: {route_piece.time_delta}")
        else:
            print("   ⚠️ No route information available for this edge.")

    print(f"\n✅ Total Edges: {graph.number_of_edges()}")


def print_edge(graph: nx.DiGraph, start_stop: str, end_stop: str):
    """
    Prints details of a specific edge in the graph.

    :param graph: A NetworkX directed graph.
    :param start_stop: The starting stop.
    :param end_stop: The destination stop.
    """
    print(f"\n📌 Details for Edge: {start_stop} → {end_stop}")

    if graph.has_edge(start_stop, end_stop):
        edge_data = graph[start_stop][end_stop]

        if "data" in edge_data:
            for route_piece in edge_data["data"]:
                print(f"   🚌 Line: {route_piece.line} | Departure: {route_piece.departure_time.strftime('%H:%M:%S')} | "
                      f"Arrival: {route_piece.arrival_time.strftime('%H:%M:%S')} | Duration: {route_piece.time_delta}")
        else:
            print("   ⚠️ No route information available for this edge.")
    else:
        print("   ❌ No direct route found between these stops.")
