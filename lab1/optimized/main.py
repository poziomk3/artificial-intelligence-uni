import networkx as nx

from lab1.optimized.a_star_fastest_route_pruned import a_star_fastest_route_optimized
from lab1.optimized.a_star_optimized_line_changes import a_star_optimized_line_changes
from lab1.optimized.a_star_time import a_star_fastest_route
from lab1.optimized.a_star_transfers import a_star_minimize_line_changes
from lab1.optimized.algo_commons import *
from lab1.optimized.cost_function import time_cost
from lab1.optimized.djikstra_time import *
from lab1.optimized.read_graph import *
import pandas as pd
import time

from lab1.optimized.tabu import tabu_search
from lab1.optimized.tabu_b import tabu_search_full

FILE_PATH = "../connection_graph.csv"
GRAPH_FILE = "graph.gpickle"

if __name__ == "__main__":
    graph = read_graph(GRAPH_FILE, FILE_PATH)

    # print_all_nodes(graph)
    # print_edge(graph, "DWORZEC AUTOBUSOWY", "EPI")
    start_stop = "DWORZEC AUTOBUSOWY"
    end_stop = "Kiełczów - pętla/Wrocławska"
    start_time = pd.Timestamp("6:30:00")

    # dijkstra_fastest_route(graph, start_stop, end_stop, start_time)

    # a_star_minimize_line_changes(graph, start_stop, end_stop, start_time, manhattan_distance)
    # a_star_minimize_line_changes(graph, start_stop, end_stop, start_time, haversine_line_change_estimate)
    # a_star_optimized_line_changes(graph, start_stop, end_stop, start_time, haversine_line_change_estimate)

    # a_star_fastest_route(graph, start_stop, end_stop, start_time, manhattan_distance)
    # a_star_fastest_route(graph, start_stop, end_stop, start_time, haversine_time_estimate)
    # a_star_fastest_route_optimized(graph, start_stop, end_stop, start_time, haversine_time_estimate)
    # #
    start_stop = "DWORZEC AUTOBUSOWY"
    checkpoints = ["Kiełczów - pętla/Wrocławska", "EPI", "Vivaldiego", "Pułaskiego", "Rynek"]

    full_path, cost = tabu_search(graph, start_stop, checkpoints, "t", start_time)

    for step in full_path:
        src, dst, line, dep_time, arr_time = step
        print(f"{line} {dep_time.time()} {src} -> {arr_time.time()} {dst}")

    print(f"\nKoszt: {cost}")
