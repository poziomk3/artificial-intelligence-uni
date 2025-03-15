import networkx as nx

from lab1.optimized.a_star_time import a_star_fastest_route
from lab1.optimized.a_star_transfers import a_star_minimize_line_changes
from lab1.optimized.algo_commons import *
from lab1.optimized.djikstra_time import *
from lab1.optimized.read_graph import *
import pandas as pd

FILE_PATH = "../connection_graph.csv"
GRAPH_FILE = "graph.gpickle"

if __name__ == "__main__":
    graph = read_graph(GRAPH_FILE, FILE_PATH)

    # print_all_nodes(graph)
    # print_edge(graph, "DWORZEC AUTOBUSOWY", "EPI")
    start_stop = "PL. GRUNWALDZKI"
    end_stop = "GALERIA DOMINIKAŃSKA"
    start_time = pd.Timestamp("16:30:00")

    result_astar_transfers = a_star_minimize_line_changes(graph, start_stop, end_stop, start_time, manhattan_distance)
    print_algo_result(result_astar_transfers)

    result_astar_time = a_star_fastest_route(graph, start_stop, end_stop, start_time, euclidean_distance)
    print_algo_result(result_astar_time)

    result = dijkstra_fastest_route(graph, start_stop, end_stop, start_time)
    print_algo_result(result)
