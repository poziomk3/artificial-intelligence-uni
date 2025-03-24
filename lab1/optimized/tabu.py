from lab1.optimized.a_star_time import a_star_fastest_route
from lab1.optimized.a_star_transfers import a_star_minimize_line_changes
from lab1.optimized.algo_commons import haversine_time_estimate, haversine_line_change_estimate
import random


def calculate_route(graph, stops, start_time, criterion):
    total_cost = 0
    total_path = []
    current_time = start_time

    for i in range(len(stops)):
        src = stops[i]
        dst = stops[(i + 1) % len(stops)]  # last goes back to start

        if criterion == 't':
            result = a_star_fastest_route(graph, src, dst, current_time, haversine_time_estimate)
            if not result:
                return None, float('inf')
            path, time_spent, _ = result
            current_time += time_spent
            total_cost += time_spent.total_seconds()
        else:  # 'p' for line changes
            result = a_star_minimize_line_changes(graph, src, dst, current_time, haversine_line_change_estimate)
            if not result:
                return None, float('inf')
            path, line_changes, _ = result
            total_cost += line_changes
            if path:
                current_time = path[-1][4]

        total_path.extend(path)

    return total_path, total_cost


def tabu_search(graph, start, checkpoints, criterion, start_time, iterations=100):
    stops = [start] + checkpoints
    best_solution = checkpoints[:]
    random.shuffle(best_solution)

    best_cost = float('inf')
    best_path = []

    tabu_list = set()
    evaluated_routes = 0
    for _ in range(iterations):
        neighborhood = []

        # create neighbors by swapping two checkpoints
        for i in range(len(best_solution)):
            for j in range(i + 1, len(best_solution)):
                neighbor = best_solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighborhood.append(neighbor)

        improved = False
        for neighbor in neighborhood:
            state = tuple(neighbor)
            if state in tabu_list:
                continue
            evaluated_routes += 1
            route, cost = calculate_route(graph, [start] + neighbor + [start], start_time, criterion)
            if cost < best_cost:
                best_solution = neighbor
                best_cost = cost
                best_path = route
                improved = True

        tabu_list.add(tuple(best_solution))

        if not improved:
            break
    print(f"🔍 Liczba przeliczonych tras: {evaluated_routes}")
    return best_path, best_cost
