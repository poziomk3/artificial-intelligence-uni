from lab1.optimized.tabu import calculate_route
import random


def tabu_search_b(graph, start, checkpoints, criterion, start_time, iterations=100):
    from collections import deque

    best_solution = checkpoints[:]
    random.shuffle(best_solution)

    best_cost = float('inf')
    best_path = []

    tabu_length = max(5, int(len(checkpoints) * 1.5))
    tabu_list = deque(maxlen=tabu_length)

    evaluated_routes = 0  # 🔢 licznik tras

    for _ in range(iterations):
        neighborhood = []

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

            evaluated_routes += 1  # 🔢 inkrementacja

            route, cost = calculate_route(graph, [start] + neighbor + [start], start_time, criterion)
            if cost < best_cost:
                best_solution = neighbor
                best_cost = cost
                best_path = route
                improved = True

        tabu_list.append(tuple(best_solution))

        if not improved:
            break

    print(f"🔍 Liczba przeliczonych tras: {evaluated_routes}")
    return best_path, best_cost
