from lab1.optimized.tabu import calculate_route
import random

from lab1.optimized.tabu import calculate_route
import random
from collections import deque


def tabu_search_full(graph, start, checkpoints, criterion, start_time, iterations=100, sample_size=30):
    best_solution = checkpoints[:]
    random.shuffle(best_solution)

    best_cost = float('inf')
    best_path = []

    # (b) dynamiczna długość tabu
    tabu_length = max(5, int(len(checkpoints) * 1.5))
    tabu_list = deque(maxlen=tabu_length)

    evaluated_routes = 0

    for _ in range(iterations):
        all_neighbors = []

        # (d) tworzenie wszystkich sąsiadów przez zamianę dwóch punktów
        for i in range(len(best_solution)):
            for j in range(i + 1, len(best_solution)):
                neighbor = best_solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                all_neighbors.append(neighbor)

        # (d)
        neighborhood = random.sample(all_neighbors, min(sample_size, len(all_neighbors)))

        improved = False
        for neighbor in neighborhood:
            state = tuple(neighbor)

            route, cost = calculate_route(graph, [start] + neighbor + [start], start_time, criterion)
            evaluated_routes += 1

            # (c) zasada aspiracji
            if state in tabu_list and cost >= best_cost:
                continue

            if cost < best_cost:
                best_solution = neighbor
                best_cost = cost
                best_path = route
                improved = True

        # (b) aktualizacja tabu listy
        tabu_list.append(tuple(best_solution))

        # jeśli nie było poprawy – przerwij
        if not improved:
            break

    print(f"🔍 Liczba przeliczonych tras: {evaluated_routes}")
    return best_path, best_cost
