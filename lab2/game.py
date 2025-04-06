from Clobber import Clobber
from heuristic import heuristic_1, heuristic_2
from minimax import minimax
from minimaxAlphaBeta import minimax_ab
import time


def next_move(game, depth=3, heuristic_fn=heuristic_1, use_alphabeta=True):
    if use_alphabeta:
        _, move = minimax_ab(game, depth, float('-inf'), float('inf'), True, heuristic_fn)
    else:
        _, move = minimax(game, depth, True, heuristic_fn)
    return move


def generate_initial_board(rows=5, cols=6):
    board = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if (i + j) % 2 == 0:
                row.append('B')  # czarne pole
            else:
                row.append('W')  # białe pole
        board.append(row)
    return board



start_time = time.time()
nodes_visited = 0
rounds = 0

game = Clobber(generate_initial_board())
game.print_board()

while not game.is_terminal():
    heuristic = heuristic_2 if game.current_player == 'B' else heuristic_1
    value, move = minimax_ab(game, depth=3, alpha=float('-inf'), beta=float('inf'),
                              maximizing_player=True, heuristic=heuristic)
    if move is None:
        break
    game = game.make_move(move)
    nodes_visited += 1
    rounds += 1
    print(f"Ruch {rounds} ({game.get_opponent()}): {move}")
    game.print_board()

end_time = time.time()

print(f"Gra zakończona. Liczba rund: {rounds}")
print(f"Zwycięzca: {game.get_opponent()}")  # ostatni wykonany ruch = wygrany
print(f"Czas działania: {end_time - start_time:.3f} s")
print(f"Liczba odwiedzonych węzłów: {nodes_visited}")

