import time

from lab2.boardGeneration import generate_initial_board
from lab2.heuristic import *
from lab2.minimax import minimax
from lab2.minimaxAlphaBeta import minimax_ab

DEPTH = 4


def run_game(start_board, heuristic, search_depth=4, use_alphabeta=False, verbose=False):
    game = Clobber(start_board)
    rounds = 0
    nodes_visited = [0]
    start_time = time.time()

    while not game.is_terminal():
        maximizing = (game.current_player == 'B')
        if use_alphabeta:
            score, move = minimax_ab(game, search_depth, float('-inf'), float('inf'),
                                     maximizing, heuristic, nodes_visited)
        else:
            score, move = minimax(game, search_depth, maximizing, heuristic, nodes_visited)

        if move is None:
            break
        if (verbose):
            print(f"Chosen move: {move}")

        game = game.make_move(move)
        rounds += 1
        if verbose:
            print(f"Round {rounds} ({'B' if maximizing else 'W'} moves):")
            game.print_board()

    end_time = time.time()
    winner = game.get_opponent()

    print("Final board state:")
    game.print_board()
    print(f"Rounds played: {rounds}")
    print(f"Winner: {winner}")
    print(f"Execution time: {end_time - start_time:.3f}s")
    print(f"Visited nodes: {nodes_visited[0]}\n\n")


def run_test(heur_name, heuristic_func, depth=DEPTH, alphabeta=True):
    print(f"Test: 5x6 board | {heur_name} | depth {depth} | {'alpha-beta' if alphabeta else 'minimax'}")
    custom_board = [
            ['B', '_', 'W', '_', 'B', '_'],
            ['B', 'W', '_', 'B', 'B', 'W'],
            ['W', 'W', 'B', '_', 'W', 'B'],
            ['_', 'B', '_', 'W', '_', 'B'],
            ['B', 'W', 'W', '_', 'B', '_']
        ]
    board = generate_initial_board(5,6)
    run_game(board, heuristic_func, search_depth=depth, use_alphabeta=alphabeta, verbose=False)


if __name__ == '__main__':
    for heur_name, heur_func in HEURISTICS:
        run_test(heur_name, heur_func)

    # # Optional: test larger or custom boards
    # print("Test: 6x6 board, heuristic_2, depth 4, alpha-beta enabled")
    # run_game(generate_initial_board(6, 6), heuristic_2, search_depth=4, use_alphabeta=True)
    #
    # print("Test: Custom board, heuristic_3, depth 2, alpha-beta enabled")
    # custom_board = [
    #     ['B', '_', 'W', '_', 'B', '_'],
    #     ['_', 'W', '_', 'B', '_', 'W'],
    #     ['W', '_', 'B', '_', 'W', '_'],
    #     ['_', 'B', '_', 'W', '_', 'B'],
    #     ['B', '_', 'W', '_', 'B', '_']
    # ]
    # run_game(custom_board, heuristic_3, search_depth=2, use_alphabeta=True)
