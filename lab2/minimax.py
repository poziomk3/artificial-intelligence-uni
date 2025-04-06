def minimax(game, depth, maximizing_player, heuristic):
    if depth == 0 or game.is_terminal():
        return heuristic(game), None

    best_move = None

    if maximizing_player:
        max_eval = float('-inf')
        for move in game.get_valid_moves():
            eval, _ = minimax(game.make_move(move), depth - 1, False, heuristic)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in game.get_valid_moves():
            eval, _ = minimax(game.make_move(move), depth - 1, True, heuristic)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move
