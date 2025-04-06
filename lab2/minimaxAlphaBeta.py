def minimax_ab(game, depth, alpha, beta, maximizing_player, heuristic):
    if depth == 0 or game.is_terminal():
        return heuristic(game), None

    best_move = None

    if maximizing_player:
        max_eval = float('-inf')
        for move in game.get_valid_moves():
            eval, _ = minimax_ab(game.make_move(move), depth - 1, alpha, beta, False, heuristic)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in game.get_valid_moves():
            eval, _ = minimax_ab(game.make_move(move), depth - 1, alpha, beta, True, heuristic)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
