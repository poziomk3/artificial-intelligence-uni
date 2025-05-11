from Clobber import Clobber


def heuristic_1(game: Clobber):
    """Różnica liczby możliwych ruchów między graczem a przeciwnikiem"""
    own_moves = len(game.get_valid_moves())
    opponent_game = Clobber(game.board, game.get_opponent())
    opponent_moves = len(opponent_game.get_valid_moves())
    return own_moves - opponent_moves


def heuristic_2(game: Clobber):
    """Zlicza liczbę własnych pionków minus przeciwnika"""
    own = sum(row.count(game.current_player) for row in game.board)
    opponent = sum(row.count(game.get_opponent()) for row in game.board)
    return own - opponent


def heuristic_3(game: Clobber):
    """Liczba możliwych zbitych pionków (ataków)"""
    return len(game.get_valid_moves())  # im więcej możliwości bicia, tym lepiej


def heuristic_4(game: Clobber):
    """Potencjalna mobilność: liczba możliwych przyszłych ruchów (1 poziom w przód)"""
    future_moves = 0
    for move in game.get_valid_moves():
        next_state = game.make_move(move)
        future_moves += len(next_state.get_valid_moves())
    return future_moves


def heuristic_5(game: Clobber):
    """Izolacja pionków przeciwnika – im mniej przeciwnik ma sąsiadów, tym lepiej"""
    board = game.board
    opponent = game.get_opponent()
    rows, cols = len(board), len(board[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    isolated = 0

    for i in range(rows):
        for j in range(cols):
            if board[i][j] == opponent:
                has_neighbor = False
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols and board[ni][nj] == game.current_player:
                        has_neighbor = True
                        break
                if not has_neighbor:
                    isolated += 1
    return isolated


def heuristic_6(game: Clobber):
    """Centralizacja – premiuje pozycje bliższe centrum planszy"""
    rows = len(game.board)
    cols = len(game.board[0])
    center_i = rows / 2
    center_j = cols / 2
    score = 0

    for i in range(rows):
        for j in range(cols):
            if game.board[i][j] == game.current_player:
                distance = abs(i - center_i) + abs(j - center_j)
                score -= distance  # im bliżej centrum, tym lepiej
    return score


HEURISTICS = [
    ("heuristic_1", heuristic_1),
    ("heuristic_2", heuristic_2),
    ("heuristic_3", heuristic_3),
    ("heuristic_4", heuristic_4),
    ("heuristic_5", heuristic_5),
    ("heuristic_6", heuristic_6),
]
