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
