import copy


class Clobber:
    def __init__(self, board, current_player='B'):
        self.board = board  # plansza jako lista list
        self.current_player = current_player  # 'B' albo 'W'

    def get_opponent(self):
        return 'W' if self.current_player == 'B' else 'B'

    def get_valid_moves(self):
        moves = []
        rows = len(self.board)
        cols = len(self.board[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # góra, dół, lewo, prawo

        for i in range(rows):
            for j in range(cols):
                if self.board[i][j] == self.current_player:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if self.board[ni][nj] == self.get_opponent():
                                moves.append(((i, j), (ni, nj)))
        return moves

    def make_move(self, move):
        (x1, y1), (x2, y2) = move
        new_board = copy.deepcopy(self.board)
        new_board[x2][y2] = self.current_player
        new_board[x1][y1] = '_'
        return Clobber(new_board, self.get_opponent())

    def is_terminal(self):
        return len(self.get_valid_moves()) == 0

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()
