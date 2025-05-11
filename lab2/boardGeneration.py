def generate_initial_board(rows=5, cols=6):
    board = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if (i + j) % 2 == 0:
                row.append('B')
            else:
                row.append('W')
        board.append(row)
    return board
