'''
file used to find the legal moves in a position
'''
import chess.pgn
import chess
from datetime import datetime
import io
import numpy as np

def process_board(iBoard):
    '''
    pawn = 1
    knight = 2
    bishop = 3
    rook = 4
    queen = 5
    king = 6

    black is just negative
    '''

    board = io.StringIO(iBoard)
    np_board = []
    piece_translation = {
        '.': 0,
        'p': -1,
        'n': -2,
        'b': -3,
        'r': -4,
        'q': -5,
        'k': -6,
        'P': 1,
        'N': 2,
        'B': 3,
        'R': 4,
        'Q': 5,
        'K': 6
    }
    for index, line in enumerate(board):
        np_board.append([])
        for char in line:
            if char in piece_translation:
                np_board[index].append(piece_translation[char])
            # print(np_board)

    np_board = np.array(np_board)
    # print(np_board)
    # print(iBoard)
    return np_board


def gen_data(game, white: bool):
    pgn = io.StringIO(game)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    # print(board)
# board.legal_moves
    data = []
    save_move = white
    for move in game.mainline_moves():
        # print(board)
        if save_move:
            legal = []
            for legal_move in board.legal_moves:
                legal.append(str(legal_move))

            data.append((process_board(str(board)), legal, str(move)))

        board.push(move)
        save_move = not save_move

    return data

def gen_all(white_games, black_games):
    data = []
    for game in white_games:
        data.extend(gen_data(game, white = True))

    for game in black_games:
        data.extend(gen_data(game, white = False))

    return data

if __name__ == "__main__":
    from get_games import get_games, split_bw

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 2

    retreaved_games = get_games('whoisis', start, end, games)
    # print(type(retreaved_games), type(1))
    white, black = split_bw(retreaved_games, 'whoisis')
    # print(white[0].index('1. '))
    data = gen_all(white, black)
    print(data)
