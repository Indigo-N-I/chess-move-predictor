'''
file used to find the legal moves in a position
'''
import chess.pgn
import chess
from datetime import datetime
import io
import numpy as np
from get_games import get_games, split_bw

def process_board(iBoard, not_dummy = True):
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
        'n': -2 if not_dummy else -1,
        'b': -3 if not_dummy else -1,
        'r': -4 if not_dummy else -1,
        'q': -5 if not_dummy else -1,
        'k': -6 if not_dummy else -1,
        'P': 1 if not_dummy else 1,
        'N': 2 if not_dummy else 1,
        'B': 3 if not_dummy else 1,
        'R': 4 if not_dummy else 1,
        'Q': 5 if not_dummy else 1,
        'K': 6 if not_dummy else 1
    }
    for index, line in enumerate(board):
        np_board.append([])
        for char in line:
            if char in piece_translation:
                np_board[index].append(piece_translation[char])
            # print(np_board)
    # print(iBoard.legal_moves)
    np_board = np.array(np_board)
    # print(np_board)
    # print(iBoard)
    return np_board

def gen_data(game, white: bool, not_dummy = True):
    pgn = io.StringIO(game)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    # print(board)
# board.legal_moves
    data = []
    save_move = white
    for move in game.mainline_moves():
        # print(move)
        # print(board.legal_moves)
        if save_move:
            legal = []
            for legal_move in board.legal_moves:
                legal.append(str(legal_move))

            data.append([process_board(str(board), not_dummy), legal, str(move)])

        board.push(move)
        save_move = not save_move

    return data

def gen_all(white_games, black_games, not_dummy = True):
    data = []
    data.extend(gen_games(white_games, True, not_dummy))
    data.extend(gen_games(black_games, False, not_dummy))

    return data

def gen_games(games, white, not_dummy = True):
    data = []

    for game in games:
        data.extend(gen_data(game, white, not_dummy))

    return data

def get_moves(name, start, end, games, split = True, not_dummy = True):
    retreaved_games = get_games('whoisis', start, end, games)

    white, black = split_bw(retreaved_games, 'whoisis')

    if split:
        return gen_games(white, True, not_dummy), gen_games(black, False, not_dummy)

    data = gen_all(white, black, not_dummy)

    return data

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 2

    retreaved_games = get_games('whoisis', start, end, games)
    # print(type(retreaved_games), type(1))
    white, black = split_bw(retreaved_games, 'whoisis')
    # print(white[0].index('1. '))
    data = gen_all(white, black)
    # print(data)
