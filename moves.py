'''
file used to find the legal moves in a position
'''
import chess.pgn
import chess
from datetime import datetime
import io
import numpy as np
from get_games import get_games, split_bw

def get_bitboard():
    return [[0]*8,
                [0]*8,
                [0]*8,
                [0]*8,
                [0]*8,
                [0]*8,
                [0]*8,
                [0]*8]

def create_bitboards(board):
    #black pieces
    b_king = get_bitboard()
    b_queen = get_bitboard()
    b_bish = get_bitboard()
    b_horse = get_bitboard()
    b_rook = get_bitboard()
    b_pawn = get_bitboard()

    #white pieces
    w_king = get_bitboard()
    w_queen = get_bitboard()
    w_bish = get_bitboard()
    w_horse = get_bitboard()
    w_rook = get_bitboard()
    w_pawn = get_bitboard()

    piece_translation = {
        'p': b_pawn,
        'n': b_horse,
        'b': b_bish,
        'r': b_rook,
        'q': b_queen,
        'k': b_king,
        'P': w_pawn,
        'N': w_horse,
        'B': w_bish,
        'R': w_rook,
        'Q': w_queen,
        'K': w_king
    }

    for row, line in enumerate(board):
        for col, char in enumerate(line):
            if char in piece_translation:

                piece_translation[char][row][col//2] = 1


    return [
        b_king,
        b_queen,
        b_bish,
        b_horse,
        b_rook,
        b_pawn,
        w_king,
        w_queen,
        w_bish,
        w_horse,
        w_rook,
        w_pawn
    ]

# redfine to bitboards
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

    #just to collapse
    if 0:
        pass
        '''
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


        np_board = np.array(np_board)


        '''

    return create_bitboards(board)

def gen_all_next_moves(game, white: bool, not_dummy = True):
    # print(game)
    pgn = io.StringIO(game)
    game = chess.pgn.read_game(pgn)
    board = game.board()

    # board.legal_moves
    data = []
    save_move = white
    # a = 1 if white else 0
    for move in game.mainline_moves():


        if save_move:
            data = []
            for legal_move in board.legal_moves:
                dummy_board = board.copy()
                dummy_board.push(legal_move)
                if str(legal_move) != str(move):
                    data.append([process_board(str(dummy_board), not_dummy),0])
                else:
                    data.append([process_board(str(dummy_board), not_dummy),1])
            # print(to_move)
            # bitboard = process_board(str(board), not_dummy)
            # bitboard.append(to_move)
            # data.append([bitboard, legal, str(move)])

        board.push(move)
        save_move = not save_move

    return data

def win_loss_data_gen(game):
    pgn = io.StringIO(game)
    result = game[game.index("Result")+ 7:  game.index("Result") + 20]
    white_elo = game[game.index("WhiteElo") + 10: game.index("\"", game.index("WhiteElo") + 10)]
    black_elo = game[game.index("BlackElo") + 10: game.index("\"", game.index("BlackElo") + 10)]
    # print(white_elo)
    # print(black_elo)
    # print(game[game.index("BlackElo"):])
    white_elo = int(white_elo)/1000 if "?" not in white_elo else 1.5
    black_elo = int(black_elo)/1000 if "?" not in black_elo else 1.5
    # print(white_elo)
    # print(black_elo)
    if "0-1" in result:
        result = [1,0,0]
    elif "1/2" in result:
        result = [0,1,0]
    else:
        result = [0,0,1]
    game = chess.pgn.read_game(pgn)
    board = game.board()

    # board.legal_moves
    data = []
    a = 1
    black = np.array([[black_elo]*8]*8)
    white = np.array([[white_elo]*8]*8)
    for move in game.mainline_moves():

        to_move = np.array([[a]*8]*8)
        # print(to_move)
        bitboard = process_board(str(board))
        bitboard.append(to_move)
        bitboard.append(white)
        bitboard.append(black)
        data.append([bitboard, result])

        a *= -1

        board.push(move)

    return data

def gen_data(game, white: bool, not_dummy = True):
    pgn = io.StringIO(game)
    # print(pgn)
    # print(game)
    game = chess.pgn.read_game(pgn)
    board = game.board()

    # board.legal_moves
    data = []
    save_move = white
    a = 1 if white else 0
    for move in game.mainline_moves():


        if save_move:
            legal = []
            for legal_move in board.legal_moves:
                legal.append(str(legal_move))

            to_move = np.array([[a]*8]*8)
            # print(to_move)
            bitboard = process_board(str(board), not_dummy)
            bitboard.append(to_move)
            data.append([bitboard, legal, str(move)])
            # print([process_board(str(board), not_dummy), legal, str(move)])
            # print(data[0])
        # print(data)


        # print(move)
        board.push(move)
        save_move = not save_move
    # print(data[0])
    # print("data in gen games")
    return data

def gen_all(white_games, black_games, not_dummy = True):
    data = []
    data.extend(gen_games(white_games, True, not_dummy))
    # print(gen_games(white_games, True, not_dummy))
    # print("data")
    data.extend(gen_games(black_games, False, not_dummy))

    return dat

def gen_all_game_moves(white_games, black_games, not_dummy = True):
    data = []
    for game in white_games:
        data.extend(gen_all_next_moves(game, True, not_dummy))
    # print(gen_games(white_games, True, not_dummy))
    # print("data")
    for game in black_games:
        data.extend(gen_all_next_moves(game, False, not_dummy))

    return data


def gen_games(games, white, not_dummy = True):
    data = []

    for game in games:
        data.extend(gen_data(game, white, not_dummy))

    return data

def gen_win_loss(games):
    data = []
    for game in games:
        data.extend(win_loss_data_gen(game))

    return data

def get_moves(name, start, end, games, split = True, not_dummy = True):
    print("getting moves")
    retreaved_games = get_games('whoisis', start, end, games)

    white, black = split_bw(retreaved_games, 'whoisis')

    if split:
        # print(gen_games(black, False, not_dummy)[0])
        return gen_games(white, True, not_dummy), gen_games(black, False, not_dummy)

    data = gen_all(white, black, not_dummy)
    # print(data[0])
    # print("data is", data)
    return data

def get_all_moves(name, start, end, games, split = True, not_dummy = True):
    print("getting all moves")
    retreaved_games = get_games('whoisis', start, end, games)

    white, black = split_bw(retreaved_games, 'whoisis')

    if split:
        # print(gen_games(black, False, not_dummy)[0])
        return gen_games(white, True, not_dummy), gen_games(black, False, not_dummy)
    # print(white)
    data = gen_all_game_moves(white, black, not_dummy)
    return data

def get_win_loss(name, start, end, games):
    retreaved_games = get_games('whoisis', start, end, games)

    white, black = split_bw(retreaved_games, 'whoisis')

    data = gen_all_win_loss(white, black)
    return data

def gen_all_win_loss(white, black):

    data = []
    data.extend(gen_win_loss(white))
    data.extend(gen_win_loss(black))

    return data

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 13

    retreaved_games = get_games('whoisis', start, end, games)
    # for game in retreaved_games:
    #     print(game)
    #     break

    white, black = split_bw(retreaved_games, 'whoisis')
    # print(black)
    data = gen_all_win_loss(black, white)
