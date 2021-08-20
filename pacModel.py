from moves import get_moves
from datetime import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import get_piece_moved, legal_start
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from personal_PAC import PAC
TESTMODEL = True

def legal_pieces(legal_moves, bit_boards, black = True):
    '''
    Translates the legal moves and bit boards into the pieces that can actually be moved
    6 - k
    5 - q
    4 - b
    3 - n
    2 - r
    1 - p
    '''
    ORDER = [6,5,4,3,2,1]
    legal = []
    location = legal_start(legal_moves)
    pos = [(7 - l // 8, l % 8) for l in location]
    # x = [l % 8 for l in location]
    # y = [8 - l // 8 for l in location]
    bit_boards = bit_boards[:7] if black else bit_boards[7:]
    for y, x in pos:
        for i in range(len(ORDER)):
            # print(ORDER[i])
            # print(bit_boards[i], y,x)
            if bit_boards[i][y][x]:
                # print("legal")
                legal.append(ORDER[i] - 1)
    return list(set(legal))

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2021, 7, 7)
    games = 500
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)
    # for board in black[0][0]:
    #     for line in board:
            # print(line)
    #     print('new bitboard')
    # print(legal_pieces(black[0][1], black[0][0]))
    # black.extend(white)

    if TESTMODEL:
        print("transorming data")
        train = black[:-len(black)//9]
        test = black[len(black)//9:]
        x_train = np.array([data[0] for data in train])
        x_test = np.array([data[0] for data in test])
        y_train = get_piece_moved([data[0] for data in train], legal_start([data[2] for data in train], False))
        y_test = get_piece_moved([data[0] for data in test], legal_start([data[2] for data in test], False))
        legal_train = []
        for t in train:
            legal_train.append(legal_pieces(t[1],t[0]))
        legal_test = []
        for t in test:
            legal_test.append(legal_pieces(t[1],t[0]))

        pac = PAC(cap_loss = .85)
        #def __init__(self, cap_loss = .65, decay = .001, decay_increase = .0001, loss_decay = False):
        pac.fit(x_train, y_train, legal_train)

        ans = pac.predict(x_test, legal_test)
        score = accuracy_score(y_test, ans)
        # print(set(y_test))
        con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
        print("With restrictions")
        print(score)
        print(con_mat)

        pac.fit(x_train, y_train)
        ans = pac.predict(x_test)
        score = accuracy_score(y_test, ans)
        con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
        print("no restrictions")
        print(score)
        print(con_mat)

        pac.fit(x_train, y_train, legal_train)
        ans = pac.predict(x_test)
        score = accuracy_score(y_test, ans)
        con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
        print("Training restrictions")
        print(score)
        print(con_mat)

        pac.fit(x_train, y_train)
        ans = pac.predict(x_test, legal_test)
        score = accuracy_score(y_test, ans)
        con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
        print("Test restrictions")
        print(score)
        print(con_mat)
