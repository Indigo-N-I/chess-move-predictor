from moves import get_moves
from datetime import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import get_piece_moved, legal_start
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import random


if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2021, 7, 7)
    games = 500
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)
    train, test = train_test_split(black)


    train = black[:-len(black)//9]
    test = black[len(black)//9:]
    x_train = np.array([data[0] for data in train])
    x_test = np.array([data[0] for data in test])
    y_train = get_piece_moved([data[0] for data in train], legal_start([data[2] for data in train], False))
    y_test = get_piece_moved([data[0] for data in test], legal_start([data[2] for data in test], False))

    ans = []
    for board in x_test:
        ans.append(random.randint(0,5))

    score = accuracy_score(y_test, ans)
    con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
    print("randoms")
    print(score)
    print(con_mat)
