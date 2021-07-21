from moves import get_moves
from datetime import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import get_piece_moved, legal_start
from sklearn.linear_model import PassiveAggressiveClassifier

from personal_PAC import PAC

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2021, 7, 7)
    games = 20
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)

    # black.extend(white)

    print("transorming data")
    train, test = train_test_split(black)
    x_train = np.array([data[0] for data in train])
    x_test = np.array([data[0] for data in test])
    y_train = get_piece_moved([data[0] for data in train], legal_start([data[2] for data in train], False))
    y_test = get_piece_moved([data[0] for data in test], legal_start([data[2] for data in test], False))

    pac = PAC()
    pac.fit(x_train, y_train)
