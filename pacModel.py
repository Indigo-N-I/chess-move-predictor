from moves import get_moves
from datetime import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import get_piece_moved, legal_start
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from personal_PAC import PAC

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2021, 7, 7)
    games = 500
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)

    # black.extend(white)

    print("transorming data")
    train = black[:-len(black)//9]
    test = black[len(black)//9:]
    x_train = np.array([data[0] for data in train])
    x_test = np.array([data[0] for data in test])
    y_train = get_piece_moved([data[0] for data in train], legal_start([data[2] for data in train], False))
    y_test = get_piece_moved([data[0] for data in test], legal_start([data[2] for data in test], False))

    pac = PAC(loss_decay = True)
    #def __init__(self, cap_loss = .65, decay = .001, decay_increase = .0001, loss_decay = False):
    pac.fit(x_train, y_train)

    ans = pac.predict(x_test)
    score = accuracy_score(y_test, ans)
    con_mat = confusion_matrix(y_test,ans, labels=list(set(y_test)))
    print(score)
    print(con_mat)
