from moves import get_moves, get_all_moves, get_win_loss
from datetime import datetime
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from functions import AllowValid
from sklearn.model_selection import train_test_split
from collections import Counter
from se_module import SELayer
import pickle
from collections import defaultdict

modifiers = "categories"
FILTERS = 128
BLOCKS = 10
GATHERING_DATA = False

class PieceSelection(nn.Module):
    def __init__(self):
        super(PieceSelection, self).__init__()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(8192, 3)
        self.conv = nn.Conv2d(15,FILTERS,kernel_size = 3, padding = 1)
        self.SE1 = SELayer(FILTERS, 32)
        self.SE2 = SELayer(FILTERS, 32)
        self.SE3 = SELayer(FILTERS, 32)

    def forward(self, x, valid = ''):
        x = self.conv(x)
        x = self.SE1(x) + self.SE2(x)
        # x2 = self.SE2(x)
        # x3 = self.SE3(x)
        # x4 = self.SE4(x)
        # x5 = self.SE5(x)
        # x6 = self.SE6(x)
        # x7 = self.SE7(x)
        # x8 = self.SE8(x)
        # x9 = self.SE9(x)
        # x10 = self.SE10(x)

        # x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10

        # print(x.shape)
        x = self.flatten(x)
        # self.mask.set_mask(valid)
        # print("shape of x is:", x.shape)
        logits = self.relu(x)
        # print("shape of logits is:", logits.shape)
        # print(self.linear.weight)
        logits = self.linear(logits)
        # print("shape of logits is:", logits.shape)
        return logits

# try this:
# Choose a piece (like k/r/p/q/b instead of square) then
# predict the square to go to

def translate_to_num(location):
    '''
    Translate standard chess square notation to integers between 0 and 63
    '''
    #A1 -> 0
    #B1 -> 1

    add = ord(location[0]) - ord('a')


    return (int(location[1]) - 1)*8 + add
    # pass

def translate_from_num(location):
    '''
    Translate integers between 0 and 63 to standard chess square notation
    '''
    row = location // 8 + 1
    col = chr(location % 8 + ord('a'))

    return col + str(row)


def piece_select_loss(output, valid, target):
    '''
    output - output of nerual netowrk
    valid  - possible selection of the piece in number form i.e. A1 is 0 and H8 is 63, is a list
    target - the true piece to be selected
    '''
    allowed_starts = [translate_to_num(move[:2]) for move in valid]
    possible = allow_only_valid(output, allowed_starts)
    c = nn.CrossEntropyLoss()

    return c(possible, torch.tensor([translate_to_num(target[:2])]))

def legal_start(moves, to_set = True):
    '''
    Given a list of moves find the valid starting position of moves

    returns an array of integers from 0-> 63
    '''
    legal = []
    for move in moves:
        legal.append(translate_to_num(move[:2]))

    return list(set(legal)) if to_set else legal

def legal_end(moves):
    '''
    Given a list of legal moves and the starting location of the move, finds valid end positions of the move

    returns an array of integers from 0->63 for leagal final piece location
    '''
    legal = []
    for move in moves:


        legal.append(translate_to_num(move[2:4]))

    return legal

def get_piece_moved(board, spots):
    final_pieces = []
    print(len(board), len(spots))
    for i in range(len(board)):


        # for bitboard in board[i]:
        #     print(np.array(bitboard))


        # print(board)
        bit_board_vals = [board[i][j][-(spots[i]//8+1)][spots[i]%8] for j in range(len(board[i]))]
        final_pieces.append(bit_board_vals.index(1)%6)


    return final_pieces


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
      dev = "cuda:0"
    else:
      dev = "cpu"
    print(f"using {dev}")

    if GATHERING_DATA:
        start = datetime(2018, 12, 8)
        end = datetime(2021, 3, 7)
        game_num = 500
        print("gathering games")

        games = get_win_loss('whoisis', start, end, game_num)

        print("transorming data")
    else:
        games = pickle.load(open("win_loss.p", "rb"))
    train, test = train_test_split(games)

    x = [data[0] for data in train]
    y = [tuple(data[1]) for data in train]
    # print(f"Total amount of 1's train: {sum(y)}/{len(y)}")
    print(y)

    print(len(train))

    x_test = [data[0] for data in test]
    y_test = [tuple(data[1]) for data in test]
    # print(f"Total amount of 1's test: {sum(y_test)}/{len(y_test)}")


    print("creating network")
    test = PieceSelection()

    # cwd = os.getcwd()
    #
    # string = cwd + f"\\model games_{500} epoch_{10}.pb"
    #
    # test.load_state_dict(torch.load(string))
    criterion = nn.CrossEntropyLoss()
    lr = .01
    opt = optim.Rprop(test.parameters(), lr = .01)

    print("training")
    for epoch in range(36):
        running_loss = 0.0
        opt = optim.SGD(test.parameters(), lr = lr*(.9**epoch),weight_decay=1e-5)
        for index, data in enumerate(x):
            # print(index)
            opt.zero_grad()
            output = test(torch.tensor([data]).type(torch.FloatTensor))
            # print(len(x))
            # print(len(y))


            # print(output)
            # print(torch.tensor(y[index]).type(torch.FloatTensor))
            loss = criterion(output, torch.tensor([y[index]]).type(torch.FloatTensor))
            # print(loss)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if index % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0

        correct = 0
        quest = 0
        zeros = 0
        ones = 0
        test_loss = 0
        total = []
        results = defaultdict(list)
        if epoch % 5 == 0:
            for index, data in enumerate(x_test):
                output = test(torch.tensor([data]).type(torch.FloatTensor))
                loss =  criterion(output, torch.tensor([y[index]]).type(torch.FloatTensor))
                results[y[index]].append((loss, output))

            pickle.dump(results, open(f"result epoch_{epoch} {modifiers}.p", "wb"))
            # print(f"Correct Precent test: {correct/quest}")
            # print(f"Test loss: {test_loss/index}")
            # # print(f"Predicted precent: {correct/quest}\n0: {zeros}\n1: {ones}")
            # print(f"Stats: {np.mean(total)}, {np.std(total)}")
            cwd = os.getcwd()
            string = cwd + f"\\model games_{game_num} epoch_{epoch} {modifiers}.pb"

            torch.save(test.state_dict(), string)
            print(f"ssaved as {string}")

    correct = 0
    quest = 0
    for index, data in enumerate(x):
        output = test(torch.tensor([data]).type(torch.FloatTensor), valid[index])
        _, predicted = torch.max(output, 1)
        if predicted[0] == target[index]:
            correct += 1
        quest += 1
    print(f"Correct Precent train: {correct/quest}")
    # torch.save(the_model.state_dict(), )


            # running_loss += loss.item()
            # if




    # for j in range(1,9):
    #     for i in ['a','b','c','d','e','f','g','h']:
    #         print(translate_to_num(i+str(j)))

    # for k in range(64):
    #     print(translate_from_num(k))


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #

    # y = torch.rand(1, 8, 8, device=device)

    # model = PieceSelection().to(device)
    # X = torch.tensor([black[0][0]]).type(torch.FloatTensor)

    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
