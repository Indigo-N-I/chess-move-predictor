from moves import get_moves
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

class MaskingLayer(nn.Module):
    def __init__(self):
        super(MaskingLayer, self).__init__()

    # def __init__(self, mask):
    #     super(MaskingLayer, self).__init__()
    #     self.set_mask(mask)

    def forward(self, input):
        # print("doing allowvalid")
        b = AllowValid.apply
        a = b(input, self.to_mask)
        # print(a)
        return b(input, self.to_mask)

    def set_mask(self, mask):
        # print('set mask to:', mask)
        self.to_mask = mask

    def has_mask(self):
        if self.to_mask:
            return True
        return False

# most likely should rename piece selection to a more general name as
# it will be used twice
class PieceSelection(nn.Module):
    def __init__(self):
        super(PieceSelection, self).__init__()
        self.mask = MaskingLayer()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(.2),
            nn.LeakyReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x, valid = ''):
        # print(x)
        x = self.flatten(x)
        self.mask.set_mask(valid)
        logits = self.linear_relu_stack(x)
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
    # print(location)
    add = ord(location[0]) - ord('a')
    # print(add)
    # print(location, (int(location[1]) - 1)*8 + add  )
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
    # print(possible.size())
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
        # print(moves)
        # print(move)
        legal.append(translate_to_num(move[2:4]))

    return legal

def translate_to_pieces(board, spots):
    final_pieces = []
    print(len(board), len(spots))
    for i in range(len(board)):
        # print(spots[i], spots[i]//8, spots[i]%8)
        # print(translate_from_num(spots[i]))
        # print(board[i])
        a = abs(board[i][-(spots[i]//8+1)][spots[i]%8])
        final_pieces.append(a - 1)
        # print(f"adding {a}")
    return final_pieces


if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2021, 3, 7)
    games = 300
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)
    # for i in range(len(black)):
    #     black[i][0] *= -1

    # black.extend(white)
    # print(black[-1])
    print("transorming data")
    train, test = train_test_split(black)
    # print(train[0])

    x = [data[0] for data in train]
    valid = [legal_start(data[1]) for data in train]
    # print(translate_to_pieces(train[0], train[2]))
    # target = torch.tensor(legal_end([data[2] for data in train]))
    counts = translate_to_pieces([data[0] for data in train], legal_start([data[2] for data in train], False))

    print(Counter(counts))

    target = torch.tensor(translate_to_pieces([data[0] for data in train], legal_start([data[2] for data in train], False))).type(torch.LongTensor)

    x_test = [data[0] for data in test]
    valid_test = [legal_start(data[1]) for data in test]
    # target_test = torch.tensor(legal_end([data[2] for data in test]))
    target_test = torch.tensor(translate_to_pieces([data[0] for data in test], legal_start([data[2] for data in test], False))).type(torch.LongTensor)
    print(target_test)
    # print(torch.tensor(x).shape)
    # print(print(len(valid)))

    print("creating network")
    test = PieceSelection()
    criterion = nn.CrossEntropyLoss()
    lr = .01
    opt = optim.SGD(test.parameters(), lr = .01)
    # print(test)

    # cwd = os.getcwd()
    # torch.save(test.state_dict(), cwd)

    print("training")
    for epoch in range(35):
        running_loss = 0.0
        opt = optim.SGD(test.parameters(), lr = lr*(.9**epoch),weight_decay=1e-5)
        for index, data in enumerate(x):
            # print(valid[index])
            opt.zero_grad()
            output = test(torch.tensor([data]).type(torch.FloatTensor), valid[index])
            # print(torch.tensor(output).shape, output)
            # print(target.shape, output.shape)
            # print(target, index)
            # print(output, torch.tensor([target[index]]))
            loss = criterion(output, torch.tensor([target[index]]))
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if index % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0
        # print(f"epoch {epoch} results, index max: {index}")
        correct = 0
        quest = 0
        test_loss = 0
        for index, data in enumerate(x_test):
            output = test(torch.tensor([data]).type(torch.FloatTensor), valid_test[index])
            loss = criterion(output, torch.tensor([target_test[index]]))
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            if predicted[0] == target_test[index]:
                correct += 1
            quest += 1
        print(f"Correct Precent test: {correct/quest}")
        print(f"Test loss: {test_loss/index}")

    correct = 0
    quest = 0
    for index, data in enumerate(x):
        output = test(torch.tensor([data]).type(torch.FloatTensor), valid[index])
        _, predicted = torch.max(output, 1)
        if predicted[0] == target[index]:
            correct += 1
        quest += 1
    print(f"Correct Precent train: {correct/quest}")
    # cwd = os.getcwd()
    # torch.save(test.state_dict(), cwd)
            # print(loss.item())

            # running_loss += loss.item()
            # if




    # for j in range(1,9):
    #     for i in ['a','b','c','d','e','f','g','h']:
    #         print(translate_to_num(i+str(j)))

    # for k in range(64):
    #     print(translate_from_num(k))
    # print(black[0][1])
    # print(np.transpose((black[5][0]>0).nonzero()), '\n', black[5][0])
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # print('Using {} device'.format(device))
    # y = torch.rand(1, 8, 8, device=device)

    # model = PieceSelection().to(device)
    # X = torch.tensor([black[0][0]]).type(torch.FloatTensor)
    # print(X.shape)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    # print(pred_probab)
