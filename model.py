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
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            self.mask
        )

    def forward(self, x, valid = ''):
        # print(x)
        x = self.flatten(x)
        self.mask.set_mask(valid)
        logits = self.linear_relu_stack(x)
        return logits

def translate_to_num(location):
    '''
    Translate standard chess square notation to integers between 0 and 63
    '''
    #A1 -> 0
    #B1 -> 1
    # print(location)
    add = ord(location[0]) - ord('a')
    # print(add)
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

def legal_start(moves):
    '''
    Given a list of moves find the valid starting position of moves

    returns an array of integers from 0-> 63
    '''
    legal = []
    for move in moves:
        legal.append(translate_to_num(move[:2]))

    return list(set(legal))

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

if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 500
    print("gathering games")

    white, black = get_moves('whoisis', start, end, games, split = True)
    for i in range(len(black)):
        black[i][0] *= -1

    # black.extend(white)
    # print(black[-1])
    print("transorming data")
    x = [data[0] for data in black]
    valid = [legal_start(data[1]) for data in black]
    target = torch.tensor(legal_end([data[2] for data in black]))
    # print(torch.tensor(x).shape)
    # print(print(len(valid)))

    print("creating network")
    test = PieceSelection()
    criterion = nn.CrossEntropyLoss()
    lr = .01
    opt = optim.SGD(test.parameters(), lr = .01)
    # print(test)

    print("training")
    for epoch in range(35):
        running_loss = 0.0
        opt = optim.SGD(test.parameters(), lr = lr*(.9**epoch))
        for index, data in enumerate(x):
            # print(valid[index])
            opt.zero_grad()
            output = test(torch.tensor([data]).type(torch.FloatTensor), valid[index])
            # print(output, "Target", torch.tensor([target[index]]).shape)
            loss = criterion(output, torch.tensor([target[index]]))
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if index % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, running_loss / 2000))
                running_loss = 0.0
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
