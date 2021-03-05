from moves import get_moves
from datetime import datetime
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

class AllowValid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, valid):
        '''
        Valid is the valid indexes
        '''
        # print(ctx)
        ctx.save_for_backward(inputs)
        return allow_only_valid(inputs, vallid)

    def allow_only_valid(self, all, valid):
        '''
        Returns only the valid positions as values, all else are set to 0
        '''
        return all.detach() * np.isin(np.arange(0,64), valid)

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


###### needs to be put into custom file for custom pytorch autograd function
def piece_select_loss(output, valid, target):
    '''
    output - output of nerual netowrk
    valid  - possible selection of the piece in number form i.e. A1 is 0 and H8 is 63, is a list
    target - the true piece to be selected
    '''
    allowed_starts = [translate_to_num(move[:2]) for move in valid]
    possible = allow_only_valid(output, allowed_starts)
    c = nn.CrossEntropyLoss()
    print(possible.size())
    return c(possible, torch.tensor([translate_to_num(target[:2])]))

def legal_start(moves):
    '''
    Given a list of moves find the valid starting position of moves

    returns an array of integers from 0-> 63
    '''
    legal = []
    for move in moves:
        legal.append(translate_to_num(move[:2]))

    return legal

def legal_end(moves, start):
    '''
    Given a list of legal moves and the starting location of the move, finds valid end positions of the move

    returns an array of integers from 0->63 for leagal final piece location
    '''
    legal = []
    for move in moves:
        if moves[:2] == start:
            legal.append(translate_to_num[-2:])

    return legal
### everything above needs to be put into a new file


if __name__ == "__main__":
    AV = AllowValid.apply
    # test = AV(3,4)
