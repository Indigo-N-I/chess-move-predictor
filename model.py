from moves import get_moves
from datetime import datetime
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

class PieceSelection(nn.Module):
    def __init__(self):
        super(PieceSelection, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )

    def forward(self, x):
        print(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":

    start = datetime(2018, 12, 8)
    end = datetime(2020, 12, 9)
    games = 20

    white, black = get_moves('whoisis', start, end, games, split = True)
    for i in range(len(black)):
        black[i][0] *= -1

    test = PieceSelection()
    opt = optim.SGD(test.parameters(), lr = .01)

    for i in range(20):
        opt.zero_grad()
        output = test(torch.tensor([black[0][0]]).type(torch.FloatTensor))
        loss = piece_select_loss(output, black[0][1], black[0][2])
        loss.backward()
        opt.step()




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
