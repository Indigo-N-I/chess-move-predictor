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
    # signal boosting?
    @staticmethod
    def forward(ctx, inputs, valid):
        '''
        Valid is the list of valid indexes
        '''
        # print(ctx)
        # print("inputs:", inputs, inputs.shape)
        ctx.save_for_backward(inputs)
        ctx.valid = valid

        for i in range(inputs.shape[0]):
            c = np.ones(inputs.shape)
            c[:,ctx.valid] = 0
            d = np.ma.masked_array(inputs[i], mask = c)
            d = np.ma.exp(d)
            d = d + 1
            d = np.ma.log(d)

        return torch.tensor([d.data])
        # return inputs

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # print(grad_output.shape)
        grad_input = grad_output.clone()
        for i in range(input.shape[0]):
            c = np.ones(input.shape)
            c[:,ctx.valid] = 0
            d = np.ma.masked_array(input[i], mask = c)
            d = np.ma.exp(d)
            # print("input:", grad_input)
            # print("D:", d)
            # print("data:", d.data)
            d = d/(d+1)
            # print("input:", grad_input)
            # print("D:", d)
            # print("data:", d.data)
            d = d * -1
            grad = d.data
            c = c*-1 + 1
            d = np.ma.masked_array(grad, mask = c)
            d = d/d
            # print(torch.tensor(d.data).shape)
            d = torch.tensor([d.data]) * grad_input
        # print(grneruad_input.shape)
        # print(torch.tensor(d))
        return d, None
        # return , None

def sigmoid(x):
    x = x * -1
    x = np.ma.exp9x)
    x = x + 1
    x = x ** -1
    return x

def sig_prime(x):
    x = sigmoid(x)
    y = x ** 2
    return y - x

def sig_double(x):
    x = sig_prime(x)
    y = sigmoid(x)
    return 2*x*y - x

if __name__ == "__main__":
    AV = AllowValid.apply
    # test = AV(3,4)
