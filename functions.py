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
        Valid is the list of valid indexes
        '''
        # print(ctx)
        # print("inputs:", inputs, inputs.shape)
        ctx.save_for_backward(inputs)
        ctx.valid = valid

        for i in range(inputs.shape[0]):
            c = np.ones(inputs.shape)
            # print(c)
            c[:,ctx.valid] = 0
            d = np.ma.masked_array(inputs[i], mask = c)
            # print(d)
            # print("masked")
            # print(d.data)
            d = np.ma.exp(d)
            # print("exp")
            # print(d.data)
            d = d + 1
            # print("+1")
            # print(d.data)
            d = np.ma.log(d)
            # print("log")
            # print(d.data)
            # inputs[i] = np.ma.getdata(d)
        # print(ctx.valid[13], "<- to mask")
        # print(inputs[13], '<- result')
        # print(d.data)
        # print(inputs)
        # print(valid)
        # print(torch.tensor([d.data]).shape)
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
            d = np.ma.masked_array(grad_input[i], mask = c)
            d = d * -1
            d = np.ma.exp(d)
            d = d + 1
            d = d** -1
            # grad_input = np.ma.getdata(d)
        # print(grneruad_input.shape)
        # return torch.tensor(d.data), None
        return grad_output, None


if __name__ == "__main__":
    AV = AllowValid.apply
    # test = AV(3,4)
