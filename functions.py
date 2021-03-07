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
        ctx.save_for_backward(inputs)
        ctx.valid = valid

        for i in range(inputs.shape[0]):
            c = np.zeros(inputs[i].shape)
            c[ctx.valid[i]] = 1
            d = np.ma.masked_array(inputs[i], mask = c)
            d *= 3
        # print(ctx.valid[13], "<- to mask")
        # print(inputs[13], '<- result')

        return inputs

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # print(grad_output.shape)
        grad_input = grad_output.clone()
        for i in range(input.shape[0]):
            c = np.zeros(input[i].shape)
            c[ctx.valid[i]] = 1
            d = np.ma.masked_array(grad_input[i], mask = c)
            d *= 3
        # print(grad_input.shape)
        return grad_input, None


if __name__ == "__main__":
    AV = AllowValid.apply
    # test = AV(3,4)
