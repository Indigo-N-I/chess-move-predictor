import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from functions import AllowValid
from model import PieceSelection
from collections import defaultdict
import numpy as np
import os


games = pickle.load(open("win_loss.p", "rb"))
# model = PieceSelection()
# cwd = os.getcwd()
#
# string = cwd + f"\\model games_{500} epoch_{10}.pb"
# model.load_state_dict(torch.load(string))
#
#
# x = [data[0] for data in games]
y = [data[1] for data in games]
print(y.count(-1), y.count(1),y.count(0))

#
# criterion = nn.MSELoss()
#
# results = defaultdict(list)
#
for index, data in enumerate(x):
    output = model(torch.tensor([data]).type(torch.FloatTensor))
    loss =  criterion(output, torch.tensor([y[index]]).type(torch.FloatTensor))
    results[y[index]].append((loss, output))

    pickle.dump(results, open("result.p", "wb"))

results = pickle.load(open("result.p", 'rb'))

mse_result = {}
pred_result = {}

for key, values in results.items():
    mse_result[key] = [value[0] for value in values]
    pred_result[key] = [value[1] for value in values]

for key in mse_result.keys():
    non_torch = [value.detach().numpy() for value in mse_result[key]]
    mse_result[key] = non_torch
    print(f"Ananysis of {key}:\n\tMSE: {np.mean(non_torch)}, {np.std(non_torch)}")
    non_torch = [value.detach().numpy() for value in pred_result[key]]
    pred_result[key] = non_torch
    print(f"\tResults: {np.mean(non_torch)}, {np.std(non_torch)}")

for key, value in pred_result.items():
    flattened = np.array(value).flatten()

    zero = np.sum((flattened < .33) * (flattened > -.33) )
    one = np.sum(flattened > .33)
    neg = np.sum(flattened < -.33)

    print(f"{key}:\n\ttie {zero}\n\twhite {one}\n\tblack {neg}")
