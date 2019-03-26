import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from model import Model

policy = Model(32, 64, 2, 1).cuda()
loss_func = nn.MSELoss()
optimizer = optim.Adam(policy.parameters())

losses = []

for epoch in range(100):
    x = torch.randn(8, 1, 32).cuda()
    y = torch.randn(1, 1, 2).cuda()

    policy.zero_grad()
    policy.hidden = policy.init_hidden()

    y_hat = policy(x)

    loss = loss_func(y_hat, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

print(losses)
