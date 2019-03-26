import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Environment():

    def __init__(self, data, starting_capital=0.5, starting_step=10, initial_position=1):
        """starting_capital in BTC"""

        self.data = data # Features by timesteps
        self.initial_position = initial_position
        self.position = self.initial_position
        self.starting_capital = starting_capital
        self.capital = self.starting_capital
        self.current_step = starting_step
        self.state = torch.Tensor(self.data[self.current_step])

        self.fee = 1 - 0.000750 # Percentage
        self.spread = 0.5 # USD

        self.done = False

    def reset(self):

        self.current_step = self.initial_position
        self.capital = self.starting_capital
        self.state = torch.Tensor(self.data[self.current_step])

    def get_value(self):
        return (self.position ** 2) ** (1/2.0) * self.data[self.current_step] # At close

    def step(self, action):
        # Action space of (Buy, Sell), whichever one is highest gets picked
        if self.current_step == self.data.shape[1]:
            self.done = True
            new_state = self.data[self.current_step]
            self.state = new_state
            return new_state, "Done"

        if self.get_value() <= 0:
            self.done = True
            new_state = self.data[self.current_step]
            self.state = new_state
            return new_state, "Bankrupted"

        chosen_action = torch.argmax(action)
        if chosen_action == 0:
            # Buy Order
            old_capital = self.capital
            self.capital = self.get_value()
            reward = self.capital - old_capital
            self.position = self.fee * (self.capital - self.spread / self.data[self.current_step])
            new_state = self.data[self.current_step]
            self.state = new_state
            return new_state, "Bought"

        if chosen_action == 1:
            # Sell Order
            old_capital = self.capital
            self.capital = self.get_value()
            reward = self.capital - old_capital
            self.position = -(self.fee * (self.capital - self.spread / self.data[self.current_step]))
            new_state = self.data[self.current_step]
            self.state = new_state
            return new_state, "Sold"

        self.current_step += 1

class policy(nn.Module):

    def __init__(self, input_dim):
        super(policy, self).__init__()
        self.l1 = nn.Linear(input, 128)
        self.l2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 2)

    def forward(self, x)
        x = F.elu(self.l1(x))
        x = F.elu(self.l2(x))
        return F.sigmoid(self.output(x))