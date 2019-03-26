import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class environment():

    def __init__(self, data, action=2, starting_capital=10, initial_position=1000, series_length=5, use_cuda=True, spread=0.5,
                 taker_fee=0.0750, inaction_penalty=0, starting_point=0, position_encode=1):

        if use_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.starting_capital = starting_capital
        self.initial_position = initial_position
        self.series_length = series_length
        self.cur_step = self.initial_position

        # Data shape should be (sample_size, feature_dimension)
        self.data = data
        self.state = torch.FloatTensor(torch.zeros(data.shape[1] + 2)) # 2 for position BTC and portfolio value

        self.state[0] = self.initial_position
        self.state[1] = self.starting_capital
        self.data_to_state()

        # Simulation shit
        self.spread = spread
        # Converts fee given as percentage to coefficient to calculate fee
        self.fee_coefficient = 1 - (taker_fee / 100)
        self.inaction_penalty = inaction_penalty
        self.done = False

    def data_to_state(self):
        x = 2
        for i in range(self.series_length):
            for j in range(self.data.shape[1]):
                self.state[x] = self.data[self.cur_step - self.series_length + i, j]

    def portfolio_val(self):
        return (self.state)

    def reset(self):

        self.capital = self.starting_capital
        self.cur_step = self.starting_point
        self.position = self.initial_position

        self.state = self._set_state(self.starting_point)

    def step(self, action):
        # Action should be index of the action taken, 0 = buy, 1 = sell, 2 = do nothing, action should also have amount
        cur_timestep = self.cur_step
        steps_left = self.data.shape[1] - cur_timestep
        
        if action[0] ==  0:
            if self.data[cur_timestep, 0] * action[1] > self.capital:
                self.position

        self.curstep += 1
        