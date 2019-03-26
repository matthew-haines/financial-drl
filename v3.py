import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

torch.manual_seed(1)
np.random.seed(1)

# Input to network in shape (timesteps, batch_size, features)


class Environment():

    def __init__(self, data, raw_data, timesteps=100, initial_capital=1000, initial_position=0.8, spread=0.5, fee=0.075):

        self.data = torch.Tensor(data).cuda()  # Shape (timesteps, features)
        self.raw_data = torch.Tensor(raw_data).cuda()
        self.initial_step = timesteps
        self.timesteps = timesteps

        self.total_steps = data.shape[0] - self.initial_step - 1  # Total steps

        self.cur_step = self.initial_step

        self.initial_capital = initial_capital  # In USD
        self.initial_position = initial_position

        self.cur_position = initial_position
        self.cur_capital = initial_capital

        self.positions = np.zeros((self.data.shape[0]))
        self.capital_values = np.zeros((self.data.shape[0]))
        self.position_changes = 0

        self.spread = spread
        self.fee = (100 - fee) / 100

        self.state = torch.zeros(timesteps, 1, self.data.shape[1]+2).cuda()

    def reset(self):

        self.cur_position = self.initial_position
        self.cur_capital = self.initial_capital
        self.cur_step = self.initial_step

        self.positions.fill(0.0)  # Reset to zeros
        self.capital_values.fill(0.0)  # Reset to zeros
        self.positions[:self.timesteps].fill(self.initial_position)
        self.capital_values[self.timesteps].fill(self.initial_capital)

        self.position_changes = 0

        self.generate_state()
        return self.state

    def generate_state(self):

        for i in range(0, self.timesteps):
            self.state[i, 0, 0] = self.positions[self.cur_step -
                                                 self.timesteps + i]
            self.state[i, 0, 1] = self.capital_values[self.cur_step -
                                                      self.timesteps + i]
            self.state[i, 0, 2:] = self.data[self.cur_step -
                                             self.timesteps + i, :]
            
        for i in range(2, 7): # Normalize with z-score
            mean = torch.average(self.state[:, 0, i])
            sigma = 0
            for j in range(self.timesteps):
                sigma += (self.state[j, 0, i] - mean) ** 2
            sigma = (sigma / self.timesteps) ** (1/2) # std_dev
            for j in range(self.timesteps):
                self.state[j, 0, i] = (self.state[j, 0, i] - mean) / sigma

    def get_account_value(self):

        return (self.cur_position**2)**(1/2) * self.raw_data[self.cur_step, 3]

    def step(self, action):
        # Must return reward and next step
        prev_acc_value = self.get_account_value()
        done = False
        if self.cur_step == self.data.shape[1]:
            done = True
        if action == 0:
            # Buy
            if self.cur_position < 0:
                # Determines position by dividing the acc value by the lowest price and multiplying by fee (.99925)
                self.cur_position = prev_acc_value / \
                    (self.raw_data[self.cur_step, 3] + self.spread) * self.fee

                self.position_changes += 1

        elif action == 1:
            # Sell
            if self.cur_position > 0:
                self.cur_position = - \
                    (prev_acc_value /
                     (self.raw_data[self.cur_step, 3] + self.spread) * self.fee)

                self.position_changes += 1

        self.cur_step += 1
        self.cur_capital = self.get_account_value()
        # If made money, reward positive, if not, negative
        reward = np.log(self.cur_capital / prev_acc_value)

        if self.get_account_value() <= self.initial_capital / 3:
            done = True

        # Adding stuff to position / value data
        self.positions[self.cur_step] = self.cur_position
        self.capital_values[self.cur_step] = self.cur_capital

        return self.state, reward, done


def copy_parameters(source_model, target_model):
    """Copies parameters from a source model to a target model. Used for target network soft update."""
    temp = source_model.state_dict()
    target_model.load_state_dict(temp)


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        # Shape (samples, timesteps, features)
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim).cuda()
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim).cuda()
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim).cuda()
        self.out = nn.Linear(self.hidden_dim, self.output_dim).cuda()

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, self.batch_size, self.hidden_dim).cuda())

    def forward(self, sample):
        self.hidden = self.init_hidden()

        l1_out = F.elu(self.l1(sample))
        l2_out = F.elu(self.l2(l1_out))
        lstm_out, self.hidden = self.lstm(l2_out, self.hidden)
        out = self.out(lstm_out)
        return out[99, 0, :]


class Agent():
    # Call as agent(args).cuda() for cuda performance
    def __init__(self, state_space, action_space, batch_size=1, replay_length=1024):
        self.batch_size = batch_size

        self.memory = deque(maxlen=replay_length)
        self.gamma = 0.99  # Discount Rate
        self.epsilon = 1.0  # Exploration Rate
        self.epsilon_decay = 0.9995  # Amount Epsilon Decay
        self.epsilon_min = 0.001
        self.alpha = 0.00025
        self.tau = 0.001  # For updating target network

        self.policy = Model(state_space, 256, action_space, 1).cuda()
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.alpha)
        self.target_policy = Model(state_space, 256, action_space, 1)
        copy_parameters(self.policy, self.target_policy)
        self.target_loss = nn.MSELoss()
        self.target_optimizer = optim.Adam(
            self.target_policy.parameters(), lr=self.alpha)

    def predict(self, x):
        return self.policy(x)

    def fit(self, x, y):
        self.policy.zero_grad()

        y_hat = self.policy(x)

        loss = self.loss(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        # Soft update
        net_dict = self.policy.state_dict()
        target_net_dict = self.target_policy.state_dict()
        for name, param in net_dict.items():
            target_net_dict[name] = (1 - self.tau) * target_net_dict[name] + self.tau * param

    def remember(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration
            return random.randrange(action_space)

        actions = self.predict(state)
        return torch.argmax(actions)

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)

        for state, action, next_state, reward, done in batch:
            target_actual = reward

            if not done:
                target_actual = reward + self.gamma * \
                    torch.argmax(self.predict(next_state))

            target = self.policy(state)

            loss = self.fit(state, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, path):
        torch.save(self.policy, 'code/models/'+path)

    def load(self):
        self.policy = torch.load('codemodels/1')


# Globals
timesteps = 100
state_space = 7  # OHLCV +  POSITION + CAPITAL
action_space = 2

n_simulations = 100

raw_data = np.loadtxt(
    '/Users/matthew/Projects/financial_drl/data/prepared_15.txt')
data = raw_data

env = Environment(data, raw_data, timesteps=timesteps, initial_capital=1000.0)
agent = Agent(state_space, action_space, batch_size=16, replay_length=480)

# Training
done = False
for sim in range(n_simulations):
    state = env.reset()
    losses = []
    for i in range(env.total_steps):
        if i % 100 == 0:
            cur_loss = np.average(losses)
            losses.clear()
            print('Step {}/{}, Loss: {}, Capital: {}, Positions: {}'.format(i,
                                                                            env.total_steps, cur_loss, env.cur_capital, env.position_changes))
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        reward = reward if not done else -10  # -1000% reward equivalent

        agent.remember(state, action, next_state, reward, done)
        state = next_state

        if done:
            print("Simulation {}/{}, Position changes: {}, Time Lasted: {}, Ending Capital: {}".format(
                sim, n_simulations, env.position_changes, env.cur_step-env.initial_step, env.cur_capital))
            break

        if len(agent.memory) > agent.batch_size:
            losses.append(agent.replay())
        
        agent.update_target()

    agent.save('sim')
