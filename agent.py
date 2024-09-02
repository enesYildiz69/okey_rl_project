import torch
import torch.nn.functional as F
from model import DQNetwork
import random

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNetwork(state_size, action_size)
        self.target_model = DQNetwork(state_size, action_size)
        self.epsilon = config.epsilon_start
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self, minibatch, optimizer, criterion):
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                target = reward + 0.99 * torch.max(self.target_model(next_state))

            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target

            optimizer.zero_grad()
            loss = criterion(target_f, self.model(state))
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
