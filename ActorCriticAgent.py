import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(hidden_size * 9 * 9, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.out_layer(x))
        x = torch.softmax(x, dim=0)

        return x


class Critic(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super().__init__()
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(hidden_size * 9 * 9, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = self.out_layer(x)

        return x

# Define the Actor-Critic agent
class ActorCriticAgent:
    def __init__(self, input_channels, hidden_size, output_size, learning_rate, gamma):
        self.actor = Actor(input_channels, hidden_size, output_size)
        self.critic = Critic(input_channels, hidden_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def act(self, state):
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        
        action_probs = self.actor(state)
        distrib = Categorical(action_probs)
        action = distrib.sample()
        self.saved_log_probs.append(distrib.log_prob(action))
        action = action.numpy()

        return action

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.actor(state)
        m = Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        actor_loss = []
        critic_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            actor_loss.append(-log_prob * R)
            ##montecarlo critic training approach
            critic_loss.append((R - self.critic(torch.from_numpy(state).float().unsqueeze(0))) ** 2)

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor = torch.stack(actor_loss).sum()
        loss_critic = torch.stack(critic_loss).sum()
        loss = loss_actor + loss_critic
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.saved_log_probs = []
        self.rewards = []