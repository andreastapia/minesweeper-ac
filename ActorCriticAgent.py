import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ACPolicy(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(hidden_size * 9 * 9, hidden_space1)
        self.out_layer_actor = nn.Linear(hidden_space1, output_size)
        self.out_layer_critic = nn.Linear(hidden_space1, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))

        action_probs = torch.softmax(self.out_layer_actor(x), dim=0)
        value = self.out_layer_critic(x)

        return action_probs, value

# Define the Actor-Critic agent
class ActorCriticAgent:
    def __init__(self, input_channels, hidden_size, output_size, learning_rate, gamma):
        self.policy = ACPolicy(input_channels, hidden_size, output_size)
        #self.critic = Critic(input_channels, hidden_size)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        #self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps = 1e-8
        self.saved_log_probs = []
        self.rewards = []
        self.saved_values = []

    def act(self, state):
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        
        action_probs, value = self.policy(state)
        distrib = Categorical(action_probs)
        action = distrib.sample()
        self.saved_log_probs.append(distrib.log_prob(action))
        self.saved_values.append(value)
        action = action.numpy()

        return action

    def update(self):
        actor_loss = []
        critic_loss = []

        R = 0
        returns = []
        #se calcula el retorno acumulado por cada accion realizada
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value

            # calculate actor (policy) loss
            actor_loss.append(-log_prob * advantage)

            # calculate critic (value) loss using MSE smooth loss
            critic_loss.append(F.mse_loss(value[0], R))

        self.policy_optimizer.zero_grad()
        loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
        loss.backward()
        self.policy_optimizer.step()

        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

    def train(n_steps):
        pass