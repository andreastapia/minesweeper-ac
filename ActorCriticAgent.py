import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

## Combina el Actor y Crítico en una sola red, no se usa actualmente
class ACPolicy(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(hidden_size * 9 * 9, hidden_space1)
        self.out_layer_actor = nn.Linear(hidden_space1, output_size)
        self.out_layer_critic = nn.Linear(hidden_space1, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))

        action_probs = torch.softmax(self.out_layer_actor(x), dim=0)
        value = self.out_layer_critic(x)

        return action_probs, value

#Actor que retorna una distribución de probabilidades de acciones
class Actor(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        hidden_space1 = 512
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.fc_layer1 = nn.Linear(hidden_size * 3 * 3, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.out_layer(x))
        x = torch.softmax(x, dim=0)

        return x

#Crítico que evalua el valor de la transición
class Critic(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super().__init__()
        hidden_space1 = 512
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.fc_layer1 = nn.Linear(hidden_size * 3 * 3, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = self.out_layer(x)

        return x

#Agente que implementa las clases Actor y Crítico
class ActorCriticAgent:
    def __init__(self, input_channels, hidden_size, output_size, learning_rate, gamma):
        self.actor = Actor(input_channels, hidden_size, output_size)
        self.critic = Critic(input_channels, hidden_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.eps = 1e-8
        self.eps_exp = 0.01
        self.saved_log_probs = []
        self.rewards = []
        self.saved_values = []
        self.selected_actions = []
        self.output_size = output_size

    def act(self, state):
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        
        action_probs = self.actor(state)
        distrib = Categorical(action_probs)

        # available_actions = set(range(self.output_size)) - set(self.selected_actions)  # Exclude selected actions
        # if len(available_actions) == 0:
        #     # If all actions have been selected, return a random action
        #     return np.random.choice(range(self.output_size))
        action = distrib.sample()

        p = np.random.random()
        if p < self.eps_exp:
            action = torch.tensor(np.random.choice(range(self.output_size)))

        # while action.item() not in available_actions:
        #     action = distrib.sample()

        self.saved_log_probs.append(distrib.log_prob(action))
        self.saved_values.append(self.critic(state))
        action = action.item()
        self.selected_actions.append(action)
        return action

    def update(self):
        actor_loss = []
        critic_loss = []

        R = 0
        returns = []
        #se calcula el retorno acumulado para cada accion realizada
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

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor = torch.stack(actor_loss).sum()
        loss_critic = torch.stack(critic_loss).sum()
        loss = loss_actor + loss_critic
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        self.selected_actions = []
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

    def train(n_steps):
        pass