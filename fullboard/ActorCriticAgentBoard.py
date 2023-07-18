import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.nn.utils as nn_utils
from torch.distributions import Categorical

#Actor que retorna una distribución de probabilidades de acciones
class Actor(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(hidden_size * 7 * 7, hidden_space1)
        self.fc_layer2 = nn.Linear(hidden_space1, hidden_space1)
        self.fc_layer3 = nn.Linear(hidden_space1, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, output_size)
        self.dropout = nn.Dropout(0.5)

        init.kaiming_uniform_(self.fc_layer1.weight)
        init.kaiming_uniform_(self.fc_layer2.weight)
        init.kaiming_uniform_(self.fc_layer3.weight)
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        #x = F.relu(self.fc_layer3(x))
        #x = self.dropout(x)
        x = self.out_layer(x)
        x = F.softmax(x, dim=-1)

        return x

#Crítico que evalua el valor de la transición
class Critic(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super().__init__()

        hidden_space1 = 512
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc_layer1 = nn.Linear(64 * 7 * 7, hidden_space1)
        self.dropout = nn.Dropout(0.5)
        self.fc_layer2 = nn.Linear(hidden_space1, hidden_space1)
        self.fc_layer3 = nn.Linear(hidden_space1, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, 1)

        init.kaiming_uniform_(self.fc_layer1.weight)
        init.kaiming_uniform_(self.fc_layer2.weight)
        init.kaiming_uniform_(self.fc_layer3.weight)
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        #x = F.relu(self.fc_layer3(x))
        #x = self.dropout(x)
        x = self.out_layer(x)

        return x

#Agente que implementa las clases Actor y Crítico
class ActorCriticAgent:
    def __init__(self, input_channels, hidden_size, output_size, learning_rate, gamma):
        self.actor = Actor(input_channels, hidden_size, output_size)
        self.critic = Critic(input_channels, hidden_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=0.0001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=0.0001)
        self.gamma = gamma
        self.eps = 1e-8
        self.eps_exp_end = 0.1
        self.eps_exp_start = 1.0
        self.eps_exp_anneal_steps = 10000
        self.eps_exp_curr = self.eps_exp_start
        self.exploration_noise = 0.1
        self.saved_log_probs = []
        self.rewards = []
        self.saved_values = []
        self.selected_actions = []
        self.actor_losses = []
        self.critic_losses = []
        self.output_size = output_size

    def act(self, state):
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        
        action_probs = self.actor(state)

        #noise injection
        noisy_action_probs = action_probs + torch.rand_like(action_probs)*self.exploration_noise

        distrib = Categorical(noisy_action_probs)

        action = distrib.sample()

        #e-gredy
        p = np.random.random()
        if p < self.eps_exp_curr:
            action = torch.tensor(np.random.choice(range(self.output_size)))

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

        #normalized returns, may affect training when winning suddenly
        #returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value
            advantage = advantage.detach()

            # calculate actor (policy) loss
            actor_loss.append(-log_prob * advantage)

            # calculate critic (value) loss using MSE 
            critic_loss.append(F.mse_loss(value, R))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor = torch.stack(actor_loss).sum()
        loss_critic = torch.stack(critic_loss).sum()
        
        #self.actor_losses.append(loss_actor)
        #self.critic_losses.append(loss_critic)
        loss_actor.backward()
        loss_critic.backward() 

        max_grad_norm = 1.0  # Maximum gradient norm
        nn_utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        nn_utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)

        self.optimizer_actor.step()               
        self.optimizer_critic.step()

        self.eps_exp_curr = np.maximum(self.eps_exp_end, self.eps_exp_curr - (self.eps_exp_start - self.eps_exp_end) / self.eps_exp_anneal_steps)
        self.selected_actions = []
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

    def get_actor_training_loss(self):
        return [tensor.item() for tensor in self.actor_losses] 

    def get_critic_training_loss(self):
        return [tensor.item() for tensor in self.critic_losses]

    def train(n_steps):
        pass