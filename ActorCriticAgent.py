import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.init as init

#Actor que retorna una distribución de probabilidades de acciones
class Actor(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.fc_layer1 = nn.Linear(hidden_size * 3 * 3, hidden_space1)
        self.fc_layer2 = nn.Linear(hidden_space1, hidden_space1)
        self.fc_layer3 = nn.Linear(hidden_space1, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.fc_layer2(x))
        x = torch.relu(self.fc_layer3(x))
        x = self.dropout(x)
        x = self.out_layer(x)
        x = torch.softmax(x, dim=0)
        return x

#Crítico que evalua el valor de la transición
class Critic(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super().__init__()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        hidden_space1 = 128
        self.conv1 = nn.Conv2d(input_channels,hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0)
        self.fc_layer1 = nn.Linear(hidden_size * 3 * 3, hidden_space1)
        self.dropout = nn.Dropout(0.5)
        self.fc_layer2 = nn.Linear(hidden_space1, hidden_space1)
        self.fc_layer3 = nn.Linear(hidden_space1, hidden_space1)
        self.out_layer = nn.Linear(hidden_space1, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.float()))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc_layer1(x))
        x = torch.relu(self.fc_layer2(x))
        x = torch.relu(self.fc_layer3(x))
        x = self.dropout(x)
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
        self.eps_exp = 0.01
        self.exploration_noise = 0.1
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

        #noise injection
        noisy_action_probs = action_probs + torch.rand_like(action_probs)*self.exploration_noise

        distrib = Categorical(noisy_action_probs)

        action = distrib.sample()

        #e-gredy
        p = np.random.random()
        if p < self.eps_exp:
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
            critic_loss.append(F.mse_loss(value[0], R))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor = torch.stack(actor_loss).sum()
        loss_critic = torch.stack(critic_loss).sum()
        
        loss_actor.backward()
        loss_critic.backward() 

        self.optimizer_actor.step()               
        self.optimizer_critic.step()

        self.selected_actions = []
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []

    def train(n_steps):
        pass