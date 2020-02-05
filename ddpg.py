import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


# Можно написать и на питоновском листе, но можно и на деке.
class replay_buffer:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
        
        
# Если делать меньше слои, например, 64 юнита, то модель гораздо хуже учится.
class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self, state):
        res = F.relu(self.fc1(state))
        res = F.relu(self.fc2(res))
        res = torch.tanh(self.fc3(res))
        return res
        
        
class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        res = torch.cat([state, action], 1)
        res = F.relu(self.fc1(res))
        res = F.relu(self.fc2(res))
        res = self.fc3(res)
        return res

        
# Сделал маленький буффер и достаточно большой размер батча, чтобы не пропустить закатывание на верх.
# Да, модель может забывать предыдущий опыт, но это же тележка. ¯\_(ツ)_/¯
class ddpg:
    def __init__(self, state_dim=2, action_dim=1, gamma=0.99, tau=1e-3, buffer_maxlen=5000, batch_size=640, critic_lr=1e-4, actor_lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = gamma
        
        self.buffer_maxlen = buffer_maxlen
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer(self.buffer_maxlen, self.batch_size) 
        
        self.tau = tau
        self.critic = critic(self.state_dim, self.action_dim).to(self.device)
        self.actor = actor(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr) 
        
        self.critic_target = critic(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = actor(self.state_dim, self.action_dim).to(self.device)
        self.soft_update(1)
            
    
    def update(self, transition):
        self.replay_buffer.push(transition)
        
        if len(self.replay_buffer) > self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
       
            local_pred = self.critic.forward(states, actions)
            next_actions = self.actor_target.forward(next_states)
            target_pred = rewards + self.gamma * self.critic_target.forward(next_states, next_actions.detach())
            
            critic_loss = F.mse_loss(local_pred, target_pred.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward() 
            self.critic_optimizer.step()

            # chain rule magic + switching min to max by *(-1)
            actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.tau)
            
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()
        return action
            
    def soft_update(self, cur_tau):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * cur_tau + target_param.data * (1.0 - cur_tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * cur_tau + target_param.data * (1.0 - cur_tau))
            
    def save(self):
        torch.save(self.critic.state_dict(), 'critic.pkl')
        torch.save(self.actor.state_dict(), 'actor.pkl')