import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Q_Net(nn.Module):
    def __init__(self, state_size, hidden_layer_size, hidden_layers, action_size, seed=1):
        super(Q_Net, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        assert(hidden_layer_size > 0)
        assert(hidden_layer_size == len(hidden_layers))
        self.input_layer = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = []
        self.hidden_layer_size = hidden_layer_size
        
        for i in range(self.hidden_layer_size-1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
        
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for i in range(self.hidden_layer_size-1):
            x = F.relu(self.hidden_layers[i](x))
        return self.output_layer(x)
    
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=1):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
        
class Agent:
    def __init__(self, state_size, hidden_layer_size, hidden_layers, action_size, epsilon, seed=1):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.seed = random.seed(seed)
        
        self.q_net_local = Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
        self.q_net_target = Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_net_local.parameters(), lr=LR)
        
        self.replay_buffer = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
    def get_policy_probs(self, action):
        greedy_action = np.argmax(action.to('cpu'))
        policy_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
        policy_probs[greedy_action] += (1-self.epsilon)
        
        return policy_probs
        
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self.learn(experiences)
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.q_net_local.eval()
        with torch.no_grad():
            action = self.q_net_local(state)
        self.q_net_local.train()
        
        return np.random.choice(self.action_size, p=self.get_policy_probs(action))
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        target = self.q_net_target(next_states).detach().max(1)[0].unsqueeze(1)
        target = rewards + (GAMMA * target * (1-dones))
        
        expected = self.q_net_local(states).gather(1, actions)
        
        loss = F.mse_loss(expected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.q_net_local, self.q_net_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)