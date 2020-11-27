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

class Dueling_Q_Net(nn.Module):
    def __init__(self, state_size, hidden_layer_size, hidden_layers, action_size, seed=1):
        super(Dueling_Q_Net, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        assert(hidden_layer_size > 0)
        assert(hidden_layer_size == len(hidden_layers))
        self.input_layer = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = []
        self.hidden_layer_size = hidden_layer_size
        
        for i in range(self.hidden_layer_size-1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            
        self.hidden_layers = nn.Sequential(*self.hidden_layers)

        self.state_output_layer = nn.Linear(hidden_layers[-1], 1)
        self.action_output_layer = nn.Linear(hidden_layers[-1], action_size)
        
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for i in range(self.hidden_layer_size-1):
            x = F.relu(self.hidden_layers[i](x))
        state_out = self.state_output_layer(x)
        action_out = self.action_output_layer(x)

        output = state_out + (action_out - ((1 / self.action_size) * action_out))
        return output
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self, completion=None):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6, prob_beta=0.5, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        self.pos = 0
        
    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if len(self.memory) > 0 else 1
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(max_priority)
        
    def sample(self, completion):
        beta = self.prob_beta + (1 - self.prob_beta) * completion
            
        prob_a = np.array(self.priorities) ** self.prob_alpha
        p_i = prob_a / prob_a.sum()
        
        sampled_indices = np.random.choice(len(self.memory), self.batch_size, p=p_i)
        experiences = [self.memory[idx] for idx in sampled_indices]
        
        N = len(self.memory)
        weights = (N * p_i[sampled_indices]) ** (-1 * beta)
        weights = weights / weights.max()
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, sampled_indices, weights)

    def update_priorities(self, batch_indicies, batch_priorities):
        for idx, priority in zip(batch_indicies, batch_priorities):
            self.priorities[idx] = priority[0]
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
class Agent:
    def __init__(self, state_size, hidden_layer_size, hidden_layers, action_size, epsilon, network_type=1, replay_buffer_type=1, seed=1):
        """
        Params
        ======
            state_size (int): dimension of each state
            hidden_layer_size (int): the number of hidden layers
            hidden_layers (array-like): layer size of each hidden layers
            action_size (int): dimension of each action
            epsilon (float)
            network_type (int)
              1: Original Q-Network
              2: Dueling Q-Network
            replay_buffer_type (int)
              1: Original Replay Buffer
              2: Prioritized Replay Buffer
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.seed = random.seed(seed)

        self.network_type = network_type
        if network_type == 1:
            self.qnetwork_local = Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
            self.qnetwork_target = Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
        elif network_type == 2:
            self.qnetwork_local = Dueling_Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
            self.qnetwork_target = Dueling_Q_Net(state_size, hidden_layer_size, hidden_layers, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.replay_buffer_type = replay_buffer_type
        if replay_buffer_type == 1:
            self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        elif replay_buffer_type == 2:
            self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
    def get_policy_probs(self, action):
        greedy_action = np.argmax(action.to('cpu'))
        policy_probs = np.ones(self.action_size) * (self.epsilon / self.action_size)
        policy_probs[greedy_action] += (1-self.epsilon)
        
        return policy_probs
        
    def step(self, state, action, reward, next_state, done, completion):
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample(completion)
                self.learn(experiences, GAMMA)
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        return np.random.choice(self.action_size, p=self.get_policy_probs(action))
    
    def learn(self, experiences, p_eps=1e-5):

        if self.replay_buffer_type == 1:
            states, actions, rewards, next_states, dones = experiences
        elif self.replay_buffer_type == 2:
            states, actions, rewards, next_states, dones, sample_indices, weights = experiences

        # Local Network's Greedy Actions (Maximum Q-value)
        greedy_action = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)

        target = self.qnetwork_target(next_states).detach().gather(1, greedy_action)
        target = rewards + (GAMMA * target * (1-dones))

        expected = self.qnetwork_local(states).gather(1, actions)

        if self.replay_buffer_type == 1:
            loss = F.mse_loss(expected, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif self.replay_buffer_type == 2:
            td_err = target - expected
            loss = ((td_err ** 2) * weights).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.replay_buffer.update_priorities(sample_indices, td_err.abs().detach().cpu().numpy() + p_eps)
            self.optimizer.step()       
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1 - TAU) * target_param.data)