import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)      # replay buffer size
BATCH_SIZE = 128            # minibatch size
GAMMA = 0.99                # discount factor
TARGET_NETWORK_MIX = 1e-3   # for soft update of target parameters
UPDATE_ITER = 20            # learning timestep interval
LEARN_NUM = 10              # number of learning passes
OU_SIGMA = 0.2              # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15             # Ornstein-Uhlenbeck noise parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_dims, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dims (array): array of dimensions of each hidden layer
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = DDPG_Net(state_size, action_size, hidden_dims).to(device)
        self.actor_target = DDPG_Net(state_size, action_size, hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)

        # Critic Network (w/ Target Network)
        self.critic_local = DDPG_Net(state_size+action_size, 1, hidden_dims).to(device)
        self.critic_target = DDPG_Net(state_size+action_size, 1, hidden_dims).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, t_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn at defined interval, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t_step % UPDATE_ITER == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = torch.tanh(self.actor_local(state)).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = torch.tanh(self.actor_target(next_states))
        exp_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
        # Compute Q targets for current states (y_i)
        exp_q_values = rewards + (GAMMA * exp_q_values * (1 - dones))
        # Compute critic loss
        q_values = self.critic_local(torch.cat([states, actions], dim=1))
        critic_loss = (q_values - exp_q_values).pow(2).mul(.5).sum(-1).mean()
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions = torch.tanh(self.actor_local(states))
        policy_loss = -self.critic_local(torch.cat([states, actions], dim=1)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_target, self.critic_local)
        self.soft_update(self.actor_target, self.actor_local)

        # ---------------------------- update noise ---------------------------- #
        self.noise.reset()

    def soft_update(self, target, src):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            target: PyTorch model (weights will be copied to)
            src: PyTorch model (weights will be copied from)
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - TARGET_NETWORK_MIX) + param * TARGET_NETWORK_MIX)
            
    def save_model_params(self):
        torch.save(self.actor_local.state_dict(), 'actor_local_checkpoint.pth')
        torch.save(self.actor_target.state_dict(), 'actor_target_checkpoint.pth')
        torch.save(self.critic_local.state_dict(), 'critic_local_checkpoint.pth')
        torch.save(self.critic_target.state_dict(), 'critic_target_checkpoint.pth')

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=OU_THETA, sigma=OU_SIGMA):
        """Initialize parameters and noise process.
        Params
        ======
            mu: long-running mean
            theta: the speed of mean reversion
            sigma: the volatility parameter
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DDPG_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        """Initialize a Network for DDPG.

        Params
        ======
            input_dim (int): dimension of input layer
            output_dim (int): dimension of output layer
            hidden_dims (array): array of dimensions of each hidden layer
        """
        super(DDPG_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        assert(len(hidden_dims) >= 1)
        
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])
        
        if len(self.hidden_dims) > 1:
            hidden_layers = []
            for i in range(len(hidden_dims)-1):
                hidden_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
            self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        
        output = self.output_layer(x)
        
        return output