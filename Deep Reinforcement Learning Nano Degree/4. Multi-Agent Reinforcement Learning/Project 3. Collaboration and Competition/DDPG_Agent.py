import numpy as np
import random
import copy
import os
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DDPG_Net import Net
from Replay_Buffer import ReplayBuffer

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

BUFFER_SIZE = int(1e6)      # replay buffer size
BATCH_SIZE = 128            # minibatch size
GAMMA = 0.99                # discount factor
TARGET_NETWORK_MIX = 1e-3   # for soft update of target parameters
UPDATE_ITER = 20            # learning timestep interval
LEARN_NUM = 10              # number of learning passes
OU_SIGMA = 0.2              # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15             # Ornstein-Uhlenbeck noise parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_name, state_size, action_size, hidden_dims, random_seed, num_agents=1):
        """Initialize an Agent object.

        Params
        ======
            agent_name (str): the name of agent
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dims (array): array of dimensions of each hidden layer
            random_seed (int): random seed
        """
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Net(state_size, action_size, hidden_dims).to(device)
        self.actor_target = Net(state_size, action_size, hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)
        
        # Make sure the Actor Target Network has the same weight values as the Local Network
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

        # Critic Network (w/ Target Network)
        self.critic_local = Net((state_size+action_size)*num_agents, 1, hidden_dims).to(device)
        self.critic_target = Net((state_size+action_size)*num_agents, 1, hidden_dims).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)
        
        # Make sure the Critic Target Network has the same weight values as the Local Network
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

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
        if self.agent_name not in os.listdir('./agents'):
            os.mkdir('./agents/%s' % self.agent_name)
        
        torch.save(self.actor_local.state_dict(), './agents/%s/actor_local_checkpoint.pth' % self.agent_name)
        torch.save(self.actor_target.state_dict(), './agents/%s/actor_target_checkpoint.pth' % self.agent_name)
        torch.save(self.critic_local.state_dict(), './agents/%s/critic_local_checkpoint.pth' % self.agent_name)
        torch.save(self.critic_target.state_dict(), './agents/%s/critic_target_checkpoint.pth' % self.agent_name)

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