import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DDPG_Agent import DDPG_Agent
from DDPG_Net import Net
from Replay_Buffer import ReplayBuffer, PrioritizedReplayBuffer

ACTOR_LR = 1e-4
CRITIC_LR = 5e-3

BUFFER_SIZE = int(1e6)      # replay buffer size
BATCH_SIZE = 128            # minibatch size
GAMMA = 0.995               # discount factor
TARGET_NETWORK_MIX = 1e-3   # for soft update of target parameters
UPDATE_ITER = 4             # learning timestep interval
LEARN_NUM = 3               # number of learning passes
OU_SIGMA = 0.2              # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15             # Ornstein-Uhlenbeck noise parameter
P_EPS=1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    def __init__(self, state_size, action_size, hidden_dims, num_agents, random_seed, per_flag=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_dims (array): array of dimensions of each hidden layer
            num_agents (int): the number of agents
            random_seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        self.agents = [DDPG_Agent('Agent%d' % (i+1), state_size, action_size, hidden_dims, random_seed, num_agents) for i in range(num_agents)]
        if per_flag:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=random_seed)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def step(self, states, actions, rewards, next_states, dones, t_step, completion):
        """Save experience in replay memory, and use sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn at defined interval, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t_step % UPDATE_ITER == 0:
            for _ in range(LEARN_NUM):
                # Update by Agent
                for agent in self.agents:
                    experiences = self.memory.sample(completion)
                    self.learn(experiences, agent)
    
    def act(self, states):
        """Returns actions for given state as per current policy."""
        return [agent.act(state) for agent, state in zip(self.agents, states)]        
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
    
    def learn(self, experiences, agent):
        
        states, actions, rewards, next_states, dones, sample_indices, weights = experiences
        
        indices = [torch.tensor(np.arange(i, BATCH_SIZE*self.num_agents, self.num_agents)).to(device) for i in range(self.num_agents)]
        
        agent_states = [states.index_select(0, ind) for ind in indices]
        agent_actions = [actions.index_select(0, ind) for ind in indices]
        agent_next_states = [next_states.index_select(0, ind) for ind in indices]
        
        all_states=torch.cat(agent_states, dim=1).to(device)
        all_actions=torch.cat(agent_actions, dim=1).to(device)
        all_next_states=torch.cat(agent_next_states, dim=1).to(device)
             
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        all_next_actions = torch.cat([torch.tanh(agent.actor_target(next_states)) for next_states in agent_next_states], dim=1).to(device)
        exp_q_values = agent.critic_target(torch.cat([all_next_states, all_next_actions], dim=1))
        # Compute Q targets for current states (y_i)
        exp_q_values = rewards + (GAMMA * exp_q_values * (1 - dones))
        
        # Compute critic loss
        q_values = agent.critic_local(torch.cat([all_states, all_actions], dim=1))
        critic_loss = (q_values - exp_q_values).pow(2).mul(.5).sum(-1).mean()
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions = torch.cat([torch.tanh(agent.actor_local(states)) for states in agent_states], dim=1).to(device)
        policy_loss = -agent.critic_local(torch.cat([all_states, all_actions], dim=1)).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        policy_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_target, agent.critic_local)
        agent.soft_update(agent.actor_target, agent.actor_local)

        # ---------------------------- update noise ---------------------------- #
        agent.noise.reset()
        
        # ---------------------------- update priority ------------------------- #
        self.memory.update_priorities(sample_indices, (q_values - exp_q_values).abs().detach().cpu().numpy() + P_EPS)
    
    def save_model_params(self):
        for agent in self.agents:
            agent.save_model_params()