### Description of the files

- `README.md`: Description the project and the implementation
- `Tennis.ipynb`: Code for the running project (Including the outputs)
- `DDPG_Agent.py`: Code for the agents (Implementation of DDPG agent)
- `DDPG_Net.py`: Code for the neural network (Actor and Critics networks)
- `Replay_Buffer.py`: Code for the Replay Buffer and Prioritized Experinces Replay
- `MADDPG_Agent.py`: Code for the Multi-Agent DDPG agent
- `agent` folder: The model's weights for each type of agent
  - actor_local_checkpoint.pth
  - actor_target_checkpoint.pth
  - critic_local_checkpoint.pth
  - critic_local_checkpoint.pth

### Description of the model

#### Agent

- **Algorithm**: Multi-Agent Deep Deterministic Policy Gradients(MADDPG) algorithm
  
  - Each agent is trained by DDPG algorithm
    - Actor and Critic Network Structure
      ![image](https://user-images.githubusercontent.com/8471958/102154011-5d9db880-3e2d-11eb-8c40-bbc8a7cd2f11.png)
      - Actor's input layer dimension is state size(24)
      - Critic's input layer dimension is (state size(24) + actor size(2)) x the number of agents(2)
    - DDPG Algorithm Structure
      ![image](https://user-images.githubusercontent.com/8471958/101946027-6e3afe00-3ba3-11eb-8e76-d246a8e2bb39.png)
    
  - Each step contains interactions of each agent
  
    ![image](https://user-images.githubusercontent.com/8471958/102171552-a5333d00-3e4b-11eb-8ff5-528b6571e447.png)
  
    - Each agent returns their action by state
    - Environment return next states and rewards by actions
    - Memory saves states, actions, rewards, next_states and dones
    - In the each iteration(determined by inherent variables), each agent update their weights using memories
- **Parameters**
  
  - `state_size`: The number of states
  - `action-size`: The number of actions
  - `hidden_dims`: The dimension of each hidden layers
  - `num_agents`: The number of agents
  - `random_seed`: Random seed
- **Inherent Variables**(with value)
  
  - `ACTOR_LR`(0.0001): The learning rate for actor
  - `CRITIC_LR`(0.005): The learning rate for critic
  - `BUFFER_SIZE`(1,000,000): Replay Buffer Size 
  - `BATCH_SIZE`(128): Mini-batch Size
  - `GAMMA`(0.995): Discount Factor
  - `TARGET_NETWORK_MIX`(0.001): the ratio of target parameter for soft update
  - `UPDATE_ITER`(4): Learning Interval
  - `LEARN_NUM`(3): The number of learning passes
  - `OU_SIGMA`(0.2): Ornstein-Uhlenbeck noise parameter
  - `OU_TEHTA`(0.15): Ornstein-Uhlenbeck noise parameter

#### Run Agents

1. Initialize environment
2. Initialize agents
   - `state_size` is 24, `action_size` is 2
   - `hidden_dims` is [400, 300]
3. Train each agent for 10,000 episodes
   - The maximum timestamp of each episode is 1,000
   - The early stopping condition is that an average score is over 0.5

#### Results

```python
Episode 100 (32 sec) 	 -- Average Score: 0.00	Max Score: 0.00
Episode 200 (35 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 300 (34 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 400 (36 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 500 (34 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 600 (35 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 700 (34 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 800 (35 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 900 (37 sec) 	 -- Average Score: 0.01	Max Score: 0.10
Episode 1000 (43 sec) 	 -- Average Score: 0.02	Max Score: 0.10
Episode 1100 (43 sec) 	 -- Average Score: 0.02	Max Score: 0.10
Episode 1200 (38 sec) 	 -- Average Score: 0.01	Max Score: 0.10
Episode 1300 (38 sec) 	 -- Average Score: 0.00	Max Score: 0.10
Episode 1400 (41 sec) 	 -- Average Score: 0.01	Max Score: 0.10
Episode 1500 (52 sec) 	 -- Average Score: 0.04	Max Score: 0.10
Episode 1600 (47 sec) 	 -- Average Score: 0.02	Max Score: 0.10
Episode 1700 (56 sec) 	 -- Average Score: 0.04	Max Score: 0.20
Episode 1800 (51 sec) 	 -- Average Score: 0.04	Max Score: 0.20
Episode 1900 (49 sec) 	 -- Average Score: 0.04	Max Score: 0.20
Episode 2000 (56 sec) 	 -- Average Score: 0.05	Max Score: 0.20
Episode 2100 (68 sec) 	 -- Average Score: 0.09	Max Score: 0.40
Episode 2200 (60 sec) 	 -- Average Score: 0.07	Max Score: 0.40
Episode 2300 (71 sec) 	 -- Average Score: 0.09	Max Score: 0.40
Episode 2400 (60 sec) 	 -- Average Score: 0.07	Max Score: 0.40
Episode 2500 (45 sec) 	 -- Average Score: 0.03	Max Score: 0.40
Episode 2600 (51 sec) 	 -- Average Score: 0.05	Max Score: 0.40
Episode 2700 (75 sec) 	 -- Average Score: 0.09	Max Score: 0.40
Episode 2800 (97 sec) 	 -- Average Score: 0.11	Max Score: 0.40
Episode 2900 (172 sec) 	 -- Average Score: 0.20	Max Score: 0.90
Episode 3000 (200 sec) 	 -- Average Score: 0.23	Max Score: 1.50
Episode 3100 (278 sec) 	 -- Average Score: 0.32	Max Score: 1.80
Episode 3152 (266 sec) 	 -- Average Score: 0.51	Max Score: 2.60
```

![image](https://user-images.githubusercontent.com/8471958/102172931-c21d3f80-3e4e-11eb-8b67-7e2870a167eb.png)

#### Future ideas for improving the agent's performance

1. Using the Prioritized Experince Replay(PER) instead of Replay Buffer
   - In the previous project(Navigation), there were always better results when use PER instead of Replay Buffer


