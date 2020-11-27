### Description of the files

- README.md: Description the project and the implementation
- Navigation.ipynb: Code for the running project (Including the outputs)
- banana_dqn.py: Code for the agents (Implementation of agent, Q-Nets, Experience replays)
- banana_checkpoint_[type].pth: The local model's weights for each type of agent
  - banana_checkpoint_double_dqn.pth
  - banana_checkpoint_double_dqn_per.pth
  - banana_checkpoint_dueling_dqn.pth
  - banana_checkpoint_dueling_dqn_per.pth
- model.pt: The best model's weights

### Description of the model

#### Agent

- Based on Double DQN, return 4 different agents by parameter
  - Double DQN, Double DQN + PER, Dueling DQN, Dueling DQN + PER
- Parameters
  - `state_size`: the number of states
  - `hidden_layer_size`: the number of hidden layers for Q-Network
  - `hidden_layers`: the size of each hidden layer
  - `action_size`: the number of actions
  - `epsilon`: the probability not to select greedy action
  - `network_type`(optional, default: 1)
    * `1`: Original Q-Network
    * `2`: Dueling Q-Network
  - `replay_buffer_type`(optional, default: 1)
    * `1`: Original Replay Buffer
    * `2`: Prioritized Replay Buffer
  - `seed`(optional, default: 1)
- Inherent Variables
  - `BUFFER_SIZE`: 100000
  - `BATCH_SIZE`: 64
  - `GAMMA`: 0.99
  - `TAU`: 0.001
  - `LR` (Learning Rate): 0.0005
  - `UPDATE_EVERY`: 4

#### Run Agents

1. Initialize environment(Banana Game)

2. Initialize 4 agents (Double DQN, Double DQN + PER, Dueling DQN, Dueling DQN + PER)

   - `state_size` is 37, `action_size` is 4

   - `hidden_layer_size` is 3, `hidden_layers` is [64, 128, 64]

   - Agent structure

     ![image](https://user-images.githubusercontent.com/8471958/100484141-1a62ec00-30b0-11eb-8817-6fbaf389e1bb.png)

   - `epsilon` is 0.005

     - This `epsilon` is decreased 10% by each episode (`epsilon_decay` is 0.9)

3. Train each agent for 2000 episodes

4. Training is end to reach the max episodes or to reach to a specific score (15.0)

#### Results

1. Double DQN

   ```
   Episode 0	Average Score: 0.00
   Episode 100	Average Score: 1.47
   Episode 200	Average Score: 4.85
   Episode 300	Average Score: 9.08
   Episode 400	Average Score: 10.74
   Episode 500	Average Score: 14.29
   Episode 540	Average Score: 15.00
   Environment solved in 440 episodes!	Average Score: 15.00
   ```

2. Double DQN + PER

   ```
   Episode 0	Average Score: 0.00
   Episode 100	Average Score: 3.04
   Episode 200	Average Score: 7.52
   Episode 300	Average Score: 10.66
   Episode 400	Average Score: 14.23
   Episode 437	Average Score: 15.10
   Environment solved in 337 episodes!	Average Score: 15.10
   ```

3. Dueling DQN

   ```
   Episode 0	Average Score: 0.00
   Episode 100	Average Score: 5.32
   Episode 200	Average Score: 7.77
   Episode 300	Average Score: 9.12
   Episode 400	Average Score: 12.29
   Episode 500	Average Score: 12.84
   Episode 600	Average Score: 12.96
   Episode 700	Average Score: 12.52
   Episode 800	Average Score: 13.23
   Episode 900	Average Score: 14.82
   Episode 924	Average Score: 15.00
   Environment solved in 824 episodes!	Average Score: 15.00
   ```

4. Dueling DQN + PER

   ```
   Episode 0	Average Score: 0.00
   Episode 100	Average Score: 5.75
   Episode 200	Average Score: 8.56
   Episode 300	Average Score: 13.36
   Episode 368	Average Score: 15.02
   Environment solved in 268 episodes!	Average Score: 15.02
   ```

- Chart
  ![image](https://user-images.githubusercontent.com/8471958/100484477-737f4f80-30b1-11eb-93c6-c000924029dd.png)
  ![image](https://user-images.githubusercontent.com/8471958/100484500-8abe3d00-30b1-11eb-9727-f42ea67dbb82.png)
- **Dueling DQN + PER** > Double DQN + PER > Double DQN > Dueling DQN