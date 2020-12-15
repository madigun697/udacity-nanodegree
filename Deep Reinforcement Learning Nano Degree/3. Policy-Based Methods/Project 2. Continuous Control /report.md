### Description of the files

- README.md: Description the project and the implementation
- Continuous_Control.ipynb: Code for the running project (Including the outputs)
- DDPG.py: Code for the agents (Implementation of DDPG agent)
- [model]__checkpoint.pth: The model's weights for each type of agent
  - actor_local_checkpoint.pth
  - actor_target_checkpoint.pth
  - critic_local_checkpoint.pth
  - critic_local_checkpoint.pth



### Description of the model

#### Agent

- **Algorithm**: Deep Deterministic Policy Gradients(DDPG) algorithm
  - DDPG algorithm apply the advantages of DQN algorithm into the Actor-Critifc approach
    - Replay Buffer: Reduce the correlation between samples
    - Target Q Network: Be stable network during update
  - Actor and Critic Network Structure
    ![image](https://user-images.githubusercontent.com/8471958/102170550-7d42da00-3e49-11eb-929d-a6ba8688088f.png)
    - Actor's input layer dimension is state size(33)
    - Critic's input layer dimension is sum of state size(33) and action size(4)
  - DDPG Algorithm Structure
    ![image](https://user-images.githubusercontent.com/8471958/101946027-6e3afe00-3ba3-11eb-8e76-d246a8e2bb39.png)
- **Parameters**
  - `state_size`: The number of states
  - `action-size`: The number of actions
  - `hidden_dims`: The dimension of each hidden layers
  - `random_seed`: Random seed
- **Inherent Variables**(with value)
  - `BUFFER_SIZE`(1,000,000): Replay Buffer Size 
  - `BATCH_SIZE`(128): Mini-batch Size
  - `GAMMA`(0.99): Discount Factor
  - `TARGET_NETWORK_MIX`(0.001): the ratio of target parameter for soft update
  - `UPDATE_ITER`(20): Learning Interval
  - `LEARN_NUM`(10): The number of learning passes
  - `OU_SIGMA`(0.2): Ornstein-Uhlenbeck noise parameter
  - `OU_TEHTA`(0.15): Ornstein-Uhlenbeck noise parameter

#### Run Agents

1. Initialize environment
2. Initialize agents
   - `state_size` is 33, `action_size` is 4
   - `hidden_dims` is [400, 300]
3. Train each agent for 200 episodes
   - The maximum timestamp of each episode is 1000

#### Results

```python
Episode 1 (91 sec)  -- 	Min: 0.0	Max: 1.9	Mean: 0.7	Mov. Avg: 0.7
Episode 2 (94 sec)  -- 	Min: 0.0	Max: 3.9	Mean: 1.7	Mov. Avg: 1.2
Episode 3 (95 sec)  -- 	Min: 1.9	Max: 6.1	Mean: 3.7	Mov. Avg: 2.0
Episode 4 (96 sec)  -- 	Min: 2.7	Max: 6.8	Mean: 5.2	Mov. Avg: 2.8
Episode 5 (96 sec)  -- 	Min: 2.2	Max: 9.5	Mean: 4.6	Mov. Avg: 3.2
Episode 6 (97 sec)  -- 	Min: 3.2	Max: 7.1	Mean: 5.1	Mov. Avg: 3.5
Episode 7 (97 sec)  -- 	Min: 3.2	Max: 9.9	Mean: 6.3	Mov. Avg: 3.9
Episode 8 (97 sec)  -- 	Min: 4.1	Max: 9.2	Mean: 6.7	Mov. Avg: 4.3
Episode 9 (98 sec)  -- 	Min: 3.8	Max: 11.0	Mean: 7.4	Mov. Avg: 4.6
Episode 10 (98 sec)  -- 	Min: 3.2	Max: 12.0	Mean: 7.8	Mov. Avg: 4.9
Episode 11 (99 sec)  -- 	Min: 3.8	Max: 15.0	Mean: 7.7	Mov. Avg: 5.2
Episode 12 (99 sec)  -- 	Min: 5.6	Max: 16.6	Mean: 9.8	Mov. Avg: 5.6
Episode 13 (101 sec)  -- 	Min: 6.4	Max: 20.6	Mean: 12.1	Mov. Avg: 6.1
Episode 14 (101 sec)  -- 	Min: 8.9	Max: 17.6	Mean: 12.7	Mov. Avg: 6.5
Episode 15 (102 sec)  -- 	Min: 6.4	Max: 21.7	Mean: 12.7	Mov. Avg: 7.0
Episode 16 (103 sec)  -- 	Min: 3.1	Max: 25.1	Mean: 15.4	Mov. Avg: 7.5
Episode 17 (103 sec)  -- 	Min: 7.4	Max: 19.1	Mean: 13.8	Mov. Avg: 7.8
Episode 18 (104 sec)  -- 	Min: 5.7	Max: 22.5	Mean: 14.7	Mov. Avg: 8.2
Episode 19 (105 sec)  -- 	Min: 9.0	Max: 24.0	Mean: 18.5	Mov. Avg: 8.8
Episode 20 (106 sec)  -- 	Min: 12.1	Max: 25.4	Mean: 19.3	Mov. Avg: 9.3
Episode 21 (108 sec)  -- 	Min: 15.7	Max: 25.7	Mean: 20.0	Mov. Avg: 9.8
Episode 22 (107 sec)  -- 	Min: 13.8	Max: 26.5	Mean: 22.2	Mov. Avg: 10.4
Episode 23 (108 sec)  -- 	Min: 17.3	Max: 34.0	Mean: 24.3	Mov. Avg: 11.0
Episode 24 (109 sec)  -- 	Min: 15.4	Max: 31.1	Mean: 25.6	Mov. Avg: 11.6
Episode 25 (110 sec)  -- 	Min: 18.1	Max: 33.4	Mean: 26.8	Mov. Avg: 12.2
Episode 26 (112 sec)  -- 	Min: 17.5	Max: 38.5	Mean: 30.3	Mov. Avg: 12.9
Episode 27 (111 sec)  -- 	Min: 20.2	Max: 35.8	Mean: 31.0	Mov. Avg: 13.6
Episode 28 (113 sec)  -- 	Min: 22.7	Max: 36.7	Mean: 28.2	Mov. Avg: 14.1
Episode 29 (114 sec)  -- 	Min: 23.9	Max: 36.8	Mean: 32.3	Mov. Avg: 14.7
Episode 30 (116 sec)  -- 	Min: 23.0	Max: 37.9	Mean: 31.2	Mov. Avg: 15.3
Episode 31 (117 sec)  -- 	Min: 23.1	Max: 37.3	Mean: 31.6	Mov. Avg: 15.8
Episode 32 (118 sec)  -- 	Min: 16.8	Max: 39.3	Mean: 31.8	Mov. Avg: 16.3
Episode 33 (119 sec)  -- 	Min: 23.0	Max: 39.5	Mean: 31.4	Mov. Avg: 16.8
Episode 34 (119 sec)  -- 	Min: 20.3	Max: 38.4	Mean: 32.2	Mov. Avg: 17.2
Episode 35 (120 sec)  -- 	Min: 20.7	Max: 39.6	Mean: 33.1	Mov. Avg: 17.7
Episode 36 (122 sec)  -- 	Min: 21.7	Max: 39.4	Mean: 33.1	Mov. Avg: 18.1
Episode 37 (123 sec)  -- 	Min: 26.4	Max: 39.6	Mean: 35.7	Mov. Avg: 18.6
Episode 38 (124 sec)  -- 	Min: 29.6	Max: 39.6	Mean: 35.9	Mov. Avg: 19.0
Episode 39 (126 sec)  -- 	Min: 22.4	Max: 37.9	Mean: 34.0	Mov. Avg: 19.4
Episode 40 (126 sec)  -- 	Min: 18.9	Max: 39.4	Mean: 33.1	Mov. Avg: 19.7
Episode 41 (129 sec)  -- 	Min: 24.1	Max: 39.3	Mean: 33.3	Mov. Avg: 20.1
Episode 42 (129 sec)  -- 	Min: 26.9	Max: 39.4	Mean: 36.3	Mov. Avg: 20.5
Episode 43 (132 sec)  -- 	Min: 26.7	Max: 39.5	Mean: 36.4	Mov. Avg: 20.8
Episode 44 (132 sec)  -- 	Min: 24.7	Max: 38.0	Mean: 34.2	Mov. Avg: 21.1
Episode 45 (133 sec)  -- 	Min: 27.9	Max: 39.5	Mean: 34.9	Mov. Avg: 21.4
Episode 46 (135 sec)  -- 	Min: 30.4	Max: 38.0	Mean: 35.7	Mov. Avg: 21.8
Episode 47 (136 sec)  -- 	Min: 29.9	Max: 39.3	Mean: 35.1	Mov. Avg: 22.0
Episode 48 (139 sec)  -- 	Min: 26.9	Max: 38.4	Mean: 33.9	Mov. Avg: 22.3
Episode 49 (140 sec)  -- 	Min: 27.2	Max: 39.5	Mean: 34.8	Mov. Avg: 22.5
Episode 50 (140 sec)  -- 	Min: 27.7	Max: 38.2	Mean: 36.1	Mov. Avg: 22.8
Episode 51 (142 sec)  -- 	Min: 31.7	Max: 39.2	Mean: 36.1	Mov. Avg: 23.1
Episode 52 (142 sec)  -- 	Min: 32.4	Max: 38.6	Mean: 36.0	Mov. Avg: 23.3
Episode 53 (142 sec)  -- 	Min: 33.4	Max: 39.6	Mean: 36.9	Mov. Avg: 23.6
Episode 54 (141 sec)  -- 	Min: 32.4	Max: 39.6	Mean: 37.0	Mov. Avg: 23.8
Episode 55 (143 sec)  -- 	Min: 32.8	Max: 39.6	Mean: 36.8	Mov. Avg: 24.1
Episode 56 (141 sec)  -- 	Min: 30.5	Max: 39.6	Mean: 36.1	Mov. Avg: 24.3
Episode 57 (140 sec)  -- 	Min: 33.8	Max: 39.2	Mean: 37.5	Mov. Avg: 24.5
Episode 58 (142 sec)  -- 	Min: 31.8	Max: 39.0	Mean: 36.5	Mov. Avg: 24.7
Episode 59 (141 sec)  -- 	Min: 35.9	Max: 39.5	Mean: 37.7	Mov. Avg: 24.9
Episode 60 (144 sec)  -- 	Min: 36.3	Max: 39.4	Mean: 37.8	Mov. Avg: 25.1
Episode 61 (143 sec)  -- 	Min: 21.9	Max: 39.5	Mean: 36.6	Mov. Avg: 25.3
Episode 62 (142 sec)  -- 	Min: 34.8	Max: 39.5	Mean: 37.8	Mov. Avg: 25.5
Episode 63 (142 sec)  -- 	Min: 22.0	Max: 39.5	Mean: 36.8	Mov. Avg: 25.7
Episode 64 (141 sec)  -- 	Min: 36.5	Max: 39.6	Mean: 38.4	Mov. Avg: 25.9
Episode 65 (140 sec)  -- 	Min: 27.4	Max: 39.5	Mean: 36.8	Mov. Avg: 26.1
Episode 66 (141 sec)  -- 	Min: 34.2	Max: 39.6	Mean: 37.2	Mov. Avg: 26.2
Episode 67 (143 sec)  -- 	Min: 34.9	Max: 39.4	Mean: 37.8	Mov. Avg: 26.4
Episode 68 (142 sec)  -- 	Min: 28.4	Max: 39.4	Mean: 36.1	Mov. Avg: 26.6
Episode 69 (142 sec)  -- 	Min: 28.6	Max: 39.5	Mean: 36.9	Mov. Avg: 26.7
Episode 70 (141 sec)  -- 	Min: 31.9	Max: 39.5	Mean: 36.8	Mov. Avg: 26.9
Episode 71 (141 sec)  -- 	Min: 30.2	Max: 39.7	Mean: 36.9	Mov. Avg: 27.0
Episode 72 (142 sec)  -- 	Min: 31.9	Max: 39.2	Mean: 36.7	Mov. Avg: 27.1
Episode 73 (140 sec)  -- 	Min: 35.4	Max: 39.4	Mean: 37.6	Mov. Avg: 27.3
Episode 74 (140 sec)  -- 	Min: 30.7	Max: 39.6	Mean: 36.8	Mov. Avg: 27.4
Episode 75 (140 sec)  -- 	Min: 33.6	Max: 38.9	Mean: 37.4	Mov. Avg: 27.5
Episode 76 (139 sec)  -- 	Min: 28.6	Max: 37.9	Mean: 35.1	Mov. Avg: 27.6
Episode 77 (142 sec)  -- 	Min: 29.9	Max: 39.4	Mean: 34.7	Mov. Avg: 27.7
Episode 78 (143 sec)  -- 	Min: 30.1	Max: 39.3	Mean: 36.0	Mov. Avg: 27.8
Episode 79 (141 sec)  -- 	Min: 28.8	Max: 39.6	Mean: 36.7	Mov. Avg: 27.9
Episode 80 (140 sec)  -- 	Min: 33.3	Max: 39.0	Mean: 36.8	Mov. Avg: 28.1
Episode 81 (142 sec)  -- 	Min: 22.3	Max: 38.6	Mean: 33.6	Mov. Avg: 28.1
Episode 82 (141 sec)  -- 	Min: 30.4	Max: 37.5	Mean: 34.8	Mov. Avg: 28.2
Episode 83 (142 sec)  -- 	Min: 34.7	Max: 38.3	Mean: 36.4	Mov. Avg: 28.3
Episode 84 (144 sec)  -- 	Min: 29.0	Max: 38.0	Mean: 33.5	Mov. Avg: 28.4
Episode 85 (143 sec)  -- 	Min: 25.5	Max: 38.8	Mean: 33.5	Mov. Avg: 28.4
Episode 86 (142 sec)  -- 	Min: 25.0	Max: 37.0	Mean: 33.7	Mov. Avg: 28.5
Episode 87 (142 sec)  -- 	Min: 23.1	Max: 35.7	Mean: 32.1	Mov. Avg: 28.5
Episode 88 (141 sec)  -- 	Min: 26.2	Max: 38.6	Mean: 33.2	Mov. Avg: 28.6
Episode 89 (143 sec)  -- 	Min: 28.4	Max: 37.4	Mean: 33.0	Mov. Avg: 28.6
Episode 90 (143 sec)  -- 	Min: 28.7	Max: 37.9	Mean: 33.8	Mov. Avg: 28.7
Episode 91 (143 sec)  -- 	Min: 28.3	Max: 38.0	Mean: 34.2	Mov. Avg: 28.8
Episode 92 (143 sec)  -- 	Min: 21.6	Max: 36.4	Mean: 29.6	Mov. Avg: 28.8
Episode 93 (143 sec)  -- 	Min: 28.1	Max: 39.5	Mean: 33.1	Mov. Avg: 28.8
Episode 94 (143 sec)  -- 	Min: 29.6	Max: 38.6	Mean: 34.0	Mov. Avg: 28.9
Episode 95 (142 sec)  -- 	Min: 29.9	Max: 37.6	Mean: 34.4	Mov. Avg: 28.9
Episode 96 (143 sec)  -- 	Min: 32.1	Max: 37.9	Mean: 35.0	Mov. Avg: 29.0
Episode 97 (143 sec)  -- 	Min: 31.8	Max: 37.6	Mean: 35.3	Mov. Avg: 29.1
Episode 98 (144 sec)  -- 	Min: 31.5	Max: 38.4	Mean: 35.5	Mov. Avg: 29.1
Episode 99 (143 sec)  -- 	Min: 31.2	Max: 38.4	Mean: 35.7	Mov. Avg: 29.2
Episode 100 (142 sec)  -- 	Min: 33.7	Max: 38.1	Mean: 36.1	Mov. Avg: 29.3
Episode 101 (143 sec)  -- 	Min: 30.8	Max: 38.4	Mean: 34.8	Mov. Avg: 29.6
Episode 102 (144 sec)  -- 	Min: 29.0	Max: 39.6	Mean: 34.6	Mov. Avg: 29.9
Episode 103 (143 sec)  -- 	Min: 30.4	Max: 38.7	Mean: 35.9	Mov. Avg: 30.2
Episode 104 (143 sec)  -- 	Min: 33.0	Max: 38.3	Mean: 35.3	Mov. Avg: 30.5
Episode 105 (144 sec)  -- 	Min: 29.7	Max: 38.6	Mean: 35.3	Mov. Avg: 30.9
Episode 106 (143 sec)  -- 	Min: 30.1	Max: 38.8	Mean: 36.0	Mov. Avg: 31.2
Episode 107 (144 sec)  -- 	Min: 30.5	Max: 38.3	Mean: 36.3	Mov. Avg: 31.5
Episode 108 (143 sec)  -- 	Min: 29.3	Max: 36.5	Mean: 33.6	Mov. Avg: 31.7
Episode 109 (142 sec)  -- 	Min: 28.6	Max: 37.8	Mean: 35.1	Mov. Avg: 32.0
Episode 110 (143 sec)  -- 	Min: 30.9	Max: 38.9	Mean: 35.6	Mov. Avg: 32.3
Episode 111 (142 sec)  -- 	Min: 31.2	Max: 36.8	Mean: 34.5	Mov. Avg: 32.6
Episode 112 (143 sec)  -- 	Min: 29.5	Max: 37.7	Mean: 34.9	Mov. Avg: 32.8
Episode 113 (143 sec)  -- 	Min: 32.9	Max: 39.0	Mean: 37.2	Mov. Avg: 33.1
Episode 114 (142 sec)  -- 	Min: 33.1	Max: 36.9	Mean: 35.0	Mov. Avg: 33.3
Episode 115 (142 sec)  -- 	Min: 31.0	Max: 38.4	Mean: 34.6	Mov. Avg: 33.5
Episode 116 (143 sec)  -- 	Min: 27.5	Max: 37.9	Mean: 34.3	Mov. Avg: 33.7
Episode 117 (142 sec)  -- 	Min: 29.8	Max: 36.8	Mean: 33.1	Mov. Avg: 33.9
Episode 118 (144 sec)  -- 	Min: 25.5	Max: 37.5	Mean: 33.3	Mov. Avg: 34.1
Episode 119 (144 sec)  -- 	Min: 22.1	Max: 38.7	Mean: 32.3	Mov. Avg: 34.2
Episode 120 (144 sec)  -- 	Min: 29.3	Max: 38.0	Mean: 33.6	Mov. Avg: 34.3
Episode 121 (145 sec)  -- 	Min: 24.0	Max: 37.9	Mean: 32.7	Mov. Avg: 34.5
Episode 122 (144 sec)  -- 	Min: 33.4	Max: 38.9	Mean: 36.4	Mov. Avg: 34.6
Episode 123 (144 sec)  -- 	Min: 34.6	Max: 37.8	Mean: 36.5	Mov. Avg: 34.7
Episode 124 (143 sec)  -- 	Min: 26.3	Max: 38.7	Mean: 35.4	Mov. Avg: 34.8
Episode 125 (143 sec)  -- 	Min: 33.7	Max: 38.7	Mean: 36.6	Mov. Avg: 34.9
Episode 126 (143 sec)  -- 	Min: 29.0	Max: 38.9	Mean: 33.3	Mov. Avg: 35.0
Episode 127 (143 sec)  -- 	Min: 30.6	Max: 38.9	Mean: 34.5	Mov. Avg: 35.0
Episode 128 (143 sec)  -- 	Min: 31.4	Max: 39.0	Mean: 35.4	Mov. Avg: 35.1
Episode 129 (143 sec)  -- 	Min: 32.1	Max: 38.6	Mean: 35.8	Mov. Avg: 35.1
Episode 130 (144 sec)  -- 	Min: 28.2	Max: 38.5	Mean: 35.0	Mov. Avg: 35.1
Episode 131 (145 sec)  -- 	Min: 32.4	Max: 38.8	Mean: 35.8	Mov. Avg: 35.2
Episode 132 (144 sec)  -- 	Min: 32.7	Max: 39.5	Mean: 36.2	Mov. Avg: 35.2
Episode 133 (144 sec)  -- 	Min: 31.7	Max: 38.9	Mean: 36.0	Mov. Avg: 35.3
Episode 134 (144 sec)  -- 	Min: 25.9	Max: 38.5	Mean: 35.5	Mov. Avg: 35.3
Episode 135 (144 sec)  -- 	Min: 31.8	Max: 38.1	Mean: 35.0	Mov. Avg: 35.3
Episode 136 (144 sec)  -- 	Min: 27.7	Max: 39.0	Mean: 33.5	Mov. Avg: 35.3
Episode 137 (144 sec)  -- 	Min: 26.7	Max: 37.7	Mean: 34.3	Mov. Avg: 35.3
Episode 138 (143 sec)  -- 	Min: 30.6	Max: 38.6	Mean: 34.7	Mov. Avg: 35.3
Episode 139 (144 sec)  -- 	Min: 27.3	Max: 39.2	Mean: 34.1	Mov. Avg: 35.3
Episode 140 (144 sec)  -- 	Min: 26.0	Max: 38.2	Mean: 34.3	Mov. Avg: 35.3
Episode 141 (144 sec)  -- 	Min: 27.2	Max: 35.6	Mean: 32.8	Mov. Avg: 35.3
Episode 142 (141 sec)  -- 	Min: 32.5	Max: 38.4	Mean: 35.8	Mov. Avg: 35.3
Episode 143 (145 sec)  -- 	Min: 31.3	Max: 38.5	Mean: 35.7	Mov. Avg: 35.3
Episode 144 (143 sec)  -- 	Min: 32.9	Max: 38.2	Mean: 35.9	Mov. Avg: 35.3
Episode 145 (144 sec)  -- 	Min: 29.6	Max: 38.6	Mean: 34.2	Mov. Avg: 35.3
Episode 146 (145 sec)  -- 	Min: 23.8	Max: 38.8	Mean: 34.5	Mov. Avg: 35.3
Episode 147 (145 sec)  -- 	Min: 23.0	Max: 36.7	Mean: 32.4	Mov. Avg: 35.3
Episode 148 (144 sec)  -- 	Min: 24.8	Max: 37.6	Mean: 32.1	Mov. Avg: 35.3
Episode 149 (145 sec)  -- 	Min: 27.9	Max: 38.3	Mean: 34.5	Mov. Avg: 35.2
Episode 150 (144 sec)  -- 	Min: 25.8	Max: 37.9	Mean: 35.1	Mov. Avg: 35.2
Episode 151 (144 sec)  -- 	Min: 27.8	Max: 38.9	Mean: 33.7	Mov. Avg: 35.2
Episode 152 (146 sec)  -- 	Min: 22.0	Max: 32.6	Mean: 27.5	Mov. Avg: 35.1
Episode 153 (145 sec)  -- 	Min: 20.2	Max: 34.3	Mean: 25.5	Mov. Avg: 35.0
Episode 154 (144 sec)  -- 	Min: 25.0	Max: 35.6	Mean: 32.1	Mov. Avg: 35.0
Episode 155 (145 sec)  -- 	Min: 27.7	Max: 37.9	Mean: 34.4	Mov. Avg: 34.9
Episode 156 (141 sec)  -- 	Min: 31.4	Max: 38.4	Mean: 35.0	Mov. Avg: 34.9
Episode 157 (144 sec)  -- 	Min: 32.5	Max: 37.4	Mean: 35.5	Mov. Avg: 34.9
Episode 158 (144 sec)  -- 	Min: 31.8	Max: 38.9	Mean: 36.5	Mov. Avg: 34.9
Episode 159 (143 sec)  -- 	Min: 30.1	Max: 38.2	Mean: 35.4	Mov. Avg: 34.9
Episode 160 (144 sec)  -- 	Min: 33.8	Max: 38.3	Mean: 36.1	Mov. Avg: 34.9
Episode 161 (142 sec)  -- 	Min: 32.5	Max: 38.6	Mean: 36.4	Mov. Avg: 34.9
Episode 162 (143 sec)  -- 	Min: 31.4	Max: 38.6	Mean: 36.7	Mov. Avg: 34.9
Episode 163 (145 sec)  -- 	Min: 31.0	Max: 39.0	Mean: 35.9	Mov. Avg: 34.9
Episode 164 (144 sec)  -- 	Min: 25.7	Max: 39.4	Mean: 34.0	Mov. Avg: 34.8
Episode 165 (143 sec)  -- 	Min: 33.2	Max: 39.2	Mean: 36.5	Mov. Avg: 34.8
Episode 166 (142 sec)  -- 	Min: 29.0	Max: 37.9	Mean: 35.0	Mov. Avg: 34.8
Episode 167 (146 sec)  -- 	Min: 31.6	Max: 38.4	Mean: 35.6	Mov. Avg: 34.8
Episode 168 (143 sec)  -- 	Min: 33.7	Max: 39.1	Mean: 37.1	Mov. Avg: 34.8
Episode 169 (142 sec)  -- 	Min: 30.4	Max: 39.1	Mean: 35.4	Mov. Avg: 34.8
Episode 170 (142 sec)  -- 	Min: 31.1	Max: 39.3	Mean: 36.9	Mov. Avg: 34.8
Episode 171 (142 sec)  -- 	Min: 24.0	Max: 36.9	Mean: 31.6	Mov. Avg: 34.7
Episode 172 (142 sec)  -- 	Min: 15.0	Max: 37.3	Mean: 28.7	Mov. Avg: 34.6
Episode 173 (143 sec)  -- 	Min: 17.4	Max: 34.2	Mean: 26.8	Mov. Avg: 34.5
Episode 174 (143 sec)  -- 	Min: 22.2	Max: 36.2	Mean: 31.6	Mov. Avg: 34.5
Episode 175 (141 sec)  -- 	Min: 21.5	Max: 37.9	Mean: 33.3	Mov. Avg: 34.4
Episode 176 (141 sec)  -- 	Min: 27.1	Max: 37.3	Mean: 33.3	Mov. Avg: 34.4
Episode 177 (142 sec)  -- 	Min: 28.8	Max: 38.3	Mean: 34.3	Mov. Avg: 34.4
Episode 178 (144 sec)  -- 	Min: 27.0	Max: 38.7	Mean: 36.1	Mov. Avg: 34.4
Episode 179 (142 sec)  -- 	Min: 31.6	Max: 39.4	Mean: 35.7	Mov. Avg: 34.4
Episode 180 (142 sec)  -- 	Min: 24.9	Max: 37.4	Mean: 33.2	Mov. Avg: 34.4
Episode 181 (142 sec)  -- 	Min: 26.3	Max: 39.1	Mean: 31.8	Mov. Avg: 34.3
Episode 182 (145 sec)  -- 	Min: 30.6	Max: 38.4	Mean: 35.0	Mov. Avg: 34.3
Episode 183 (140 sec)  -- 	Min: 27.4	Max: 39.4	Mean: 34.5	Mov. Avg: 34.3
Episode 184 (142 sec)  -- 	Min: 28.8	Max: 39.5	Mean: 35.9	Mov. Avg: 34.3
Episode 185 (141 sec)  -- 	Min: 33.3	Max: 37.7	Mean: 36.1	Mov. Avg: 34.4
Episode 186 (141 sec)  -- 	Min: 28.6	Max: 39.4	Mean: 36.0	Mov. Avg: 34.4
Episode 187 (143 sec)  -- 	Min: 24.4	Max: 39.4	Mean: 35.1	Mov. Avg: 34.4
Episode 188 (145 sec)  -- 	Min: 24.8	Max: 38.4	Mean: 35.4	Mov. Avg: 34.4
Episode 189 (146 sec)  -- 	Min: 30.2	Max: 38.3	Mean: 34.7	Mov. Avg: 34.5
Episode 190 (144 sec)  -- 	Min: 32.9	Max: 39.1	Mean: 35.8	Mov. Avg: 34.5
Episode 191 (146 sec)  -- 	Min: 31.5	Max: 38.8	Mean: 35.6	Mov. Avg: 34.5
Episode 192 (144 sec)  -- 	Min: 31.1	Max: 38.8	Mean: 35.8	Mov. Avg: 34.6
Episode 193 (142 sec)  -- 	Min: 23.2	Max: 38.0	Mean: 33.4	Mov. Avg: 34.6
Episode 194 (143 sec)  -- 	Min: 10.0	Max: 35.1	Mean: 24.1	Mov. Avg: 34.5
Episode 195 (144 sec)  -- 	Min: 0.3	Max: 30.1	Mean: 15.3	Mov. Avg: 34.3
Episode 196 (143 sec)  -- 	Min: 9.2	Max: 33.2	Mean: 20.6	Mov. Avg: 34.1
Episode 197 (146 sec)  -- 	Min: 6.6	Max: 34.0	Mean: 24.5	Mov. Avg: 34.0
Episode 198 (143 sec)  -- 	Min: 11.5	Max: 32.3	Mean: 26.1	Mov. Avg: 33.9
Episode 199 (144 sec)  -- 	Min: 10.9	Max: 30.7	Mean: 23.1	Mov. Avg: 33.8
Episode 200 (142 sec)  -- 	Min: 17.8	Max: 31.4	Mean: 25.5	Mov. Avg: 33.7
```

![image](https://user-images.githubusercontent.com/8471958/101865766-05fd0580-3b2c-11eb-8d7a-060e2b4f1e99.png)

#### Future ideas for improving the agent's performance

1. Using the Prioritized Experince Replay(PER) instead of Replay Buffer
   - In the previous project(Navigation), there were always better results when use PER instead of Replay Buffer

2. Using different algorithms like [Trust Region Policy Optimization(TRPO) algorithm](https://arxiv.org/pdf/1502.05477.pdf) or [Proximal Policy Optimization(PPO) algorithm](https://arxiv.org/pdf/1707.06347.pdf)