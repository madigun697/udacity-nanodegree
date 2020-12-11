[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the same folder other files placed , and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!

1. Download and unzip environment file

2. Set up environment

   ```python
   # file_name is unziped environment file path
   env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64')
   ```

3. Run codes for initialize agent

   - Get default brain

     ```python
     brain_name = env.brain_names[0]
     brain = env.brains[brain_name]
     ```

   - Get the number of agents, action size, state size

     ```python
     # reset the environment
     env_info = env.reset(train_mode=True)[brain_name]
     
     # number of agents
     num_agents = len(env_info.agents)
     
     # size of each action
     action_size = brain.vector_action_space_size
     
     # examine the state space 
     states = env_info.vector_observations
     state_size = states.shape[1]
     ```

   - Initialize agent

     ```python
     # initialize agent with hidden layer's dimensions and random seed
     agent = Agent(state_size, action_size, hidden_dims, random_seed)
     ```

4. Run agent

   ```python
   mean_scores = []                               # list of mean scores from each episode
   min_scores = []                                # list of lowest scores from each episode
   max_scores = []                                # list of highest scores from each episode
   scores_window = deque(maxlen=100)              # mean scores from most recent episodes
   moving_avgs = []                               # list of moving averages
   for i_episode in range(1, 201):								 # run episode 1~201 (200 episodes)
       env_info = env.reset(train_mode=True)[brain_name]
       states = env_info.vector_observations
       score = np.zeros(num_agents)
       agent.reset()
       start_time = time.time()
       for t in range(1000):
           actions = agent.act(states)
          
           env_info = env.step(actions)[brain_name]
           next_states = env_info.vector_observations
           rewards = env_info.rewards
           dones = env_info.local_done
           
           for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
               agent.step(state, action, reward, next_state, done, t)
               
           states = next_states
           score += rewards
           if np.any(dones):
               break
           
       agent.save_model_params()
           
       duration = time.time() - start_time
       min_scores.append(np.min(score))              # save lowest score for a single agent
       max_scores.append(np.max(score))              # save highest score for a single agent        
       mean_scores.append(np.mean(score))            # save mean score for the episode
       scores_window.append(mean_scores[-1])         # save mean score to window
       moving_avgs.append(np.mean(scores_window))    # save moving average
       
       print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format(\
             i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]))
   ```

5. Check result

   - Print each episode result

     ```
     Episode 1 (91 sec)  -- 	Min: 0.0	Max: 1.9	Mean: 0.7	Mov. Avg: 0.7
     Episode 2 (94 sec)  -- 	Min: 0.0	Max: 3.9	Mean: 1.7	Mov. Avg: 1.2
     Episode 3 (95 sec)  -- 	Min: 1.9	Max: 6.1	Mean: 3.7	Mov. Avg: 2.0
     ...
     Episode 198 (143 sec)  -- 	Min: 11.5	Max: 32.3	Mean: 26.1	Mov. Avg: 33.9
     Episode 199 (144 sec)  -- 	Min: 10.9	Max: 30.7	Mean: 23.1	Mov. Avg: 33.8
     Episode 200 (142 sec)  -- 	Min: 17.8	Max: 31.4	Mean: 25.5	Mov. Avg: 33.7
     ```

   - Show the chart of result

     ```python
     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
     ax.plot(np.arange(1, len(mean_scores)+1), mean_scores)
     ax.set_ylabel('Score')
     ax.set_xlabel('Episode #')
     ```

     ![image](https://user-images.githubusercontent.com/8471958/101865766-05fd0580-3b2c-11eb-8d7a-060e2b4f1e99.png)

#### Agent

- **Algorithm**: Deep Deterministic Policy Gradients(DDPG) algorithm
  - DDPG algorithm apply the advantages of DQN algorithm into the Actor-Critifc approach
    - Replay Buffer: Reduce the correlation between samples
    - Target Q Network: Be stable network during update
  - Actor and Critic Network Structure
    ![image](https://user-images.githubusercontent.com/8471958/101945108-91b17900-3ba2-11eb-9325-1d5f89db713c.png)
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


### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Crawler** environment.

![Crawler][image2]

In this continuous control environment, the goal is to teach a creature with four legs to walk forward without falling.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Crawler.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

