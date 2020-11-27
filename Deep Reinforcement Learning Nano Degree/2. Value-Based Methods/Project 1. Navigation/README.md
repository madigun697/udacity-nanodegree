[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

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

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
