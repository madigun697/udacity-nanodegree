[//]: # "Image References"

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### [Dependencies](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md)

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

   - __Linux__ or __Mac__: 

   ```bash
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   - __Windows__: 

   ```bash
   conda create --name drlnd python=3.6 
   activate drlnd
   ```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  

   - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
   - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image3]

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  

1. Downlaod and unzip environment file

2. Set up environment

   ```python
   # file_name is unziped environment file path
   env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
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
     # Initialize agent with hidden layer's dimensions and random seed
     from MADDPG_Agent import MADDPG_Agent
     agent = MADDPG_Agent(state_size, action_size, [400, 300], num_agents, 1)
     # If you want to use PER(Prioritized Experience Replay) instead of default Memory Buffer, set per_flag=True
     # agent = MADDPG_Agent(state_size, action_size, [400, 300], num_agents, 1, per_flag=True)
     ```

4. Run agent

   ```python
   all_scores = []													# List of highest scores from each episode
   scores_window = deque(maxlen=100)				# Higheste scores from most recent episodes
   
   start_time = time.time()
   
   for i_episode in range(1, 10001):
       env_info = env.reset(train_mode=True)[brain_name]
       states = env_info.vector_observations
       scores = np.zeros(num_agents)
       agent.reset()
   
       for t in range(1000):
           actions = agent.act(states)
          
           env_info = env.step(actions)[brain_name]
           next_states = env_info.vector_observations
           rewards = env_info.rewards
           dones = env_info.local_done
           
           agent.step(states, actions, rewards, next_states, dones, t)
               
           states = next_states
           scores += rewards
           if np.any(dones):
               break
           
       agent.save_model_params()
           
       episode_score = np.max(scores)
       all_scores.append(episode_score)			# Save the highest score for the episode
       scores_window.append(episode_score)		# Save the highest score to window
       avg_score = np.mean(scores_window)		# Average score from the most recent episodes
       
       duration = time.time() - start_time
       
       print('\rEpisode %d (%d sec) \t -- Average Score: %.2f\tEpisode Score: %.2f' % (i_episode, round(duration), avg_score, episode_score), end='')
       
       if i_episode > 1 and i_episode % 100 == 0:
           duration = time.time() - start_time
           start_time = time.time()
           
           print('\rEpisode %d (%d sec) \t -- Average Score: %.2f\tMax Score: %.2f    ' % (i_episode, round(duration), avg_score, np.max(all_scores)))
           
       if (i_episode > 99) and (avg_score >=0.5):
           duration = time.time() - start_time
           
           print('\rEpisode %d (%d sec) \t -- Average Score: %.2f\tMax Score: %.2f    ' % (i_episode, round(duration), avg_score, np.max(all_scores)))
           agent.save_model_params()
           break
   ```

5. Check result

   - Print each 100 episode result

     ```python
     Episode 100 (32 sec) 	 -- Average Score: 0.00	Max Score: 0.00
     Episode 200 (35 sec) 	 -- Average Score: 0.00	Max Score: 0.10
     Episode 300 (34 sec) 	 -- Average Score: 0.00	Max Score: 0.10
     ...
     Episode 3000 (200 sec) 	 -- Average Score: 0.23	Max Score: 1.50
     Episode 3100 (278 sec) 	 -- Average Score: 0.32	Max Score: 1.80
     Episode 3152 (266 sec) 	 -- Average Score: 0.51	Max Score: 2.60
     ```

   - Show the chart of result

     ```python
     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
     ax.plot(np.arange(1, len(mean_scores)+1), mean_scores)
     ax.set_ylabel('Score')
     ax.set_xlabel('Episode #')
     ```

     ![image](https://user-images.githubusercontent.com/8471958/102172931-c21d3f80-3e4e-11eb-8b67-7e2870a167eb.png)

### (Optional) Challenge: Crawler Environment

After you have successfully completed the project, you might like to solve the more difficult **Soccer** environment.

![Soccer][image2]

In this environment, the goal is to train a team of agents to play soccer.  

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos).  To solve this harder task, you'll need to download a new Unity environment.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Then, place the file in the `p3_collab-compet/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Soccer.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agents on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agents without enabling a virtual screen, but you will be able to train the agents.  (_To watch the agents, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)