# 2. Value-Based Methods

**Official course description**

*In the second part, you'll learn how to leverage neural networks when solving complex problems using the **Deep Q-Networks (DQN)** algorithm.  You will also learn about modifications such as **double Q-learning**, **prioritized experience replay**, and **dueling networks**. Then, you'll use what you’ve learned to create an artificially intelligent game-playing agent that can navigate a spaceship!*



*![Use the DQN algorithm to train a spaceship to land safely on a planet.](https://video.udacity-data.com/topher/2018/June/5b172a69_lunar-lander/lunar-lander.gif)*

*Use the DQN algorithm to train a spaceship to land safely on a planet.*

*You'll learn from experts at NVIDIA's Deep Learning Institute how to apply your new skills to robotics applications.  Using a [Gazebo](http://gazebosim.org) simulation, you will train a rover to navigate an environment without running into walls.*



*![Learn from experts at NVIDIA how to navigate a rover!](https://video.udacity-data.com/topher/2018/May/5b02fa9c_output/output.gif)*

*Learn from experts at NVIDIA how to navigate a rover!*

*You'll also get the **first project**, where you'll write an algorithm that teaches an agent to navigate a large world.*  



*![In Project 1, you will train an agent to collect yellow bananas while avoiding blue bananas.](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)*

*In Project 1, you will train an agent to collect yellow bananas while avoiding blue bananas.*

*All of the projects in this Nanodegree program use the rich simulation environments from the [Unity Machine Learning Agents (ML-Agents)](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/) software development kit (SDK).  You will learn more about ML-Agents in the next concept.*

[toc]

## [Deep Q-Networks](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Reinforcement%20Learning%20Nano%20Degree/2.%20Value-Based%20Methods/Lesson%202.%20Deep%20Q-Networks)

Deep Q-Networks is deep learning networks that use states as input and possible actions as output

- In the DQN, the next action is determined by greedy action expected maximum reward
  - In the traditional RL(Q-Learning), the agent can get greedy action by Q-table
  - In the DQN, greedy action is determined by the nueral network
- Issues
  - The high correlation between samples: The next sample(state) is related to previous sample
  - Non-stationary target: In the RL, the optimal Q is not stable

- Training Techniques (to solve issues): Experience Replay, Fixed Q-Targets
- Structure
  - Steps
    1. Initialize memory <img src="https://render.githubusercontent.com/render/math?math=D">(replay buffer, finite size <img src="https://render.githubusercontent.com/render/math?math=N">)
    2. Initialize action-value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}"> with random weight <img src="https://render.githubusercontent.com/render/math?math=w">
    3. Initialize target action-value weight <img src="https://render.githubusercontent.com/render/math?math=w^- \leftarrow w">
    4. Iterate episodes (Sampling and Learning)
  - Sampling: Run and store interactions between agent and environment
  - Learning: Select one of stored experience randomly and update <img src="https://render.githubusercontent.com/render/math?math=w">
- $\Delta w = \alpha \cdot \overbrace{( \underbrace{R + \gamma \max_a\hat{q}(S',  a, w^-)}_{\rm {TD~target}} - \underbrace{\hat{q}(S, A, w)}_{\rm  {old~value}})}^{\rm {TD~error}} \nabla_w\hat{q}(S, A, w)$

### Experience Replay

Agent train again with stored interaction between agent and environment

- Replay buffer: Store experience as table of tuple <img src="https://render.githubusercontent.com/render/math?math=(S, A, R, S')">
- Advantages
  - Convert reinforcement learning problem into supervised learning problem
  - Enhance agent training with rare experience
- To avoid the effect of high correlation between state and action, using random selection when train with replay buffer

### Fixed Q-Targets

To avoid non-stationary target issue, uses two network with same structure

- Local network
  - Use this netwrok when the agent need to get next action
  - Optimize the weights (minimize MSE loss) using target network as label
  - <img src="https://render.githubusercontent.com/render/math?math=w"> is the parameter in the local network
- Target network
  - Update the network using replay buffer
  - <img src="https://render.githubusercontent.com/render/math?math=w^-"> is the parameter in the target network

## Extensions to DQN

There are several improvements to the original DQN

- Six major extensions to DQN
  1. Double DQN
  2. Prioritized Experience Replay
  3. Dueling DQN
  4. [Multi-step bootstrap targets](https://arxiv.org/abs/1602.01783)
  5. [Distributional DQN](https://arxiv.org/abs/1707.06887)
  6. [Noisy DQN](https://arxiv.org/abs/1706.10295)

### Double DQN

- Seperate the selection and evaluation(TD target)
  - The DQN does the selection(to select an action to obtain maximum reward in the next state) and evaluation(to calculate the extimated sum of future rewards, <img src="https://render.githubusercontent.com/render/math?math=Q">) with single model
  - The D-DQN seperate these two process to avoid overestimation and error of noises
- Selection uses local network(<img src="https://render.githubusercontent.com/render/math?math=w">) and Evaluation uses target netwrok(<img src="https://render.githubusercontent.com/render/math?math=w^-">)
- $\Delta w = \alpha \cdot \overbrace{( \underbrace{R + \gamma \hat{q} ( S', \arg max_a\hat{q}(S',  a, w), w^-)}_{\rm {TD~target}} - \underbrace{\hat{q}(S, A, w)}_{\rm  {old~value}})}^{\rm {TD~error}} \nabla_w\hat{q}(S, A, w)$

### Prioritized Experience Replay

- To overcome the shortcomings of the original experience replay
  - Because samples to train the model are selected uniformly, some experiences have a very small chance of getting selected
  - If the train period is long, some old experiences got lost chance to select, because the memory size is limited
- Add the priority in the each experience
  - TD error, the difference between the target <img src="https://render.githubusercontent.com/render/math?math=Q"> and expected <img src="https://render.githubusercontent.com/render/math?math=Q"> is used the basement of priority
  - The priority is the sum of TD error and small <img src="https://render.githubusercontent.com/render/math?math=\epsilon">. To avoid the non-selected situation, although the TD error is zero
- Sampling probability
  - <img src="https://render.githubusercontent.com/render/math?math=\alpha">: The control value between prioritized selection and uniform selection
    - <img src="https://render.githubusercontent.com/render/math?math=P(i) = {p^ \alpha_i} \over {\sum_k P^ \alpha_k}"> 
  - <img src="https://render.githubusercontent.com/render/math?math=\beta">: To adjust bias according the non-uniform random sampling
    - <img src="https://render.githubusercontent.com/render/math?math=w_i = ({1} \over {N} {1} \over {P(i)})^ \beta"> 

### Dueling DQN

![image](https://user-images.githubusercontent.com/8471958/99925590-9990a280-2cf3-11eb-869e-26f67c5c238c.png)

- Seperates Q-value into state value and advantage value
  - <img src="https://render.githubusercontent.com/render/math?math=Q(s, a) = V(s) %2B A(a, s)"> 
  - Advantage value means that how much better selecting a specific action than other actions

### Rainbow

- Rainbow DQN is combined six extension to DQN

## [[Project] Navigation](https://github.com/madigun697/udacity-nanodegree/tree/master/Deep%20Reinforcement%20Learning%20Nano%20Degree/2.%20Value-Based%20Methods/Project%201.%20Navigation)

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.