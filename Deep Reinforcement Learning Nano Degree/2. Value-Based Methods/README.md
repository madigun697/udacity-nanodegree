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

## Deep Q-Networks

Deep Q-Networks is deep learning networks that use states as input and possible actions as output

- Trining Techniques: Experience Replay, Fixed Q-Targets
- Structure
  - Steps
    1. Initialize memory <img src="https://render.githubusercontent.com/render/math?math=D">(replay buffer, finite size <img src="https://render.githubusercontent.com/render/math?math=N">)
    2. Initialize action-value function <img src="https://render.githubusercontent.com/render/math?math=\hat{q}"> with random weight <img src="https://render.githubusercontent.com/render/math?math=w">
    3. Initialize target action-value weight <img src="https://render.githubusercontent.com/render/math?math=w^- \leftarrow w">
    4. Iterate episodes (Sampling and Learning)
  - Sampling: Run and store interactions between agent and environment
  - Learning: Select one of stored experience randomly and update <img src="https://render.githubusercontent.com/render/math?math=w">

### Experience Replay

Agent train again with stored interaction between agent and environment

- Replay buffer: Store experience as table of tuple <img src="https://render.githubusercontent.com/render/math?math=(S, A, R, S')">
- Advantages
  - Convert reinforcement learning problem into supervised learning problem
  - Enhance agent training with rare experience
- To avoid the effect of high correlation between state and action, using random selection when train with replay buffer

### Fixed Q-Targets

