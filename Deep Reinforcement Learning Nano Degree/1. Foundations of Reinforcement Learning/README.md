# 1. Foundations of Reinforcement Learning

**Official course description**

*The first part begins with a simple introduction to reinforcement learning.  You'll learn how to define real-world problems as **Markov Decision Processes (MDPs)**, so that they can be solved with reinforcement learning.*  



*![How might we use reinforcement learning to teach a robot to walk?](https://video.udacity-data.com/topher/2018/May/5af5ba38_mjg2mzcyma/mjg2mzcyma.gif)*

*How might we use [reinforcement learning](https://arxiv.org/pdf/1803.05580.pdf) to teach a robot to walk? ([Source](https://spectrum.ieee.org/automaton/robotics/industrial-robots/agility-robotics-introduces-cassie-a-dynamic-and-talented-robot-delivery-ostrich))*

*Then, you'll implement classical methods such as **SARSA** and **Q-learning** to solve several environments in OpenAI Gym.  You'll then explore how to use techniques such as **tile coding** and **coarse coding** to expand the size of the problems that can be solved with traditional reinforcement learning algorithms.*



*![Train a car to navigate a steep hill using Q-learning.](https://video.udacity-data.com/topher/2018/June/5b1721df_mountain-car-cts/mountain-car-cts.gif)*

*Train a car to navigate a steep hill using Q-learning.*

## Foundations of Reinforcement Leraning

### Reinforcement learning(RL)

- Building code that can learn to perform complex tasks by itself

#### Applications

- Games: AlphaGo, Atari breakout, DOTA
- Self-driving: Car(Uber, Google), Ship, Airplane
- Robotics: Walking robots

#### Terminologies

- **Agent**: learner or decision maker, born into the world w/o any understanding of how anything works

- **Feedback**: Rewards(Positive feedback) or discoursing feedback

- **Goal**: Maximize rewards

#### Exploration-Exploitation Dilemma

- **Exploration**: Exploring potential hypotheses for how to choose actions

- **Exploitation**: Exploiting limited knowledge about what is already known should work well

- Balancing these competing requirements

#### Reinforcement learning structure

- The agent learned to interact with environment

  ![image](https://user-images.githubusercontent.com/8471958/98735523-59770a80-2358-11eb-911b-23bbceafb418.png)

- State(Observation, <img src="https://render.githubusercontent.com/render/math?math=S_t">), Action(<img src="https://render.githubusercontent.com/render/math?math=A_t">), Reward(<img src="https://render.githubusercontent.com/render/math?math=R_t">)
- **Goal of the Agent: Maximize expected cumulative reward**

#### Epicsodic task & Continuing task

- Task: an instance of the reinforcement learning problem
- Epicsodic task
  - Tasks with a well-defined starting and ending point
  - There is a specific ending point(**Terminal state**)
  - An Episode means that interaction ends at some time step <img src="https://render.githubusercontent.com/render/math?math=T">
  - Reward is given at ending point
- Continuing task
  - Tasks that continue forever, without end
  - Interaction continues without limit

#### Goal and Rewards

##### Goal

-  Reward Hypothesis: All goals can be framed as the maximazation of expected cumulative reward

##### Reward

- Cumulative return: <img src="https://render.githubusercontent.com/render/math?math=G_t = R_{t%2B1}%2BR_{t%2B2}%2BR_{t%2B3}%2B\R_{t%2B4}%2B\cdots">
  - At time step <img src="https://render.githubusercontent.com/render/math?math=t">, the agent picks <img src="https://render.githubusercontent.com/render/math?math=A_t">to maximize (expected) <img src="https://render.githubusercontent.com/render/math?math=G_t">
- Discounted return: <img src="https://render.githubusercontent.com/render/math?math=G_t = R_{t%2B1}%2B\gamma R_{t%2B2}%2B\gamma^2 R_{t%2B3}%2B\gamma^3 R_{t%2B4}%2B\cdots">
  - discount rate <img src="https://render.githubusercontent.com/render/math?math=\gamma \in [0, 1]">

