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

