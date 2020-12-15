# 4. Multi-Agent Reinforcement Learning

**Official course description**

*Most of reinforcement learning is concerned with a single agent that  seeks to demonstrate proficiency at a single task.  In this agent's  environment, there are no other agents.  However, if we'd like our  agents to become truly intelligent, they must be able to communicate  with and learn from other agents.  In the final part of this nanodegree, we will extend the traditional framework to include multiple agents.*

*You'll also learn all about **Monte Carlo Tree Search (MCTS)** and master the skills behind DeepMind's AlphaZero.*



*![Use Monte Carlo Tree Search to play Connect 4. ([Source](https://github.com/Alfo5123/Connect4))](https://video.udacity-data.com/topher/2018/May/5afc628c_game-example/game-example.gif)*

*Use Monte Carlo Tree Search to play Connect 4. ([Source](https://github.com/Alfo5123/Connect4))*

*You'll also get the **third project**, where you'll write an algorithm to train a pair of agents to play tennis.*



*![In Project 3, you will train a pair of agents to play tennis.](https://video.udacity-data.com/topher/2018/May/5af5c69e_68747470733a2f2f626c6f67732e756e69747933642e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30392f696d616765322d322e676966/68747470733a2f2f626c6f67732e756e69747933642e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30392f696d616765322d322e676966.gif)*

*In Project 3, you will train a pair of agents to play tennis. ([Source](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/))*

[toc]

## Multi-Agent Systems

- Multi-Agent System
  - Introduction
    - There are several agents in the same environment
    - The agents interact with not only environment but also other agents' actions
  - Motivations
    - We live in a multi-agent world
    - Intelligent agents have to interact with humnas
    - Agents need to work in complex environments
  - Benefits
    - Agents can share their experience(knowledge) with other agents
    - Robust: When one agent failed, another agent can take over the tasks
- Multi-Agent Reinforcement Learning (MARL)
  - Agents train without considering other agents
    - Other agents are just part of environment(state)
    - Non-Stationarity environment: each agent recognizes state differently, according to other agents' action
  - The matter agent approach
    - With single policy, return action vector for each agent
  - Multi-Agent Environment
    - Cooperation: Maximize rewards of each agent
    - Competition: Maximize their own rewards (One agent take, the others lose)
    - Mixed Environment

## AlphaZero

### Zero-Sum Game

- Compete two agents (If one win, other must lose)

### Monte Carlo Tree Search (MCTS)

- MCTS in Zero-Sum Game
  - Initialize top-node for current state, loop over actions for some $N_{tot}$:
    1. Start from the top-node, repeatedly pick the child-node with the alrgets $U$
    2. If $N = 0$ for the node, play a random game.
       Else, expand node, play a random game from a randomly selected child
    3. Update statistics, back-propagate and update $N$ and $U$ as neede
  - Select move with highest visit counts



