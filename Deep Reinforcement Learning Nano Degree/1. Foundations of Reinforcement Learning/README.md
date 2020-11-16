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
- The agent learned to interact with environment
- **Feedback**: Rewards(Positive feedback) or discoursing feedback
- **Goal**: Maximize rewards

#### Exploration-Exploitation Dilemma

- **Exploration**: Exploring potential hypotheses for how to choose actions

- **Exploitation**: Exploiting limited knowledge about what is already known should work well

- Balancing these competing requirements

### Reinforcement learning framework

#### Setting

![image](https://user-images.githubusercontent.com/8471958/98735523-59770a80-2358-11eb-911b-23bbceafb418.png)

- State(Observation, <img src="https://render.githubusercontent.com/render/math?math=S_t">): the environment presents a situation to the agent
- Action(<img src="https://render.githubusercontent.com/render/math?math=A_t">): appropriate actions in response
- Reward(<img src="https://render.githubusercontent.com/render/math?math=R_t">): One time step later, the agent receives
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

#### Rewards

-  Reward Hypothesis: All goals can be framed as the maximazation of expected cumulative reward

- Cumulative reward(return): <img src="https://render.githubusercontent.com/render/math?math=G_t = R_{t%2B1}%2BR_{t%2B2}%2BR_{t%2B3}%2B\R_{t%2B4}%2B\cdots">
  - At time step <img src="https://render.githubusercontent.com/render/math?math=t">, the agent picks <img src="https://render.githubusercontent.com/render/math?math=A_t">to maximize (expected) <img src="https://render.githubusercontent.com/render/math?math=G_t">
- Discounted reward(return): <img src="https://render.githubusercontent.com/render/math?math=G_t = R_{t%2B1}%2B\gamma R_{t%2B2}%2B\gamma^2 R_{t%2B3}%2B\gamma^3 R_{t%2B4}%2B\cdots">
  - discount rate <img src="https://render.githubusercontent.com/render/math?math=\gamma \in [0, 1]">

#### Markov Decision Process (MDP)

![img](https://video.udacity-data.com/topher/2017/September/59c3f51a_screen-shot-2017-09-21-at-12.20.30-pm/screen-shot-2017-09-21-at-12.20.30-pm.png)

- A (finite) MDP is defined by
  - a (finite) set of states <img src="https://render.githubusercontent.com/render/math?math=S">
  - a (finite) set of actions <img src="https://render.githubusercontent.com/render/math?math=A">
  - a (finite) set of rewards <img src="https://render.githubusercontent.com/render/math?math=R">
  - the one-step dynamics of the environment
    <img src="https://render.githubusercontent.com/render/math?math=p(s',r|s,a) \doteq \mathbb{P}(S_{t%2B1}=s', R_{t%2B1}=r|S_t = s, A_t=a)"> for all <img src="https://render.githubusercontent.com/render/math?math=s, s`, a and r">
  - a discount rate <img src="https://render.githubusercontent.com/render/math?math=\gamma \in [0, 1]">

#### Policies

##### Policies

- A policy determines how an agent chooses an action in response to the current state
- It specifies how the agent responds to situations that the environment has presented.

- Deterministic policy(<img src="https://render.githubusercontent.com/render/math?math=\pi:\mathcal{S} \rightarrow \mathcal{A}">)
  - Return a specific action by state
- Stochastic policy(<img src="https://render.githubusercontent.com/render/math?math=\pi:\mathcal{S} \times \mathcal{A} \rightarrow [0,1])">
  - <img src="https://render.githubusercontent.com/render/math?math=\pi(a|s) = \mathbb{P}(A_t = a | S_t = s)"> 
  - Return probabilities of action set by state and action set

##### Optimal Policies

1. State-Value Function
   - The value of state <img src="https://render.githubusercontent.com/render/math?math=s"> under a policy <img src="https://render.githubusercontent.com/render/math?math=\pi">
   - <img src="https://render.githubusercontent.com/render/math?math=v_\pi(s) = \mathbb{E}_\pi[G_t|S_t=s]"> 
   - For each state <img src="https://render.githubusercontent.com/render/math?math=s">, it yields the expected return(<img src="https://render.githubusercontent.com/render/math?math=G_t">), if the agent start in state s(<img src="https://render.githubusercontent.com/render/math?math=S_t=s">) and then uses policy(<img src="https://render.githubusercontent.com/render/math?math=\pi">) to choose its actions for all time steps.

   - Bellman Expectation Equation
     - Equation to calculate the value of any state is the sum of the immediate reward and the discounted value of the state that follow.
     - <img src="https://render.githubusercontent.com/render/math?math=v_\pi(s) = \mathbb{E}_\pi[R_{t%2B1} %2B \gamma v_\pi(S_{t%2B1}|S_t = s)]"> 

2. Action-Value Function

   - The value of taking action <img src="https://render.githubusercontent.com/render/math?math=a"> in state <img src="https://render.githubusercontent.com/render/math?math=s"> under a policy <img src="https://render.githubusercontent.com/render/math?math=\pi">
   - <img src="https://render.githubusercontent.com/render/math?math=q_\pi(s, a) = \mathbb{E}_\pi[G_t|S_t=s,A_t=a]"> 
   - <img src="https://render.githubusercontent.com/render/math?math=v_\pi(s) = q_\pi(s, \pi(s))$ if $\pi$ is a deterministic policy and all $s \in \mathcal{S}"> 

3. Optimality
   - Compare with two policies
     - <img src="https://render.githubusercontent.com/render/math?math=\pi' \ge \pi"> if and only if <img src="https://render.githubusercontent.com/render/math?math=v_{\pi'}(s) \ge v_\pi(s)\text{ for all }s \in \mathcal{S}">
   - The optimal policy <img src="https://render.githubusercontent.com/render/math?math=\pi_\star"> satisfies <img src="https://render.githubusercontent.com/render/math?math=\pi_\star \ge \pi \text{ for all }\pi">
   - Once the agent determines the optimal action-value function <img src="https://render.githubusercontent.com/render/math?math=q_*">, it can quickly obtain an optimal policy <img src="https://render.githubusercontent.com/render/math?math=\pi_*"> by setting <img src="https://render.githubusercontent.com/render/math?math=\pi_*(s) = \arg\max_{a\in\mathcal{A}(s)} q_*(s,a)">.

### Monte Carlo Methods

- Equiprobable random policy: the agent choose an action in the action set with same probabilities.

- The action-value function with a Q-table
  - To find optimal policy, the agent tries many episodes.
  - Q-table is the expected value matrix by states and actions based on results of episodes.
    - Each episode create a matrix and final Q-table has average values of these matrixes.
- MC Prediction
  - Every-visit MC Prediction: Fill out the Q-table with average value of observations
  - First-visit MC Prediction: Fill out the Q-table with value of first observation

#### MC Control

- Control Problem: Estimate the optimal policy
- MC Control Method: a solution for the control problem
  - Step 1: Using the policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> to construct the Q-table
  - Step 2: improving the policy by changing it to be <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy with respect to the Q-table (<img src="https://render.githubusercontent.com/render/math?math=\pi' \leftarrow \epsilon\text{-greedy}(Q), \pi \leftarrow \pi'">)
  - Eventually obtain the optimal policy <img src="https://render.githubusercontent.com/render/math?math=\pi_*">

#### Greedy Policies

- Greedy Policies
  - Collect episodes with <img src="https://render.githubusercontent.com/render/math?math=\pi"> and estimate the Q-table
  - Initial policy is the equiprobable random policy
  - Use the Q-table to find a better policy <img src="https://render.githubusercontent.com/render/math?math=\pi'"> and set <img src="https://render.githubusercontent.com/render/math?math=\pi \leftarrow \pi'">
  - Iterate these steps

- Epsilon-Greedy Policies
  - Greedy policy is always selects the greedy action
  - Epsilon-Greedy policy is most likely selects the greedy action
  - Set a specific value <img src="https://render.githubusercontent.com/render/math?math=\epsilon (0\le\epsilon\le1)">
  - <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> is a probability that the agent selects random actions instead of the greedy action

#### Exploration-Exploitation Dilemma

- Exploration: Prefer to select randomly
- Exploitation: Prefer to select the greedy action
- Greedy in the Limit with Infinite Exploration(GLIE)
  - The conditions to guarantee that MC control converges to the optimal policy <img src="https://render.githubusercontent.com/render/math?math=\pi_*">
  - Every state-action pair <img src="https://render.githubusercontent.com/render/math?math=s, a">(for all <img src="https://render.githubusercontent.com/render/math?math=s \in \mathcal{S}"> and <img src="https://render.githubusercontent.com/render/math?math=a \in \mathcal{A}(s)">) is visited infinitely many times
    - The agent continues to explore (never stop to explore)
    - <img src="https://render.githubusercontent.com/render/math?math=\epsilon_t > 0"> 
  - The policy converges to a policy that is greedy with respect to the action-value function estimate <img src="https://render.githubusercontent.com/render/math?math=\Q">
    - The agent gradually exploits more and explores less
    - <img src="https://render.githubusercontent.com/render/math?math=\lim_{t\rightarrow \infty}\epsilon_t = 0"> 

####  Incremental Mean

- To reduce time, update Q-table after every episode.
- Update Q-table could be used to improve the policy
  - <img src="https://render.githubusercontent.com/render/math?math=Q \leftarrow Q %2B {1 \over N}(G - Q)"> 

- Problem
  - <img src="https://render.githubusercontent.com/render/math?math=N">(The number of steps in the episode) is bigger and bigger, the effect of denotation(<img src="https://render.githubusercontent.com/render/math?math=G-Q">) is smaller and smaller
  - Every state-action pair is not effected to <img src="https://render.githubusercontent.com/render/math?math=Q"> evenly

#### Constant-alpha

- To solve the problem of Incremental Mean, uses constant value, <img src="https://render.githubusercontent.com/render/math?math=\alpha"> instead of <img src="https://render.githubusercontent.com/render/math?math=1 \over N">
  - <img src="https://render.githubusercontent.com/render/math?math=Q \leftarrow Q %2B \alpha(G-Q)">
    <img src="https://render.githubusercontent.com/render/math?math=Q \leftarrow (1-\alpha)Q %2B \alpha G">
  - if <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0">, then the Q-table is never updated → **Exploitation**
    if <img src="https://render.githubusercontent.com/render/math?math=\alpha = 1">, then the previous result(previous $Q$) is always ignored → **Exploration**

### Temporal-Difference Methods

- Monte Carlo methods need to end the interaction
  - In the self-driving car, MC methods updates the policy after crash
- Temporal-Difference methods updates the policy every step

#### TD Control

- It's similar to MC constant-alpha
- The value is updated in every step instead of after episode ends
  - MC: Update the effect of denotation <img src="https://render.githubusercontent.com/render/math?math=G-Q">, where <img src="https://render.githubusercontent.com/render/math?math=G"> is the cumulated rewards after time <img src="https://render.githubusercontent.com/render/math?math=t">
  - TD (**Sarsa**): Using <img src="https://render.githubusercontent.com/render/math?math=R_{t%2B1} %2B \gamma Q_{t%2B1}"> instead of <img src="https://render.githubusercontent.com/render/math?math=G">

#### Sarsa (Sarsa(0))

- <img src="https://render.githubusercontent.com/render/math?math=Q(S_t, A_t) \leftarrow Q(S_t, A_t) %2B \alpha(R_{t%2B1} %2B \gamma Q(S_{t%2B1}, A_{t%2B1}) - Q(S_t, A_t))"> 

#### Q-Leraning(Sarsamax)

- In the Sarsa, uses <img src="https://render.githubusercontent.com/render/math?math=Q(S_{t%2B1}, A_{t%2B1})"> to estimate future rewards
- In the Q-Learning, uses <img src="https://render.githubusercontent.com/render/math?math=max_{a \in \mathcal{A}} Q(S_{t%2B1}, a)">
- <img src="https://render.githubusercontent.com/render/math?math=Q(S_t, A_t) \leftarrow Q(S_t, A_t) %2B \alpha(R_{t%2B1} %2B \gamma max_{a \in \mathcal{A}} Q(S_{t%2B1}, a) - Q(S_t, A_t))"> 

#### Expected Sarsa

- Using expected rewards (probability of actions x expected rewards), instead of maximum rewards in the Q-learning
- <img src="https://render.githubusercontent.com/render/math?math=Q(S_t, A_t) \leftarrow Q(S_t, A_t) %2B \alpha(R_{t%2B1} %2B \gamma \sum_{a \in \mathcal{A}} \pi (a|S_{t%2B1}) Q(S_{t%2B1}, a) - Q(S_t, A_t))"> 

