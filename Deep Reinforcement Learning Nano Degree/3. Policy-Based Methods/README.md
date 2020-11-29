# 3. Policy-Based Methods

**Official course description**

*In the third part, you'll learn about policy-based and actor-critic methods such as **Proximal Policy Optimization (PPO)**, **Advantage Actor-Critic (A2C)**, and **Deep Deterministic Policy Gradients (DDPG)**.  You’ll also learn about optimization techniques such as **evolution strategies** and **hill climbing**.*



*![Use Deep Deterministic Policy Gradients (DDPG) to train a robot to walk.](https://video.udacity-data.com/topher/2018/June/5b17223a_bipedal-walker/bipedal-walker.gif)*

*Use Deep Deterministic Policy Gradients (DDPG) to train a robot to walk.*

*You'll learn from experts at NVIDIA about the active research that they are  doing, to determine how to apply deep reinforcement learning techniques  to finance.  In particular, you'll explore an algorithm for optimal  execution of portfolio transactions.*

*You'll also get the **second project**, where you'll write an algorithm to train a robotic arm to reach moving target positions.*



*![In Project 2, you will train a robotic arm to reach target locations.](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)*

*In Project 2, you will train a robotic arm to reach target locations.*

## Policy-Based Methods

### Recap of Value-Based Methods

- Value-Based Methods obtain optimal policy by optimal value function based on interaction between agent and environment

- The basic Value-Based Methods has to make Q-table to obtain optimal policy.
- However, in the complicated state like CartPole that states has 4 different continuous values, we cannot identify a specific state to create Q-table.
  - Instead of Q-table, using the neural network to obtain the best action by state → **Deep Q-Learning**

### Why Policy-Based Methods

- Simplicity
  - More simple to estimate the optimal policy without value-function
- Stochastic Policies
  - Unlike value-based methods, policy-based methods can learn true stochastic policies
  - Policy
    - Deterministic: $\pi: s \rightarrow a$
    - Stochastic: $a \sim \pi(s, a) = \mathbb{P}[a|s]$
  - Aliased State
    - Some states are exactly same, but the agent need to act different action
      ![image](https://user-images.githubusercontent.com/8471958/100534077-4077b180-31c0-11eb-8345-23d1c0b93773.png)
- Continuous Action Space
  - Policy-based methods are well-suited for continuous action spaces

### Discrete action space vs. Continuous action space

- In the discrete action space, the nueral network is approximate a stocahstic policy (probability of each action / Softmax as an activation function)
- in the continuous action space, the nueral network returns each action entry (strength of each action entry / Tanh as an activation function: -1 ~ 1)

### Black-Box Optimization

- Gradient Ascent (Hill Climbing)
  - The nueral network for policy-based methods is represented a equation $J(\theta) = \Sigma_{\tau} P(\tau;\theta)R(\tau)$
    - There is $\theta$ to maximize expected return $J$
    - To obtain the optimal policy, adjust $\theta$ toward for high return

- Steepest Ascent Hill Climbing
  - Simulated Annealing: Search some candidates from arbitrary policy $\pi$ and choose the best next policy
  - Adaptive Noise Scaling: Change noisy radius following whether the next policy is better or not
    - If next policy is better, noisy radius decrease
    - If next policy is worse, noisy radius increase (to avoid local maxima)

- Cross-Entropy Method
  - Instead of selecting the best option in the steepest ascent hill climbing, take the average of 10 or 20 percent of policies
- Evolution Strategies
  - Instead of selecting the best option in the steepest ascent hill climbing, take the weighted sum of every policies