# 3. Policy-Based Methods

**Official course description**

*In the third part, you'll learn about policy-based and actor-critic methods such as **Proximal Policy Optimization (PPO)**, **Advantage Actor-Critic (A2C)**, and **Deep Deterministic Policy Gradients (DDPG)**.  You’ll also learn about optimization techniques such as **evolution strategies** and **hill climbing**.*



*![Use Deep Deterministic Policy Gradients (DDPG) to train a robot to walk.](https://video.udacity-data.com/topher/2018/June/5b17223a_bipedal-walker/bipedal-walker.gif)*

*Use Deep Deterministic Policy Gradients (DDPG) to train a robot to walk.*

*You'll learn from experts at NVIDIA about the active research that they are  doing, to determine how to apply deep reinforcement learning techniques  to finance.  In particular, you'll explore an algorithm for optimal  execution of portfolio transactions.*

*You'll also get the **second project**, where you'll write an algorithm to train a robotic arm to reach moving target positions.*



*![In Project 2, you will train a robotic arm to reach target locations.](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)*

*In Project 2, you will train a robotic arm to reach target locations.*

[toc]

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
    - Deterministic: <img src="https://render.githubusercontent.com/render/math?math=\pi: s \rightarrow a">
    - Stochastic: <img src="https://render.githubusercontent.com/render/math?math=a \sim \pi(s, a) = \mathbb{P}[a|s]">
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
  - The nueral network for policy-based methods is represented a equation <img src="https://render.githubusercontent.com/render/math?math=J(\theta) = \Sigma_{\tau} P(\tau">;<img src="https://render.githubusercontent.com/render/math?math=\theta)R(\tau)">
    - There is <img src="https://render.githubusercontent.com/render/math?math=\theta"> to maximize expected return <img src="https://render.githubusercontent.com/render/math?math=J">
    - To obtain the optimal policy, adjust <img src="https://render.githubusercontent.com/render/math?math=\theta"> toward for high return

- Steepest Ascent Hill Climbing
  - Simulated Annealing: Search some candidates from arbitrary policy <img src="https://render.githubusercontent.com/render/math?math=\pi"> and choose the best next policy
  - Adaptive Noise Scaling: Change noisy radius following whether the next policy is better or not
    - If next policy is better, noisy radius decrease
    - If next policy is worse, noisy radius increase (to avoid local maxima)

- Cross-Entropy Method
  - Instead of selecting the best option in the steepest ascent hill climbing, take the average of 10 or 20 percent of policies
- Evolution Strategies
  - Instead of selecting the best option in the steepest ascent hill climbing, take the weighted sum of every policies

## Policy Gradient Methods

Policy Gradient Methods is one of the Policy-Based Methods that estimate the weights of an optimal policy through gradient ascent.

- Basic Concept (Big Picture)
  - Loop
    - Collect an episode
    - Change the weigth of the policy network
      - if Won, increase the probability of each (state, action) pair
      - if Lost, decrease the probability of each (state, action) pair

- Trajectory(<img src="https://render.githubusercontent.com/render/math?math=\tau">): state-action sequence
  - <img src="https://render.githubusercontent.com/render/math?math=\tau = (s_0, a_0, s_1, a_1, s_2, \cdots, s_H, a_H, s_{H%2B1})"> 
  - <img src="https://render.githubusercontent.com/render/math?math=R(\tau) = r_1%2Br_2%2B\cdots%2Br_H%2Br_{H%2B1}"> 
  - Goal is to maximize expected return(<img src="https://render.githubusercontent.com/render/math?math=U(\theta)">): <img src="https://render.githubusercontent.com/render/math?math=max_{\theta} U(\theta)">
  - <img src="https://render.githubusercontent.com/render/math?math=U(\theta) = \Sigma_\tau P(\tau">;<img src="https://render.githubusercontent.com/render/math?math=\theta)R(\tau)">
- Gradient Ascent
  - <img src="https://render.githubusercontent.com/render/math?math=\theta \leftarrow \theta %2B \alpha\nabla_\theta U(\theta)"> 
  - However, calculation exact gradient (<img src="https://render.githubusercontent.com/render/math?math=\nabla_\theta U(\theta)">) with every possible trajectory is so expensive
    → Instead of, Estimation the gradient with a few trajectories

## Proximal Policy Optimization

### Main problem of REINFORCE

1. The update process is very **inefficient**! We run the policy once, update once, and then throw away the trajectory.
2. The gradient estimate <img src="https://render.githubusercontent.com/render/math?math=g"> is very **noisy**. By chance the collected trajectory may not be representative of the policy.
3. There is no clear **credit assignment**. A  trajectory may contain many good/bad actions and whether these actions  are reinforced depends only on the final total output.

### Noise Reduction

- The easiest option to reduce the noise in the gradient is to simply  sample more trajectories! Using distributed computing, we can collect  multiple trajectories in parallel, so that it won’t take too much time. 

### Rewards Normalization

- Learning can be improved if we normalize the rewards, where <img src="https://render.githubusercontent.com/render/math?math=\mu"> is the mean, and <img src="https://render.githubusercontent.com/render/math?math=\sigma"> the standard deviation.

### Credit Assignment

- The action at timestamp <img src="https://render.githubusercontent.com/render/math?math=t"> can only affect the future rewards, so the past rewards shouldn't be contributing to the policy gradient
- <img src="https://render.githubusercontent.com/render/math?math=g = \Sigma_t R \nabla_\theta log \pi_\theta(a_t|s_t) = \Sigma_t (R_t^{past} %2B R_t^{future}) \nabla_\theta log \pi_\theta(a_t|s_t)"> → <img src="https://render.githubusercontent.com/render/math?math=g = \Sigma_t R_t^{future} \nabla_\theta log \pi_\theta(a_t|s_t)">

### Importance Sampling

- Reuse the trajectories after update policy(<img src="https://render.githubusercontent.com/render/math?math=\pi_\theta \rightarrow \pi_{\theta^`}">)

- The sampling probability under new policy describes with sampling probability under old policy

### Proximal Policy Optimization (PPO)

- The Surrogate Function
- Clipping Policy Updates
  - To stop update, flatten the surrogate function using the clip the surrogate function within interval <img src="https://render.githubusercontent.com/render/math?math=[1-\epsilon, 1 %2B \epsilon]">
  - We want to clip only the top part
    ![img](https://video.udacity-data.com/topher/2018/September/5b9a9d58_clipped-surrogate-explained/clipped-surrogate-explained.png)

## Actor-Critic Methods

### Motivation and Fundamental

- Actor-Critic Methods is intersection of value-based methods and policy-based methods
  - Value-based Methods approximates value functions (using Deep Neural Networks)
    - <img src="https://render.githubusercontent.com/render/math?math=V_\pi(S)">(Value function), <img src="https://render.githubusercontent.com/render/math?math=Q_\pi(s, a)">(Action-Value function), <img src="https://render.githubusercontent.com/render/math?math=A_\pi(s, a)">(Advantage function)
  - Policy-based Methods approximates a policy (using Deep Neural Networks)
    - Policy-based Methods parameterizes the policy
    - <img src="https://render.githubusercontent.com/render/math?math=\pi(a|s)">(Stochastic; return probability of actions), <img src="https://render.githubusercontent.com/render/math?math=\pi(s)">(Deterministic; return a specific action)
- AC Methods uses value-based technique to further reduce the variance of policy-based methods
- Bias vs. Variance
  - Bias: How far outputs (or center of outputs) from the target
  - Variance: How sparse the outputs

### Estimated Methods

1. Monte-Carlo estimate (MC)
   - High variance and Unbiased
   - The average of rewards by episode (No estimate the future rewards)
2. Temporal-Difference estimate (TD)
   - Low variance and Low bais
   - Sum of current reward and estimated future rewards

### Actor-Critic Methods

- AC Methods consist of two parts (Nueral Networks)
  - Actor: Policy-based Methods
  - Critic: Value-based Methods (TD estimate)
- On-policy vs. Off-policy
  - On-policy(i.e., Salsa): Train with actions following the policy
  - Off-policy(i.e., Q-Learning): Train with actions or others regardless of the policy
- A3C (Asyncronous Advantage Actor-Critic)
  - N-step Bootstrapping
    - Insted of TD estimate, uses N-step Bootstrapping
    - Bootstrapping means that replace future rewards to estimated rewards like TD
    - TD is one-step bootstapping(uses one next rewards and future rewards), MC is infinite-step bootstrapping
  - Parallel training
    - Instead of replay buffer(independent sampling), use parallel training
- A2C (Advantage Actor-Critic)
  - A3C updates the global network asynchronous
  - The each agent in the A3C is A2C
- GAE (Generalized Advantage Estimation)
  - To express the each n-step bootstraping by <img src="https://render.githubusercontent.com/render/math?math=\lambda">
    - One-step bootstrapping: <img src="https://render.githubusercontent.com/render/math?math=1-\lambda">
    - Two-step bootstrapping: <img src="https://render.githubusercontent.com/render/math?math=(1-\lambda)\lambda">
    - Three-step bootstrapping: <img src="https://render.githubusercontent.com/render/math?math=(1-\lambda)\lambda^2">
    - Infinite-step bootstrapping: <img src="https://render.githubusercontent.com/render/math?math=\lambda^{T-t-1}">
  - <img src="https://render.githubusercontent.com/render/math?math=\lambda=0"> means that One-step bootstrapping, <img src="https://render.githubusercontent.com/render/math?math=\lambda=1"> means that Infinite-step bootstrapping
- DDPG (Deep Deterministic Policy Gradient)
  - Solve the issue that DQN cannot apply the environment with continuous action space
  - Soft Updates
    - In the DQN, the target networks weights are updated to the weights of regular(local) networks by every update period
    - In the DDPG, ther target networks weights are updated to mix of target and regular networks weights by everytime
      - For example, if the interpolation parameter(<img src="https://render.githubusercontent.com/render/math?math=\tau">) is 0.01, the target networks weigths are updated mix in 0.01% regular networks weights and 99.99% target networks weights





## References

https://www.slideshare.net/WoongwonLee/trpo-87165690

https://geonhee-lee.github.io/CS294-5.html

https://www.slideshare.net/RLKorea/pg-travel-guide-for-everyone

https://reinforcement-learning-kr.github.io/2018/06/29/0_pg-travel-guide/

https://github.com/reinforcement-learning-kr/pg_travel

https://www.slideshare.net/WoongwonLee/rlcode-a3c