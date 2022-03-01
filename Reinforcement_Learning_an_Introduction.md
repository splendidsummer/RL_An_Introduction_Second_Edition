# Reinforcement Learning: An Introduce Notes

## Tabular Solution Method

### Multi-Armed Bandits

#### A k-armed Bandit Problem

k-armed means we can choose k actions when playing with the bandits.
The expected reward or mean of k acitons given the action selected: called value of the action here. 
$$q_*(a) \doteq E[R_t| A_t=a ] $$

##### Action-value Method

Action-value method: 
$$Q(a) \doteq \frac{\sum_{i=1}^{t-1} R_i \cdot  \mathbb{1}_{A_i=a} }{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a} } $$

Greedy Action Selection:

$$A_t \doteq \underset {a}{argmax} Q_t(a)$$

##### Incremental Implementation 
We now turn  to the how these averages can be computed in a computationally efficient manner, in particular, with constant memory and constant per-time-step computation. 

$$Q_{n+1} = \frac {1}{n} \sum_{i=1}^{n} R_i$$
$$Q_{n+1} =  Q_n + \frac {1}{n} [R_n - Q_n]$$
 
Now we get a simply bandit algorithm: 
![iamge](pics/Simple_Bandit_Algorithms.PNG)

#### Tracking Nonstationary Problem

Stationary bandit problems: the reward probabilities do not change over time. But when we are in nonstationary situations, it makes sense to give more weight to recent rewards than to long-past reward. One of most popular ways of doing this is to use a constant step-size parameter. 

$$Q_{n+1} = (1-\alpha) ^ n \cdot   Q_n + \sum_{i=1}^{n} \alpha (1-\alpha)  ^ {n-1} R_i $$

Convergence is not guaranteed for all the choices of the sequence $\{ \alpha_n(a)\}$. If $\alpha$ satisfies the following, the above approximation are assured to converge: 

$\sum_{n=1}^{\infin}\alpha_n(a) = \infin$ and $\sum_{n=1}^{\infin}\alpha_n^2(a)  \lt \infin$ 

#### Optimistic Initial Values
Initial action values can be also be used as a simple way to encourge exploration if we set initial estimate wildly optimistic. Thus this optimmism encourges action-value to explore. 
![effect_optimistic](pics/Effect_of_Optimistic_initial_value.PNG)

#### Upper-Confidence-Bound Action Selection
Exploration is needed because there is always uncertainty about the accuracy of the action-value estimates. $\epsilon$-greedy action selection forces the non-greedy action to be tried, but indiscriminately. It would be better to select among the non-greedy actions according to taking into account both both how close their estimates are to being maximal and the uncertainties in those estimates.  One effective way to do this:
$$A_t \doteq \underset {a} {argmax} s] [Q_t(a) + c \sqrt {\frac{lnt} {N_t(a)}} ]$$ 

where $lnt$ denotes the natural logarithm of t (the number that $e = 2.71828$), $N_t(a)$ denotes the number of times that action a has been selected prior to time t 
.  The number $c> 0$ controls
the degree of exploration. If $N_t(a) = 0$, then a is considered to be a maximizing action. 

The idea of UCB action selection is that the square-root term is a measure of the uncertainty or variance in the estimate of a’s value. The quantity being max’ed over is thus a sort of upper bound on the possible true value of action a,with c determining the confidence level. Each time a is selected the uncertainty is presumably reduced: Nt(a) increments, and, as it appears in the denominator, the uncertainty term decreases. On the other hand, each time an action other than a is selected, t increases but Nt(a) does not; because t appears in the numerator, the uncertainty estimate increases. The use of the natural logarithm means that the increases get smaller over time, but are unbounded; all actions will eventually be selected, but actions with lower value estimates, or that have already been selected frequently, will be selected with decreasing frequency over time.

#### Gradient Bandit Algorithms

Here we consider learning a nummerical preference for each action $a$, which we denote $H_t(a)$. The larger the preference, the more often that action is taken. 

A softmax distribution:
$$Pr\{A_t = a\} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}} \doteq \pi_t(a)$$
$\pi_t(a)$ for the probability of taking action a at time step t.  
There is 
![replace1](pics/replace1.PNG) 


#### Associative Search (Contextual Bandits)

In a general reinforcement learning task there is more than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. Associative search tasks are often now called contextual bandits in the literature. Associative search tasks are intermediate between the k-armed bandit problem and the full reinforcement learning problem. They are like the full reinforcement learning problem in that they involve learning a policy, but like our version of the k-armed bandit problem in that each action a↵ects only the immediate reward. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem. We present this problem in the next chapter and consider its ramifications throughout the rest of the book.

### Finite Markov Desicion Process 

#### The Agent-Environment Interface

![interface](pics/agent-environment_interface.PNG)
* Trajectory 
The MDP and agent together thereby give rise to a sequence or *traject_ory* that begins like this:
$$S_0, A_0, R_1, S_1, A_1, R_2,...$$ 
* Finite MDP
    In a **finite MDP**, the sets of **states, actions, and rewards (S, A, and R) all have a finite number of elements**. In this case, the random variables $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action.
    Dynamics of MDPs: 
    $$p(s', r|s,a ) \doteq Pr\{S_t = s', R_t = r | S_{t-1}=s, A_{t-1} = a\}$$
    The above probabilities satisfy the constraint:
    $$\sum_{s' \in S} \sum{r \in R} p(s', r|s,a )  = 1) {\text{ for all s $\in$ S, a $\in$ A(s)}}$$

* **Markov Property**
  The probability given by p completely characterizes the environment's dynamics. That is, the probability of each possible values for $S_t$ and $R_t$ depends only on the immediately preceding state and action, $S_{t-1}$ and $A_{t-1}$, and not at all on earlier states and actions. 

  This is best viewed a restriction not on the decision process, but on the state. The state must include information about all aspects of the past agent–environment interaction that make a di↵erence for the future. If it does, then the state is said to have the **Markov property**.

* Expected Reward:  for state-action pairs $r: S \times A \rightarrow  \mathbb{R}$
  $$r(s,a) \doteq E[R_t | S_{t-1} = s, A_{t-1} = a] = \sum_{r \in R} r \sum_{s' \in S} p(s', r|s,a ) $$
  There are also other expected rewards by state-action-next-state triples. 
  ![three_argument](pics/three_argument_reward.PNG)

* State-transition probability
  $$p(s'| s, a) \doteq Pr\{S_t = s' | S_{t-1}=s, A_{t-1} = a \} = \sum_{r \in R} p(s', r|s,a ) $$

In particular, **the boundary between agent and environment is typically not the same as the physical boundary of a robot’s or animal’s body**. Usually, the boundary is drawn closer to the agent than that. For example, the motors and mechanical linkages of a robot and its sensing hardware should usually be considered parts of the environment rather than parts of the agent. **The general rule we follow is that anything that cannot be changed arbitrarily by the agent is considered to be outside of it and thus part of its environment.**   
 We do not assume that everything in the environment is unknown to the agent. For example, the agent often knows quite a bit about how its rewards are computed as a function of its actions and the states in which they are taken. But we always consider the reward computation to be external to the agent because it defines the task facing the agent and thus must be beyond its ability to change arbitrarily. In fact, in some cases the agent may know everything about how its environment works and still face a di reinforcement learning task, just as we may know exactly how a puzzle like Rubik’s cube works, but still be unable to solve it.   
 The agent–environment boundary represents the limit of the agent’s absolute control, not of its knowledge. 

#### Goals and Rewards 
In reinforcement learning, **the purpose or goal of the agent is formalized in terms of a special signal, called the reward**, **passing from the environment to the agent**. 

It is critical that the rewards we set up truly indicate what we want accomplished. In particular, **the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do.** The reward signal is your way of communicating to the robot what you want it to achieve, not how you want it achieved.6

#### Returns and Episodes 
The agent's goal is to maximize the cumulative reward it recieves in the long run.
* Episode task: When there is a natural notion of final time step, that is, when the agent–environment interaction breaks naturally into subsequences, which we call **episodes**. Each episode ends in a special state called terminal state ($T$). 
* Continuous task:  In many cases the agent–environment interaction does not break naturally into identifiable episodes, but goes on continually without limit, (which called continuing tasks).
* Return definition:
  $$G_t \doteq R_{t+1} + \gamma \cdot R_{t+2} + \gamma^2 \cdot R_{t+3} + \ldots = \sum_{k=0} ^ {\infin} \gamma^k R_{t+k+1}$$ where $\gamma$ is a parameter, $0 \le \gamma \le 1$, called **discounting rate**. 
  * If $\gamma \lt 1$, the infinite sum in the above formula has a finite value as long as the reward sequence $\{R_k\} $ is bounded. 
  * If the $\gamma = 0$, the agent is 'myopic' in being concerned only with maximizing immediate rewards. If each of the agent's actions happened to influence only the immediate reward, not future rewards as well. 
  * As $\gamma$ approaches to 1, the return objective takes future rewards into account more strongly, the agent becomes more farsighted. 
  * Returns at successive time step:   
    ![return_next](pics/returns_next_step.PNG) 

### Bellman Equations 
#### State-Value Bellman Equation
By recalling the definition of return, we could get the follow equation for state value:
$$
\begin{align}
v_{\pi} = E_{\pi}[G_t | S_t = s]  \\ 
v_{\pi} = \sum_a \pi(a|s) \sum_{r} \sum_{s'} P(s', r | s, a) \cdot [r + \gamma E_{\pi} [G_{t+1} | S_{t+1}  = s']]  \\
v_{\pi} = \sum_a \pi(a|s) \sum_{r} \sum_{s'} P(s', r | s, a) \cdot [r + \gamma \cdot v_{\pi}(s')] 
\end{align} 
$$ 

$$
E_{\pi}[G_{t+1} |S_{t+1} = s'] = \sum_{a'} \pi(a'| s') \sum_{r'} \sum_{s''} p(r', s''| s', a') \cdot [r' +  \gamma E_{\pi} [G_{t+2} | s_{t+2} = s'']
$$ 

So we change a value evaluation problem into solve linear equations. 

#### Action-Value Bellman Equation
$$
\begin{align}
q(s, a) \doteq E_{\pi}[G_t|s_t = s, A_t = a] \\
q(s, a) = \sum_{r} \sum_{s'} p(r, s' | s, a) \cdot (r + \gamma \cdot E_{\pi}[G_{t+1}|S_{t+1} = s']) \\ 
q(s, a) = \sum_{r} \sum_{s'} p(r, s' | s, a) \cdot (r + \gamma \sum_{a'} \pi(a' | s') q_{\pi}(s', a')) \\ 
q(s, a) = \sum_{r} \sum_{s'}  p(r, s' | s, a) \cdot (r + \gamma \cdot ( \sum_{a'} \pi(a'|s') \sum_{r'} \sum_{s''} p(r', s'' | s', a') \cdot ( r' + \gamma (E_{\pi} [G_{t+2} | S_{t+2} = s'', A_{t+1} = a))
\end{align}
$$

### Bellman Optimality Equation

#### Optimal Value Function 
$$ \pi_1  \ge \pi_2 \text{if and only if  } v_{\pi 1} \ge v_{\pi_2} \text{ for all s $ \in$ S }  $$

So we get the best policy by following:

$$v_{\pi_*} (s)\doteq E_{\pi_*}[G_t | S_t 
= s] = \underset{\pi} {max} v_{\pi}(s) \text{ for all s $\in$ S}$$

$$q_{\pi_*} (s, a)  \doteq \underset {\pi} {max} q_{\pi}(s, a) \text{ for all s $\in$ S and a $\in$ A }$$

#### Optimal State Value Function
Firstly we have the above state value function. Then we apply the optimal operation：
$$ v_*(s) = \sum_{a} \pi_* (a|s) \sum_{r} \sum_{s'} p(r, s'| s, a) (r + \gamma \cdot v_*(s'))$$
So if the we are using the above formula in a deterministic condition:
$$v_*(s) = \underset{a}{max} p(r, s'| s, a) (r + \gamma \cdot v_*(s'))$$ 

#### Optimal Action Value Function
Firstly we have action value Bellman equation, if we apply the best policy:
$$ q_*(s, a) = \sum_r \sum_{s'} p(r, s' | s, a) (r + \sum_a \pi_* (a | s) q_* (s', a'))  $$
And if we consider in a deterministic condition:
$$ q_*(s, a) = \sum_r \sum_{s'} p(s', r | s, a) (r + \gamma \cdot \underset{a}{max} q_*(s', a'))$$

**Notice that optimality equations can not be solved by a linear solver since max operation is not linear!!**
![value_diagram1](pics/value_diagram1.PNG)

![value_diagram2](pics/value_diagram2.PNG)

![backup_diagram](pics/backup_diagram.PNG)

Our framing of the reinforcement learning problem forces us to settle for approximations. However, it also presents us with some unique opportunities for achieving useful approximations. For example, in approximating optimal behavior, **there may be many states that the agent faces with such a low probability that selecting suboptimal actions for them has little impact on the amount of reward the agent receives.** Tesauro’s backgammon player, for example, plays with exceptional skill even though it might make very bad decisions on board configurations that never occur in games against experts. In fact, it is possible that TD-Gammon makes bad decisions for a large fraction of the game’s state set. **The online nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less e↵ort for infrequently encountered states.** This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs.

### Dynamic Programming

#### Policy Evaluation
We mainly follow the below formula:   
$$v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a) (r + \gamma v_k(s')) $$

![policy_eval](pics/iterative_policy_evaluation.PNG)

#### Policy Improvement
We could find greedy policy by following: 
![greedy_improve](pics/greedy_improve.PNG)

The greedy policy takes action that looks the best at the short term. (one step ahead according to $v_{\pi}$)

#### Policy Iteration

Once a policy $\pi$ has been improved using $v_{\pi}$ to yield a better policy, $v_{\pi'}$, we can then compute $v_0$ and improve it again to yield an even better. We can thus obtain a sequence of monotonically improving policies and value functions:  
$$ \pi_0 \rightarrow v_0 \rightarrow \pi_1 \rightarrow v_1 \rightarrow \pi_2 \rightarrow \cdots $$

![policy_iteration](pics/policy_iteration.PNG)

#### Value Iteration 

The policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is **when policy evaluation is stopped after just one sweep (one update of each state)**. This algorithm is called **value iteration**. It can be written as **a particularly simple update operation that combines the policy improvement and truncated policy evaluation steps**:

![value_iteration](pics/value_iteration.PNG)   
for all state $s \in S$. For arbitrary $v_0$, the sequence $\{v_k\}$ can be shown to converge to $v_*$ under the same conditions that guarantee the existence of $v_*$. 

![value_iteration_algorithm](pics/value_iteration_algorithm.PNG)

#### Asynchronous Dynamic Programming

A major drawback to the DP methods that we have discussed so far is that they involve operations over the entire state set of the MDP, that is, they require sweeps of the state set. If the state set is very large, then even a single sweep can be prohibitively expensive.



#### Efficiency of Dynamic Programming
 

### Monte-Carlo Methods 

####  Monte Carlo Predictions  

#### Monte Carlo Estimation of Action Value

#### Monte Carlo Control 

#### Monte Carlo Control with Exploring Start 

#### On-policy Vs. Off-policy 


#### Off-policy Prediction via Importance Sampling

#### Incremental Implementation

#### Off-policy Monte Carlo Control 

### Temporal-Differenece Learning 

#### TD Prediction

#### Advantages of TD Prediction Methods

#### Optimality of TD(0) 

#### Sarsa: On-policy TD Control

#### Q_learning: Off-policy TD Control 

#### Expected Sarsa

#### Maximization Bias and Double Q_learning

### n-Steps Boostrapping

#### n-Step TD Prediction 

#### n-Step Sarsa

#### n-Step Off-policy Learning

#### Off-policy Learning Without Importance Sampling: The n-step Tree Backup Algorithm


### Planning and Learning with Tabular Methods 

## Approximate Solution Methods 

### On=policy Prediction with Approximation

### On-policy Control with Approximation 

### Off-policy Control with Approximation 

### Eligibility Trace 

### Policy Gradient Method

## Looking deeper


* $$f(x_1,x_2,\ldots ,x_n) = x_1^2 + x_2^2 + \cdots + x_n^2$$ 



