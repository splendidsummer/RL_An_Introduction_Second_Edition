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
![iamge](Simple_Bandit_Algorithms.PNG)

#### Tracking Nonstationary Problem

Stationary bandit problems: the reward probabilities do not change over time. But when we are in nonstationary situations, it makes sense to give more weight to recent rewards than to long-past reward. One of most popular ways of doing this is to use a constant step-size parameter. 

$$Q_{n+1} = (1-\alpha) ^ n \cdot   Q_n + \sum_{i=1}^{n} \alpha (1-\alpha)  ^ {n-1} R_i $$

Convergence is not guaranteed for all the choices of the sequence $\{ \alpha_n(a)\}$. If $\alpha$ satisfies the following, the above approximation are assured to converge: 

$\sum_{n=1}^{\infin}\alpha_n(a) = \infin$ and $\sum_{n=1}^{\infin}\alpha_n^2(a)  \lt \infin$ 

#### Optimistic Initial Values
Initial action values can be also be used as a simple way to encourge exploration if we set initial estimate wildly optimistic. Thus this optimmism encourges action-value to explore. 
![effect_optimistic](Effect_of_Optimistic_initial_value.PNG)

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
![replace1](replace1.PNG) 


#### Associative Search (Contextual Bandits)

In a general reinforcement learning task there is more than one situation, and the goal is to learn a policy: a mapping from situations to the actions that are best in those situations. Associative search tasks are often now called contextual bandits in the literature. Associative search tasks are intermediate between the k-armed bandit problem and the full reinforcement learning problem. They are like the full reinforcement learning problem in that they involve learning a policy, but like our version of the k-armed bandit problem in that each action a↵ects only the immediate reward. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem. We present this problem in the next chapter and consider its ramifications throughout the rest of the book.

### Finite Markov Desicion Process 

#### The Agent-Environment Interface

![interface](agent-environment_interface.PNG)

The MDP and agent together thereby give rise to a sequence or *traject_ory* that begins like this:
$$S_0, A_0, R_1, S_1, A_1, R_2,...$$ 


### Dynamic Programming 

### Monte-Carlo Methods 

### Temporal-Differenece Learning 

### n-Steps Boostrapping 

### Planning and Learning with Tabular Methods 

## Approximate Solution Methods 

### On=policy Prediction with Approximation

### On-policy Control with Approximation 

### Off-policy Control with Approximation 

### Eligibility Trace 

### Policy Gradient Method

## Looking deeper


