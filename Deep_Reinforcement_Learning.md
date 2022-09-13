# Deep Reinforcement Learning
## Value Approximation 
### DQN 
#### Key poinits
* Experience Replay Buffer 
* Semi-gradient 
  
#### Algorithm 
*** 
**Algorithm1 Deep Q-learning with Experience Replay**
*** 
Initialize replay memory $\mathcal{D}$ to capacity N
Initialize action-value function $\mathcal{Q}$ with random weights
**for** episode = 1, M do 
    Initialize sequence $s_1 = \{x_1\} $ and preprocessed sequenced $\phi (s_1)$ 
    **for** t = 1, T **do**

  * with probability $y \in \epsilon$ select a random action $a_t$ 
  * otherwise select $a_t = \underset{a}{max} Q^*(\phi(s_t),a; \theta)$ 
  * Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$ 
  * Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$ 
  * Store transition ($\phi_t$, $a_t$, $r_t$, $\phi_{t+1}$) 
  * Sample random minibatch of transitions ($\phi_j$, $a_j$, $r_j$, $\phi_{j+1}$) from $\mathcal{D}$
  * Set  $$ y_j = \begin{cases} r_j, & \text {for terminal $\phi_{j+1}$} \\ r_j + \gamma \underset{a^\prime}{max} Q(\phi_{j+1}, a^\prime; \theta), & \text{for non-terminal $\phi_{j+1}$} \end{cases} $$
  * Perform a gradient decent step on $(y_j - Q(\phi_j, a_j; \theta))^2$ according to equation1. 
**end for**
**end for**
*** 


<div style="display:none">
这里有个公式要写 
</div>


### Double DQN 


### Duel DQN 
### Distributional Approaches 

## Experience Replay 
#### Replay Buffer

### PER 

### HER 

## Exploration 

## Policy Gradient

### Vanila PG 

### PPO 


## Actor Critic 



<div style="display:none">
$\epsilon$-gree
</div>



