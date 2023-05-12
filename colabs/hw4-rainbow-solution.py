# ^^ [markdown]
# # Sample Efficient Reinforcement Learning - from DQN to (almost) Rainbow
# ### Author: Michal Nauman, Editor: Mateusz Olko
# 
# In this homework we will expand upon on the Deep Q-Network (DQN) algorithm [(Mnih 2014)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). DQN has been successfully applied to a wide range of environments and has demonstrated strong performance on many tasks. However, several challenges and limitations to the DQN that have been identified in the literature:
# 
# 1. Sample complexity - DQN can require a large number of samples to learn effectively, especially in environments with high-dimensional state spaces or a large number of possible actions
# 2. Convergence - DQN is known to converge to the optimal solution under certain conditions, but the convergence properties of the algorithm are not well understood and it is not guaranteed to converge in all cases
# 3. Overestimation - DQN is known to sometimes overestimate the Q-values of certain actions, which can lead to suboptimal behavior
# 4. Sensitivity to hyperparameters - DQN can be sensitive to the choice of hyperparameters, such as the learning rate, the discount factor, and the exploration scheme.
# 
# Rainbow algorithm [(Hessel 2017)](https://arxiv.org/pdf/1710.02298.pdf) is a combination of several techniques for improving the performance of the DQN algorithm, which was originally proposed by DeepMind. By combining several techniques, the Rainbow algorithm is able to improve the sample efficiency, stability and  performance of the DQN algorithm. Overall, the Rainbow algorithm represents an important step forward in the development of reinforcement learning algorithms and is often used as a baseline for implementing more complex changes to the RL setup (e.g. [(Schwarzer 2021)](https://arxiv.org/pdf/2007.05929.pdf) or [(Srinivas 2020)](https://arxiv.org/pdf/2004.04136.pdf))
# 
# In this homework, you will learn to augment a simple DQN implementation with all the components of Rainbow except for distributional Q-learning. To test our implementations, we will use the Lunar Lander environment. Given resources, the environment is easily solved by a vanilla DQN implementation. But we do not have resources. What we have is a budget of:
# 
# 1. 40 000 environment steps
# 2. 35 000 Q-network updates 
# 
# And quite inefficient exploration scheme. As such, our basic DQN implementation will not be enough to solve Lunar Landing problem within the constraints.
# 
# ## Homework scenario and grading
# 
# We provide you with a basic implementation of the DQN. Your job is to expand it with the following modules:
# 
# 1. Double Q-Learning - [(van Hasselt 2015)](https://arxiv.org/pdf/1509.06461.pdf) **1.5 pkt**
# 2. N-step learning - [(Sutton 1988)](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf) **1.5 pkt**
# 3. Noisy linear layers - [(Fortunato et al. 2017)](https://arxiv.org/pdf/1706.10295.pdf) **1.5 pkt**
# 4. Dueling network architecture - [(Wang et al. 2015)](https://arxiv.org/pdf/1511.06581.pdf) **1.5 pkt**
# 5. Prioritized experience replay - [(Schaul et al. 2015)](https://arxiv.org/pdf/1511.05952.pdf) **1.5 pkt**
# 6. (Almost) Rainbow - [(Hessel 2017)](https://arxiv.org/pdf/1710.02298.pdf) **1.5 pkt**
# 
# Each module is designed to work independently (i.e. you can implement each individually with DQN). The final task of this homework is to combine all the implemented modules into (almost) Rainbow agent. You get the last point (**1 pkt**) for plotting results for all implemented parts.

# ^^ [markdown]
# We import the necessary modules:

# ^^
###!pip install gym[box2d]

# %%
import os
import math
import random
import time

import gym
###from google.colab import files
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ^^ [markdown]
# We define a simple class for holding the hyperparameters (do not change those!)

# ^^
# do not change!
class parse_args:
    def __init__(self):
        self.gym_id = "LunarLander-v2"
        self.capacity = 10000
        self.init_steps = 10000
        self.batch_size = 128
        self.hidden_dim = 128
        self.learning_rate = 7e-4
        self.discount = 0.99
        self.samples = 3
        self.total_timesteps = 40000
        self.target_update_freq = 50
        self.evaluate_freq = 1000
        self.evaluate_samples = 5
        self.anneal_steps = 30000
        self.epsilon_limit = 0.01
        self.cuda = True
        env = gym.make(self.gym_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")
        
args = parse_args()

# ^^ [markdown]
# And two helper functions: one for setting seeds, one for simple orthogonal initialization of linear layers, and one for saving and downloading training results.

# ^^
def set_seed_everywhere(env, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    
def weight_init(model):
    if isinstance(model, nn.Linear):
        nn.init.orthogonal_(model.weight.data)
        model.bias.data.fill_(0.0)

def download_numpy(filename, data):
    np.save(filename, data)
    ###files.download(filename)

# ^^ [markdown]
# ## 0. DQN

# ^^ [markdown]
# Deep Q-Network (DQN) [(Mnih 2014)](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) is a reinforcement learning algorithm that uses a deep neural network to learn a Q-function, which is a function that estimates the expected return for taking a given action in a given state. The goal of the DQN algorithm is to learn a policy that maximizes the expected return by learning the Q-function and selecting the action with the highest estimated return in each state.
# 
# The DQN algorithm consists of two main components: a Q-network and an experience buffer. The Q-network is a deep neural network that takes in a state as input and outputs the estimated Q-values for each possible action. The experience buffer is a data structure that stores a set of experiences. The DQN algorithm works by interacting with the environment and storing the experiences in the experience buffer. The Q-network is then trained using a mini-batch of experiences uniformly sampled from the experience buffer. This process is known as experience replay and is used to decorrelate the experiences and to stabilize the learning process. The Q-network is updated using the loss function:
# 
# $$
# \mathcal{L}_{\theta} = \frac{1}{B} \sum_{i=1}^{B} \bigl( \mathrm{TD}~(s_i, a_i, s^{'}_{i}) \bigr)^{2}
# $$
# 
# With:
# 
# $$
# \mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \underset{a^{'}_{i} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i}^{'},a_{i}^{'}) \bigr)
# $$
# 
# Where $Q_{\theta}$ and $\bar{Q}_{\theta}$ denote learned and target Q-networks respectively. The target network is a copy of the Q-network that is updated less frequently, and using it to compute the target Q-values helps to stabilize the learning process and improve the performance of the DQN algorithm. Note that to increase stability of training we use Huber loss (smooth_l1_loss) instead of L2.
# 
# There are several ways to incorporate exploration into the DQN algorithm. One common method is to use an $\epsilon$-greedy exploration strategy, where the agent takes a random action with probability $\epsilon$ and takes the action with the highest estimated Q-value with probability $1 - \epsilon$. The value of $\epsilon$ is typically decreased over time, so that the agent initially explores more and then gradually shifts towards exploitation as it learns more about the environment.
# 
# Below, we implement all the components of a basic DQN. We start with the experience buffer - a data structure that stores a set of transitions, where a transition is typically represented as a tuple $(s, a, r, s', t)$, where $s$ is the state, $a$ is the action taken in state $s$, $r$ is the reward received by performing $a$ in $s$ and getting to $s'$, $s'$ is the new state observed after performing $a$ in $s$ and $t$ is the termination boolean (true if $s'$ is terminal). Experience buffers are used to store the experiences of an agent as it interacts with an environment, and are used to train a Q-function, which is a function that estimates the expected return for taking a given action in a given state. We implement **ExperienceBuffer** class using NumPy arrays and we define two methods:
# 
# 1. *add* - adds transition to the buffer
# 2. *sample* - samples a batch of transitions from the buffer

# ^^
class ExperienceBuffer:
    def __init__(self, args):
        self.states = np.zeros((args.capacity, args.state_dim), dtype=np.float32)
        self.actions = np.zeros((args.capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((args.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((args.capacity, args.state_dim), dtype=np.float32)
        self.terminals = np.zeros((args.capacity, 1), dtype=np.int64)
        self.full = False
        self.idx = 0
        self.args = args 
        
    def add(self, state, action, reward, next_state, terminal):
        self.states[self.idx, :] = state
        self.actions[self.idx, :] = action
        self.rewards[self.idx, :] = reward
        self.next_states[self.idx, :] = next_state
        self.terminals[self.idx, :] = 1 if terminal else 0
        self.idx += 1
        if self.idx == self.args.capacity:
            self.full = True
            self.idx = 0
            
    def sample(self):
        idx = np.random.permutation(self.args.capacity)[:self.args.batch_size] if self.full else np.random.permutation(self.idx-1)[:self.args.batch_size]
        states = torch.from_numpy(self.states[idx]).to(self.args.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.args.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.args.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.args.device)
        terminals = torch.from_numpy(self.terminals[idx]).long().to(self.args.device)
        return states, actions, rewards, next_states, terminals

# ^^ [markdown]
# **QNetwork** class is a simple nn.Module MLP. Note the output size being equal to the amount of actions in the environment.

# ^^
class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
           nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim))
        self.apply(weight_init)
        
    def forward(self, x):
        return self.layers(x)

# ^^ [markdown]
# Finally we implement DQN agent. The class has following methods:
# 
# 1. *get_action* - returns action in given state using $\epsilon$-greedy
# 2. *anneal* - reduces the value of $\epsilon$ dependent on the training step
# 3. *update* - samples a batch of transitions from the experience buffer and performs a DQN update
# 4. *update_target* - performs a hard update on the target Q network $\bar{Q}_{\theta}$
# 5. *evaluate* - performs evaluation of the agent with a greedy policy 
# 6. *reset* - resets the agent (used between seeds)

# ^^
class DQN:
    def __init__(self, args):
        super(DQN, self).__init__()
        self.args = args 
        self.buffer = ExperienceBuffer(self.args)
        self.epsilon = 1
        self.q_net = QNetwork(self.args).to(self.args.device)
        self.q_target = QNetwork(self.args).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)
                
    def get_action(self, state, exploration=True):
        with torch.no_grad():
            return np.random.randint(self.args.action_dim) if np.random.sample() < self.epsilon and exploration else torch.argmax(self.q_net(state)).item()

    def anneal(self, step):
        self.epsilon = ((self.args.epsilon_limit - 1)/self.args.anneal_steps) * step + 1 if step < self.args.anneal_steps else self.epsilon

    def update(self):
        states, actions, rewards, next_states, terminals = self.buffer.sample()
        with torch.no_grad():
            q_ns = torch.max(self.q_target(next_states), dim=1)[0].unsqueeze(1)
        q_targets = rewards + (1-terminals) * self.args.discount * q_ns
        
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.q_target.load_state_dict(self.q_net.state_dict())
        
    def evaluate(self, samples):
        with torch.no_grad():
            env_test = gym.make(self.args.gym_id)
            eval_reward = 0
            for i in range(samples):
                state = env_test.reset()
                episode_reward = 0
                while True:
                    action = self.get_action(torch.tensor(state).unsqueeze(0).to(self.args.device), False)
                    next_state, reward, terminal, _ = env_test.step(action)
                    episode_reward += reward
                    state = next_state
                    if terminal:
                        eval_reward += episode_reward/samples
                        break
        return eval_reward
    
    def reset(self):
        self.buffer = ExperienceBuffer(self.args)
        self.epsilon = 1
        self.q_net = QNetwork(self.args).to(self.args.device)
        self.q_target = QNetwork(self.args).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)

# ^^ [markdown]
# Finally, we provide code for agent training:

# ^^
def train_agent(args, agent):
    results = np.zeros((args.total_timesteps//args.evaluate_freq, args.samples))
    for seed in range(args.samples):
        env = gym.make(args.gym_id)
        agent.reset()
        set_seed_everywhere(env, seed)
        state = env.reset()
        for step in range(args.total_timesteps):
            if step == args.init_steps:
                start_time = time.time()
            action = agent.get_action(torch.tensor(state).unsqueeze(0).to(args.device))
            next_state, reward, terminal, _ = env.step(action)
            agent.buffer.add(state, action, reward, next_state, terminal)
            agent.anneal(step)
            state = next_state
            if step >= args.init_steps:
                agent.update()
                if (step + 1) % args.target_update_freq == 0:
                    agent.update_target()
                if (step + 1) % args.evaluate_freq == 0:
                    eval_reward = agent.evaluate(args.evaluate_samples)
                    results[step//args.evaluate_freq, seed] = eval_reward
                    print("\rStep: {} Evaluation reward: {:.2f} Samples per second: {:}      ".format(step, eval_reward, int((step-args.init_steps)/(time.time()-start_time))), end="")
            if terminal:
                state = env.reset()
                episode_reward = 0
    return results

# ^^ [markdown]
# Note that you should not change the code above - you should be able to perform all tasks by creating new classes. We train the DQN agent with given hyperparameters and inspect the results:

# ^^
#agent = DQN(args)
#results_dqn = train_agent(args, agent)
#download_numpy("results_dqn.npy", results_dqn)
#results_dqn.mean(1)[10:].mean()
#results_dqn.mean(1)

# ^^ [markdown]
# As you can see, the vanilla DQN does not yield optimal performance given the budget and exploration constraints. Below is the first module that you have to add to the DQN algorithm.

# ^^ [markdown]
# ## 1. Double DQN
# 
# The loss function of vanilla DQN is defined as the average of single transition temporal difference (TD) error over $B$ transitions:
# 
# $$
# \mathcal{L}_{\theta} = \frac{1}{B} \sum_{i=1}^{B} \bigl( \mathrm{TD}~(s_i, a_i, s^{'}_{i}) \bigr)^{2}
# $$
# 
# With transitions $(s_i, a_i, s^{'}_{i})$ sampled uniformly from the experience buffer. The transition TD error is defined through Bellman optimality condition:
# 
# $$
# \mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \underset{a^{'}_{i} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i}^{'},a_{i}^{'}) \bigr)
# $$
# 
# Where $Q_{\theta}$ and $\bar{Q}_{\theta}$ denote learned and target Q-networks respectively. In the setup above $a_{i}^{'}$ is chosen via maximum operation over the output of the target Q-network for $s^{'}_{i}$. Using a single network to choose the best action and estimate its Q-value promotes overestimated values. Using such values for supervision leads in turn to general overoptimism of the Q-network and is known to sabotage the training.
# 
# In Double Deep Q-Network (DDQN) [(van Hasselt 2015)](https://arxiv.org/pdf/1509.06461.pdf) proposes using two Q-networks in the process of target estimation: one Q-network to choose the maximum valued action from (i.e. *argmax*); and the second one to estimate value of the chosen action (i.e. Q-value estimation for the *argmax* result). Authors show that in DDQN estimated Q-values are less likely to be inflated and lead to more stable learning and better policies. We can use $Q_{\theta}$ and $\bar{Q}_{\theta}$ to augment DQN into DDQN: 
# 
# $$
# \mathrm{TD}~(s_i, a_i, s^{'}_{i}) = Q_{\theta}~(s_i,a_i) - \bigl(r_{(s_i,a_i,s_{i}^{'})} + \gamma ~ \bar{Q}_{\theta}~(s_{i}^{'},\underset{a^{'}_{i} \sim Q_{\theta}}{\mathrm{argmax}} ~ Q_{\theta} (s_{i}^{'}, a^{'}_{i})  \bigr)
# $$
# 
# Such definition of DDQN leads to very small code changes w.r.t. vanilla DQN implementation. Although $Q_{\theta}$ and $\bar{Q}_{\theta}$ are not fully decoupled, using them leads to good performance increase without introduction of additional networks.

# ^^ [markdown]
# ### Task 1.1: Implement and train DDQN 
# Implement the *update* method for **DDQN** class (no other method of the base class should be changed): 

# ^^
class DDQN(DQN):
    def __init__(self, args):
        super(DDQN, self).__init__(args)
        
    def update(self):
        states, actions, rewards, next_states, terminals = self.buffer.sample()
        ###############
        with torch.no_grad():
            action_idxs = torch.argmax(self.q_net(next_states), dim=1)            # dim=(batch_size)
            q_ns = self.q_target(next_states).gather(1, action_idxs.unsqueeze(1)) # dim=(batch_size, 1)
        q_targets = rewards + (1-terminals) * self.args.discount * q_ns
        ###############
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        loss.backward()
        self.optimizer.step()

# ^^
#agent = DDQN(args)
#results_dqn1 = train_agent(args, agent)
#download_numpy("results_dqn1.npy", results_dqn1)
#results_dqn1.mean(1)[-10:].mean()
#results_dqn1.mean(1)

# ^^ [markdown]
# ## 2. $\mathrm{TD}_{n}$ - N-step Q-value estimation
# 
# $N$-step TD ($\mathrm{TD}_{n}$) was introduced long before neural network based RL. In regular TD, we supervise the Q-network with single-step reward summed with highest Q-value of the next state. In contrast to that, $\mathrm{TD}_{n}$ accumulated rewards over $n$ steps and sums it with the highest Q-value of the state that occured after $n$ steps [(Sutton 1988)](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf). Double DQN $\mathrm{TD}_{n}$ loss is defined by:
# 
# $$
# \mathrm{TD}_{n}(s_i, a_i, s^{'}_{i+n}) = Q_{\theta}~(s_i,a_i) - \biggl(\sum_{k=0}^{n-1} \gamma^{k} ~ r_{(s_{i+k},a_{i+k},s_{i+k}^{'})} + \gamma^{n} \underset{a^{'}_{i+n} \sim \bar{Q}_{\theta}}{\mathrm{max}} ~ \bar{Q}_{\theta}~(s_{i+n}^{'},a_{i+n}^{'}) \biggr)
# $$
# 
# Implementing $\mathrm{TD}_{n}$ requires changes to the ExperienceBuffer class. We will implement those changes using the **deque** module. This module will store $n$ of the most recent transitions, and will act as an intermediate between agent and buffers main storage. As compared to single step reward and $s_{i}^{'}$ stored by the simple ExperienceBuffer, the main storage of this upgraded buffer should store $n$ step rewards and $s_{i+n}^{'}$.

# ^^ [markdown]
# ### Task 2.1 Implement NStepBuffer
# Implement *get_nstep* method for **NStepBuffer** class (no other method of base class should be changed). The *get_nstep* method should process current memory and output a tuple of five:
# * state for which the $\mathrm{TD}_{n}$ reward was computed,
# * action chosen in that step in processed trajectory,
# * $\mathrm{TD}_{n}$ reward computed using *nstep* rewards,
# * state reached after *nstep* steps (possibly earlier if terminal state was encountered),
# * terminal flag, that notifies wheather trajectory has reached terminal state within *nstep* steps.

# ^^
from collections import deque

class NStepBuffer(ExperienceBuffer):
    def __init__(self, args, nstep):
        super(NStepBuffer, self).__init__(args)
        self.memories = deque(maxlen=nstep)
        self.nstep = nstep 
        
    def add(self, state, action, reward, next_state, terminal):
        terminal_ = 1 if terminal else 0 
        memory = (state, action, reward, next_state, terminal_)
        self.memories.append(memory)
        if len(self.memories) >= self.nstep:
            state, action, reward, next_state, terminal = self.get_nstep()
            self.states[self.idx, :] = state
            self.actions[self.idx, :] = action
            self.rewards[self.idx, :] = reward
            self.next_states[self.idx, :] = next_state
            self.terminals[self.idx, :] = terminal
            self.idx += 1
            if self.idx == self.args.capacity:
                self.full = True
                self.idx = 0
            
    def get_nstep(self):
        ###############
        terminal_idxs_ = [i for i in range(len(self.memories)) if self.memories[i][4] == 1]
        reached_idx_ = self.nstep - 1 if not terminal_idxs_ else terminal_idxs_[0]

        state, action, _, _, _ = self.memories[0]
        reward = sum([m[2]*(self.args.discount**k) for k, m in enumerate(list(self.memories)[:reached_idx_+1])]) 
        _, _, _, next_state, terminal = self.memories[reached_idx_]

        #self.memories.popleft()
        ###############
        return state, action, reward, next_state, terminal

# ^^ [markdown]
# ### Task 2.2: Implement and train N-step DQN 
# Implement the *update* method for **NStepDQN** class (no other method of base class should be changed): 

# ^^
class NStepDQN(DQN):
    def __init__(self, args, nstep=3):
        super(NStepDQN, self).__init__(args)
        self.nstep = nstep
        self.buffer = NStepBuffer(args, nstep)
        
    def update(self):
        states, actions, rewards, next_states, terminals = self.buffer.sample()
        ###############
        with torch.no_grad():
            q_ns = torch.max(self.q_target(next_states), dim=1)[0].unsqueeze(1)
        q_targets = rewards + (1-terminals) * (self.args.discount ** self.nstep) * q_ns
        ###############
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        loss.backward()
        self.optimizer.step()
        
    def reset(self):
        super().reset()
        self.buffer = NStepBuffer(self.args, self.nstep)

# ^^
#agent = NStepDQN(args)
#results_dqn2 = train_agent(args, agent)
#download_numpy("results_dqn2.npy", results_dqn2)
#results_dqn2.mean(1)[-10:].mean()

# ^^ [markdown]
# ## 3. Noisy Layer Exploration
# 
# $\epsilon$-greedy exploration is not well suited for environments that require complex sequences of actions to achieve success. $\epsilon$ value must be set manually, and finding a good value can be difficult and costly. A value that is too high will result in too much exploration and slow down learning, while a value that is too low will not allow the agent to gather enough information about the environment. Now, we will introduce a different method for exploration.
# 
# Noisy linear layer, is a type of layer that can be added to a neural network [(Fortunato et al. 2017)](https://arxiv.org/pdf/1706.10295.pdf). These layers add a learned noise to the parameters of the network, which adds stochasticity to the network output. Noisy parameters can induce complex multi-step changes in estimated Q-values and the policy. Noisy linear layers can be more effective than $\epsilon$-greedy in environments with sparse rewards or long-term dependencies, but they can also be less sample-efficient than well tuned $\epsilon$-greedy strategy in simpler settings. 
# 
# Regular linear layer has $pq + q$ parameters, where $p$ and $q$ denote number of inputs and outputs in the layer. Denoting weight matrix as $W \in \mathbb{R}^{q \times p}$, bias vector as $B \in \mathbb{R}^q$ and layer input as $X \in \mathbb{R}^p$, linear layer performs:
# 
# $$
# Y = W X + B
# $$
# 
# In contrast to that, noisy linear layer is defined as:
# 
# $$
# Y = \bigl( \mu^W + \sigma^W \odot \epsilon^W \bigr) X + \bigl( \mu^B + \sigma^B \odot \epsilon^B \bigr)
# $$
# 
# where $\mu^W + \sigma^W \odot \epsilon^W$ and $\mu^B + \sigma^B \odot \epsilon^B$ replace $W$ and $B$ in the first linear layer equation. The parameters $\mu^W \in \mathbb{R}^{q \times p}, \mu^B \in \mathbb{R}^q, \sigma^W \in \mathbb{R}^{q \times p}$ and $\sigma^B \in \mathbb{R}^q$ are learnt jointly via the single Q-network loss; $\epsilon^W \in \mathbb{R}^{q \times p}$ and $\epsilon^B \in \mathbb{R}^q$ is the random noise. In principle, the random noise can be generated following any distribution, but the authors consider two strategies:
# 
# 1. **Independent Gaussian noise** - We generate each noise entry independently. As such, we perform $pq + q$ calls to the Gaussian noise generator. Simple, but can be costly for big networks
# 
# 2. **Factorised Gaussian noise** - This is a more computationally efficient way that authors use in the original paper. Instead of generating $pq + q$ entries independently, we generate two noise vectors: $\epsilon^{p}, \epsilon^{B} \sim N(0, 1)$. Then, entries to $\epsilon^W$ are given by:
# 
# $$
# \epsilon^{W}_{i,j} = f(\epsilon^{p}_{i}) f(\epsilon^{B}_{j}) \quad \text{with} \quad f(x) = sgn(x) \sqrt{|x|}.
# $$

# ^^ [markdown]
# ### Task 3.1 Implement NoisyLinear layer 
# Implement the **NoisyLinear** class. The parameters of the noisy linear layer should be initialized with a correct initialization scheme (see section 3.2 in [Fortunato et al. 2017](https://arxiv.org/pdf/1706.10295.pdf)). The class should have the following methods:
# 
# 1. *get_noise* - the method should generate $\epsilon^{W}$ and $\epsilon^{B}$ using the factorised Gaussian noise procedure
# 2. *forward* - generate noise and perform a forward pass

# ^^
class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, std):
        super(NoisyLinear, self).__init__()
        self.w_mu = nn.Parameter(torch.Tensor(output_size, input_size))
        ###############
        self.w_sigma = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b_mu = nn.Parameter(torch.Tensor(output_size))
        self.b_sigma = nn.Parameter(torch.Tensor(output_size))
        
        self.p = input_size
        self.q = output_size

        nn.init.constant_(self.w_sigma.data, std * (input_size ** -.5))
        nn.init.constant_(self.b_sigma.data, std * (input_size ** -.5))
        nn.init.uniform_(self.w_mu.data, -(input_size ** -.5), input_size ** -.5)
        nn.init.uniform_(self.b_mu.data, -(input_size ** -.5), input_size ** -.5)
        ###############

    def get_device(self):
        return self.w_mu.device

    def get_noise(self):
        ###############
        f = lambda x : torch.sign(x) * torch.sqrt(torch.abs(x))
        b_noise = torch.empty(self.q, device=self.get_device())
        nn.init.normal_(b_noise)
        p_noise = torch.empty(self.p, device=self.get_device())
        nn.init.normal_(p_noise)
        w_noise = torch.outer(f(b_noise), f(p_noise))
        ###############
        return w_noise, b_noise

    def forward(self, x):
        ###############
        # x.dim = (batch_size, p)
        w_noise, b_noise = self.get_noise()
        # w_noise.dim = (q, p), b_noise.dim = (q, 1)
        # w_sigma.dim = (q, p), w_mu.dim = (q, p)

        ##x = torch.mm(x, (self.w_sigma * w_noise + self.w_mu).T)
        # x.dim = (batch_size, q)
        # b_sigma.dim = (q, 1), b_mu.dim = (q, 1)
        ##x = x + (self.b_sigma * b_noise + self.b_mu).T
        # x.dim = (batch_size, q)
        t1 = self.w_sigma * w_noise + self.w_mu
        t2 = self.b_sigma * b_noise + self.b_mu
        return torch.nn.functional.linear(x, t1, t2)
        ###############


class NoisyQNetwork(nn.Module):
    def __init__(self, args, std):
        super(NoisyQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            NoisyLinear(args.hidden_dim, args.hidden_dim, std), nn.ReLU(),
            NoisyLinear(args.hidden_dim, args.action_dim, std))
        
    def forward(self, x):
        return self.layers(x)

# ^^ [markdown]
# ### Task 3.2 Train NoisyDQN with NoisyLayers 
# 

# ^^
class NoisyDQN(DQN):
    def __init__(self, args, std=0.2):
        super(NoisyDQN, self).__init__(args)
        self.q_net = NoisyQNetwork(args, std).to(args.device)
        self.q_target = NoisyQNetwork(args, std).to(args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate, eps=1e-5)
        self.std = std
        
    def get_action(self, state, exploration=True):
        return torch.argmax(self.q_net(state)).item()

    def anneal(self, step):
        pass
    
    def reset(self):
        super().reset()
        self.q_net = NoisyQNetwork(self.args, self.std).to(self.args.device)
        self.q_target = NoisyQNetwork(self.args, self.std).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)

# ^^
#agent = NoisyDQN(args)
#results_dqn3 = train_agent(args, agent)
#download_numpy("results_dqn3.npy", results_dqn3)
#results_dqn3.mean(1)[-10:].mean()

# ^^ [markdown]
# ## 4. DUELING DQN
# 
# State-action advantage under policy $\pi$ is given by:
# 
# $$
# A^\pi (s, a) = Q^\pi (s, a) - V^\pi (s)
# $$
# 
# Where $A^\pi (s, a)$ denotes state-action advantage, $Q^\pi (s, a)$ denotes state-action Q-value and $V^\pi (s)$ denotes state value. Advantage is a measure of how much better a particular action is than the state value. Given optimal policy it follows that $\underset{a}{\mathrm{max}}~Q^\pi (s, a) = V^\pi (s)$ and as such $A^\pi (s, a) \leq 0$ if $\pi$ is optimal. We can use advantages to redefine Q-values:
# 
# $$
# Q^\pi (s, a) = V^\pi (s) + A^\pi (s, a)
# $$
# 
# As such, we can use separate networks to predict $A^\pi (s, a)$ and $V^\pi (s)$ and retrieve Q-values using the equation above. This is exactly the idea behind the Dueling Q-network architecture [(Wang et al. 2015)](https://arxiv.org/pdf/1511.06581.pdf). Decoupling Q-values into values and advantages offers some optimization benefits:
# 
# 1. $V^\pi (s)$ is independent of actions, as such the value network will have less parameters than a Q-network
# 2. $A^\pi (s, a)$ although action dependent, advantages oscillate around 0 and change slowly throughout the optimization
# 
# Intuitively, the dueling Q-network can more efficiently learn which states are valuable, even when the actions available in those states do not affect the environment in a meaningful way. This can be particularly helpful in large or complex environments where it may not be possible to learn good action values for every state-action pair. Dueling DQN architecture uses joint feature layer and two separate heads to represent advantage and value streams (look at Figure 1. in [(Wang et al. 2015)](https://arxiv.org/pdf/1511.06581.pdf)). To further smoothen the optimization, Dueling DQN Q-value is calculated with the following:
# 
# $$
# Q_\theta (s, a) = V_\theta (s) + \bigl( A_\theta (s, a) - \sum_{a} \frac{A_\theta (s, a)}{N_a} \bigr),
# $$
# 
# where $N_a$ is the number of possible actions.

# ^^ [markdown]
# ### Task 4.1 Implement and train DuelingQNetwork (10% points)
# Implement the **DuelingQNetwork** class and its *forward* method (no other method of the base class should be changed):

# ^^
class DuelingQNetwork(nn.Module):
    def __init__(self, args):
        super(DuelingQNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(),)
        self.advantage_head = nn.Linear(args.hidden_dim, args.action_dim)
        self.value_head = nn.Linear(args.hidden_dim, 1)
        
    def forward(self, x):
        ################
        interim_result = self.layers(x)
        advantage_result = self.advantage_head(interim_result)
        return self.value_head(interim_result) + advantage_result - torch.mean(advantage_result, dim=-1).unsqueeze(-1)
        ################
    
class DuelingDQN(DQN):
    def __init__(self, args):
        super(DuelingDQN, self).__init__(args)
        self.q_net = DuelingQNetwork(args).to(args.device)
        self.q_target = DuelingQNetwork(args).to(args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate, eps=1e-5)
        
    def reset(self):
        super().reset()
        self.q_net = DuelingQNetwork(self.args).to(self.args.device)
        self.q_target = DuelingQNetwork(self.args).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)

# ^^
#agent = DuelingDQN(args)
#results_dqn4 = train_agent(args, agent)
#download_numpy("results_dqn4", results_dqn4)
#results_dqn4.mean(1)[-10:].mean()

# ^^ [markdown]
# ## 5. Prioritized experience replay
# 
# In regular experience replay the transitions are uniformly sampled during training and used to update the agent's learning policy. Prioritized replay [(Schaul et al. 2015)](https://arxiv.org/pdf/1511.05952.pdf) is a variant of the experience replay buffer that prioritizes transitions based on the magnitude of the TD error, which is a measure of how much the agent's estimates of the action values deviate from the actual values. Transitions with a higher TD error are more important for learning, because they represent a greater deviation from the agent's current understanding of the environment. By prioritizing transitions with a higher TD error, the agent can more effectively learn from its experiences and improve its performance.
# 
# To implement a prioritized replay buffer, we need to store not only the transitions themselves, but also the TD error for each transition. We will use a queue-like mechanism to prioritize the transitions based on their TD error, and sample transitions from the buffer using a priority-based sampling distribution. New transitions arrive with maximal priority in order to guarantee that all experience is seen at least once. This allows the agent to more effectively learn from rare or unusual transitions that might be overlooked in a standard experience replay buffer.
# 
# There are two ways to prioritize transitions in the experience replay buffer based on the TD error: greedy prioritization and stochastic prioritization. When using greedy prioritization, the transitions with the highest TD errors are replayed more frequently, which can lead to overfitting and overlooking certain transitions. To address this issue, we will use a stochastic prioritization method that balances between greedy prioritization and uniform random sampling introducing more diversity in the sampled transitions.
# 
# $$
# P(i) = \frac{p_i^{\alpha} + \epsilon}{\sum_{j=1}^{D} (p_j^{\alpha} + \epsilon)}
# $$
# 
# Where $p_i > 0$ denotes the priority of transition $i$ with $i, j \in D$, $D$ denotes the buffer data, $\epsilon$ is a small positive constant and the exponent $\alpha$ determines how much prioritization is used, with $\alpha = 0$ corresponding to the uniform sampling. 
# 
# The estimation of the expected value with stochastic updates relies on those updates corresponding to the same distribution as its expectation. Prioritized replay introduces bias because it changes this distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will converge to (even if the policy and state distribution are fixed). To compensate for non-uniform sampling scheme, prioritized experience replay leverages one more mechanism - importance sampling. To this end, when calculating loss over the batch transition TD error is weighted with an importance weight:
# 
# 
# $$
# \mathcal{L}_{\theta} = \frac{1}{B} \sum_{i=1}^{B} \bigl( w_i * \mathrm{TD}~(s_i, a_i, s^{'}_{i}) \bigr)^{2}
# $$
# 
# Where:
# 
# $$
# w_i = \big( \frac{1}{D} \cdot \frac{1}{P(i)} \big)^\beta
# $$
# 
# Which given $\beta = 1$ fully compensates for the non uniform sampling. We will anneal values of $\alpha$ and $\beta$ towards 1 throughout the training.
# 
# Managing priorities is often implemented via a Segment Tree. It allows us to be very efficient in sampling transitions, while creating a bit of overhead in writing new values. **You are not required to use segment tree, and no points will be subtracted for not using it. However we encourage to do it for better efficiency and shorter training time.** We provide you with the OpenAI implementation of a SegmentTree below. We recommend that you read a bit about segment trees before moving forward:
# 
# 1. https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/
# 2. https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
# 
# **Note that the priority queue will be much slower even with good implementation. This slow down is much less visible in image-based RL, where other parts of the compute pipeline are greatly more demanding.**

# ^^
###!wget https://raw.githubusercontent.com/openai/baselines/master/baselines/common/segment_tree.py
from segment_tree import MinSegmentTree, SumSegmentTree

# ^^ [markdown]
# ### Task 5.1 Implement PrioritizedBuffer 
# Implement the **PrioritizedBuffer** class and the following methods:
# 
# 1. *add* - it should also manage the priorities in the trees
# 2. *sample* - it should samples according to priorities and return importance weights
# 3. *update_priorities* - it should update priorities in trees after performing DQN update
# 4. *get_idx* - it should sample indices according to probability ditribution
# 5. *calculate_weights* - it should calculate importance weights for given index
# 
# No other method of the base class should be changed. 

# ^^
class PrioritizedBuffer(ExperienceBuffer):    
    def __init__(self, args, alpha, beta):
        super(PrioritizedBuffer, self).__init__(args)
        tree_capacity = 1
        while tree_capacity < self.args.capacity:
            tree_capacity *= 2
        self.beta = beta 
        self.alpha = alpha
        ################
        self.p_values = np.zeros((args.capacity, 1), dtype=np.float32)
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0
        self.epsilon = 1e-8
        ################
        
    def add(self, state, action, reward, next_state, terminal):
        ################
        self.p_values[self.idx] = self.max_priority ** self.alpha + self.epsilon
        self.sum_tree[self.idx] = self.max_priority ** self.alpha + self.epsilon
        self.min_tree[self.idx] = self.max_priority ** self.alpha + self.epsilon
        super().add(state, action, reward, next_state, terminal)
        ################
                
    def sample(self):
        ################
        idx = self.get_idx()
        
        elem_count = self.args.capacity if self.full else self.idx
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * elem_count) ** (-self.beta)
        weights = self.calculate_weight(idx) / max_weight
        
        states      = torch.from_numpy(self.states[idx]).to(self.args.device)
        actions     = torch.from_numpy(self.actions[idx]).to(self.args.device)
        rewards     = torch.from_numpy(self.rewards[idx]).to(self.args.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.args.device)
        terminals   = torch.from_numpy(self.terminals[idx]).to(self.args.device)
        weights     = torch.from_numpy(weights).to(self.args.device)
        ################
        return states, actions, rewards, next_states, terminals, idx, weights
    
    def update_priorities(self, idx, priorities):
        ################
        #self.p_values[idx] = np.expand_dims(np.array(priorities), axis=1)
        for i, priority in zip(idx, priorities):
            self.p_values[i] = priority ** self.alpha + self.epsilon
            self.sum_tree[i] = priority ** self.alpha + self.epsilon
            self.min_tree[i] = priority ** self.alpha + self.epsilon
            self.max_priority = max(self.max_priority, priority)
        ################
                
    def get_idx(self):
        ################
        elem_count = self.args.capacity if self.full else self.idx
        p_total = self.sum_tree.sum(0, elem_count-1)
        every_range_len = p_total / self.args.batch_size
        randomness = np.random.rand(self.args.batch_size)
        idxs = np.empty(self.args.batch_size, dtype=np.int64)
        for i in range(self.args.batch_size):
            mass = randomness[i] * every_range_len + i * every_range_len
            idxs[i] = self.sum_tree.find_prefixsum_idx(mass)
        ################
        return idxs
    
    def calculate_weight(self, idx):
        ################
        elem_count = self.args.capacity if self.full else self.idx
        p_sample = self.p_values[idx] / self.sum_tree.sum()
        #p_sample = np.squeeze(p_sample, axis=-1)
        weight = (p_sample * elem_count) ** (-self.beta)
        ################
        return weight

# ^^ [markdown]
# ### Task 5.2 Train PrioritizedDQN 
# Implement the **PrioritizedDQN** class and its *update* method (no other method of the base class should be changed):

# ^^
class PrioritizedDQN(DQN):
    def __init__(self, args, alpha=0.0, beta=0.0): # CHANGED!!!
        super(PrioritizedDQN, self).__init__(args)
        self.buffer = PrioritizedBuffer(args, alpha, beta)
        self.alpha = alpha
        self.beta = beta
        
    def update(self):
        states, actions, rewards, next_states, terminals, idx, weights = self.buffer.sample()
        with torch.no_grad():
            q_ns = torch.max(self.q_target(next_states), dim=1)[0].unsqueeze(1)
        q_targets = rewards + (1-terminals) * self.args.discount * q_ns
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        td_errors = nn.functional.smooth_l1_loss(q_values, q_targets, reduction='none')
        loss = torch.mean(td_errors * weights)
        loss.backward()
        self.optimizer.step()
        priorities = td_errors.detach().squeeze().cpu().tolist()
        self.buffer.update_priorities(idx, priorities)
        
    def anneal(self, step):
        super().anneal(step)
        if step < self.args.anneal_steps:
            self.buffer.alpha = ((1 - self.alpha)/self.args.anneal_steps)*step + self.alpha
            self.buffer.beta = ((1 - self.beta)/self.args.anneal_steps)*step + self.beta
        else:
            pass

    def reset(self):
        super().reset()
        self.buffer = PrioritizedBuffer(self.args, self.alpha, self.beta)  

# ^^
#agent = PrioritizedDQN(args)
#results_dqn5 = train_agent(args, agent)
#download_numpy("results_dqn5.npy", results_dqn5)
#results_dqn5.mean(1)[-10:].mean()

# ^^ [markdown]
# ## 6. (Almost) Rainbow
# The final thing we are left with is to combine all the improvements into a single agent [(Hessel 2017)](https://arxiv.org/pdf/1710.02298.pdf). To this end, you have to implement three classes:
# 
# 1. **RainbowBuffer** - experience buffer that combines nstep returns and priority-based sampling
# 2. **RainbowQNetwork** - Q-network that uses noisy linear layers in a dueling setup
# 3. **RainbowDQN** - DQN that combines all of the covered techniques
# 
# ![fig1](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-07_at_9.14.13_PM_4fMCutg.png)

# ^^ [markdown]
# ### Task 6.1 Implement RainbowBuffer

# ^^
from collections import deque 

class RainbowBuffer(ExperienceBuffer):
    def __init__(self, args, nstep, alpha, beta):
        super(RainbowBuffer, self).__init__(args)
        tree_capacity = 1
        while tree_capacity < self.args.capacity:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.priority_cap = 1
        self.alpha = alpha
        self.beta = beta   
        self.memories = deque(maxlen=nstep)
        self.nstep = nstep 
        
    def add(self, state, action, reward, next_state, terminal):
        ################
        terminal_ = 1 if terminal else 0 
        memory = (state, action, reward, next_state, terminal_)
        self.memories.append(memory)
        if len(self.memories) >= self.nstep:
            self.sum_tree[self.idx] = self.priority_cap ** self.alpha + 1e-8
            self.min_tree[self.idx] = self.priority_cap ** self.alpha + 1e-8

            terminal_idxs_ = [i for i in range(len(self.memories)) if self.memories[i][4] == 1]
            reached_idx_ = self.nstep - 1 if not terminal_idxs_ else terminal_idxs_[0]
            state, action, _, _, _ = self.memories[0]
            reward = sum([m[2]*(self.args.discount**k) for k, m in enumerate(list(self.memories)[:reached_idx_+1])]) 
            _, _, _, next_state, terminal = self.memories[reached_idx_]

            self.states[self.idx, :] = state
            self.actions[self.idx, :] = action
            self.rewards[self.idx, :] = reward
            self.next_states[self.idx, :] = next_state
            self.terminals[self.idx, :] = terminal
            self.idx += 1
            if self.idx == self.args.capacity:
                self.full = True
                self.idx = 0
        ################
                
    def sample(self):
        ################
        elem_count = self.args.capacity if self.full else self.idx
        p_total = self.sum_tree.sum(0, elem_count-1)
        every_range_len = p_total / self.args.batch_size
        randomness = np.random.rand(self.args.batch_size)
        idx = np.empty(self.args.batch_size, dtype=np.int64)
        for i in range(self.args.batch_size):
            mass = randomness[i] * every_range_len + i * every_range_len
            idx[i] = self.sum_tree.find_prefixsum_idx(mass)
        
        elem_count = self.args.capacity if self.full else self.idx
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * elem_count) ** (-self.beta)
        p_sample = np.empty(self.args.batch_size)
        for i in range(self.args.batch_size):
            p_sample[i] = self.sum_tree[idx[i]] / self.sum_tree.sum()
        weight = (p_sample * elem_count) ** (-self.beta)
        weights = weight / max_weight
        
        states      = torch.from_numpy(self.states[idx]).to(self.args.device)
        actions     = torch.from_numpy(self.actions[idx]).to(self.args.device)
        rewards     = torch.from_numpy(self.rewards[idx]).to(self.args.device)
        next_states = torch.from_numpy(self.next_states[idx]).to(self.args.device)
        terminals   = torch.from_numpy(self.terminals[idx]).to(self.args.device)
        weights     = torch.from_numpy(weights).unsqueeze(1).to(self.args.device)
        ################
        return states, actions, rewards, next_states, terminals, idx, weights
    
    def update_priorities(self, idx, priorities):
        ################
        for i, priority in zip(idx, priorities):
            self.sum_tree[i] = priority ** self.alpha + 1e-8
            self.min_tree[i] = priority ** self.alpha + 1e-8
            self.priority_cap = max(self.priority_cap, priority)
        ################

# ^^ [markdown]
# ### Task 6.2 Implement RainbowQNetwork class 

# ^^
class RainbowQNetwork(nn.Module):
    def __init__(self, args, std):
        super(RainbowQNetwork, self).__init__()
        ################
        self.layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
            NoisyLinear(args.hidden_dim, args.hidden_dim, std), nn.ReLU(),)
        self.advantage_head = NoisyLinear(args.hidden_dim, args.action_dim, std)
        self.value_head = NoisyLinear(args.hidden_dim, 1, std)
        ################

    def forward(self, x):
        ################
        interim_result = self.layers(x)
        advantage_result = self.advantage_head(interim_result)
        return self.value_head(interim_result) + advantage_result - torch.mean(advantage_result, dim=-1).unsqueeze(-1)
        ################

# ^^ [markdown]
# ### Task 6.2 Implement and train RainbowDQN agent 

# ^^
class RainbowDQN(DQN):
    def __init__(self, args, nstep=3, std=0.2, alpha=0.2, beta=0.2):
        super(RainbowDQN, self).__init__(args)
        self.buffer = RainbowBuffer(args, nstep, alpha, beta)
        self.alpha = alpha
        self.beta = beta
        self.nstep = nstep 
        self.q_net = RainbowQNetwork(args, std).to(args.device)
        self.q_target = RainbowQNetwork(args, std).to(args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate, eps=1e-5)
        self.std = std
        
    def update(self):
        ################
        states, actions, rewards, next_states, terminals, idx, weights = self.buffer.sample()
        with torch.no_grad():
            action_idxs = torch.argmax(self.q_net(next_states), dim=1)            # dim=(batch_size)
            q_ns = self.q_target(next_states).gather(1, action_idxs.unsqueeze(1)) # dim=(batch_size, 1)
        q_targets = rewards + (1-terminals) * (self.args.discount ** self.nstep) * q_ns
        self.optimizer.zero_grad()
        q_values = self.q_net(states).gather(1, actions)
        td_errors = nn.functional.smooth_l1_loss(q_values, q_targets, reduction='none')
        loss = torch.mean(td_errors * weights)
        ###loss = nn.functional.smooth_l1_loss(q_values, q_targets)
        loss.backward()
        self.optimizer.step()
        priorities = td_errors.detach().squeeze().cpu().tolist()
        self.buffer.update_priorities(idx, priorities)
        ################
        
    def anneal(self, step):
        if step < self.args.anneal_steps and step > self.args.init_steps:
            self.buffer.alpha = ((1 - self.alpha)/self.args.anneal_steps)*step + self.alpha
            self.buffer.beta = ((1 - self.beta)/self.args.anneal_steps)*step + self.beta
        else:
            pass

    def get_action(self, state, exploration=True):
        return torch.argmax(self.q_net(state)).item()
    
    def reset(self):
        self.buffer = RainbowBuffer(self.args, self.nstep, self.alpha, self.beta) 
        self.q_net = RainbowQNetwork(self.args, self.std).to(self.args.device)
        self.q_target = RainbowQNetwork(self.args, self.std).to(self.args.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.learning_rate, eps=1e-5)

# ^^
#agent = RainbowDQN(args)
#results_dqn6 = train_agent(args, agent)
#download_numpy("results_dqn6.npy", results_dqn6)
#results_dqn6.mean(1)[-10:].mean()

# ^^ [markdown]
# # Task 7 Plot collected results
# Plot evaluation performance with respect to number of frames for all versions of DQN (including raw one). Draw 90% confidence intervals for each line (see seaborn.lineplot).

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(filenames):
    _, main_ax = plt.subplots(figsize=(10,10))
    main_ax.set_title("Results of different DQN implementations")
    main_ax.set_xlabel("Thousands of frames")
    main_ax.set_ylabel("Loss")

    _, axes = plt.subplots(4, 2, figsize=(10,15))

    steps = np.arange(args.init_steps, args.total_timesteps, args.evaluate_freq) / 1000
    for i, (label, file) in enumerate(filenames):
        #mean_results = np.mean(np.load(file).T[:, args.init_steps//args.evaluate_freq:], axis=0)
        #plt.plot(steps, mean_results), label=label) 
        
        sample_steps = [np.stack((steps, s)) for s in np.load(file).T[:, args.init_steps//args.evaluate_freq:]]
        result = pd.DataFrame(np.concatenate(sample_steps, axis=1).T, columns=["step", "performance"])
        sns.lineplot(result, x="step", y="performance", errorbar=("ci", 90), label=label, ax=main_ax)
        sns.lineplot(result, x="step", y="performance", errorbar=("ci", 90), ax=axes[i//2,i%2])
        sns.regplot(result, x="step", y="performance", scatter=False, ax=axes[i//2,i%2])
        axes[i//2,i%2].set_title(label)
    
    for i in range(8):
        axes[i//2,i%2].set_xlabel("")
        axes[i//2,i%2].set_ylabel("")

results = [
    ("DQN", "results_dqn.npy"),
    ("DDQN", "results_dqn1.npy"),
    ("N-step DQN", "results_dqn2.npy"),
    ("Noisy DQN", "results_dqn3.npy"),
    ("Dueling DQN", "results_dqn4.npy"),
    ("Prioritized DQN", "results_dqn5.npy"),
    ("Rainbow DQN", "results_dqn6.npy"),
]

plot_results(results)
