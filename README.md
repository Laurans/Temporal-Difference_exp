# Temporal-Difference experimentations 

mini project to use OpenAI Gym's `Taxi-v2` environment to design an algorithm to teach a taxi agent to navigate a small gridworld.
The goal is to experiment on temporal-difference methods such as Sarsa, Q-Learning and expected Sarsa. 

## Installation

### Requirements
* OpenAI Gym
* Python 3.5 and up

## Usage

`python main.py`

## Methods

### Q-Learning / Sarsamax
```
self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
```

### Expected Sarsa
```
policy = np.ones(self.nA)*self.eps/self.nA
policy[np.argmax(self.Q[next_state])] = 1-self.eps+self.eps/self.nA

self.Q[state][action] += self.alpha * (reward + self.gamma * np.dot(policy, self.Q[next_state]) - self.Q[state][action])
```
