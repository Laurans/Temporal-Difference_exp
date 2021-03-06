import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1
        self.i_episode = 0
        self.alpha = 0.1
        self.gamma = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = np.ones(self.nA)*self.eps/self.nA
        policy[np.argmax(self.Q[state])] = 1-self.eps+self.eps/self.nA
        
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        # Expected sarsa
        policy = np.ones(self.nA)*self.eps/self.nA
        policy[np.argmax(self.Q[next_state])] = 1-self.eps+self.eps/self.nA
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.dot(policy, self.Q[next_state]) - self.Q[state][action])
                               
        if done:
            self.i_episode += 1
            self.eps = 0.6**self.i_episode
