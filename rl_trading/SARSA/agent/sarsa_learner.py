import numpy as np
import random

class Sarsa_Agent:
    def __init__(self, env, num_observations, num_actions, epsilon=None, alpha=None, gamma=None):
        '''
            Initialize the parameters of Q table
        '''
        self.env = env
        self.num_observations = num_observations
        self.num_actions = num_actions
        
        self.q_table = self.form_table()
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def form_table(self):
        '''
            Initialize Q table values to zero
        '''
        q_table = np.zeros([self.num_observations, self.num_actions])
        return q_table

    def take_action(self, state):
        '''
            Take action based on epsilon greedy method for training SARSA
        '''
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def take_action_(self, state):
        '''
            Take action in test environment
        '''
        action = np.argmax(self.q_table[state])
        return action
    
    def update_q_table(self, state, action, reward, next_state):
        '''
            Update Q table using Bellman equation
        '''
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + (self.gamma * next_max) - old_value)
        self.q_table[state, action] = new_value
    
    def save_q_table(self):
        '''
            Update Q table using Bellman equation
        '''
        with open("trained_model/q_table.npy", "wb") as fp:
            np.save(fp, self.q_table)
        np.savetxt("trained_model/q_table.txt", self.q_table, delimiter=",")
    
    def load_q_table(self):
        '''
            Update Q table using Bellman equation
        '''
        with open("trained_model/q_table.npy", "rb") as fp:
            self.q_table = np.load(fp)
