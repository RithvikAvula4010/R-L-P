import numpy as np
import random
import pickle

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_qs(self, state):
        return self.q_table.setdefault(state, [0] * self.action_size)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.get_qs(state)))

    def learn(self, state, action, reward, next_state):
        current_q = self.get_qs(state)[action]
        max_next_q = max(self.get_qs(next_state))
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_next_q)
        self.q_table[state][action] = new_q

    def save(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
