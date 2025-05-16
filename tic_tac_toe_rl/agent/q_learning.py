import numpy as np
from typing import Dict, Tuple, List
import random

class QLearning:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table: Dict[str, Dict[Tuple[int, int], float]] = {}

    def get_state_key(self, board: np.ndarray) -> str:
        """Convert board state to string key."""
        return str(board.tolist())

    def get_action(self, state: str, valid_moves: List[Tuple[int, int]], training: bool = True) -> Tuple[int, int]:
        """Choose an action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = {move: 0.0 for move in valid_moves}

        if training and random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        
        state_actions = self.q_table[state]
        return max(state_actions.items(), key=lambda x: x[1])[0]

    def update(self, state: str, action: Tuple[int, int], reward: float, next_state: str, next_valid_moves: List[Tuple[int, int]]) -> None:
        """Update Q-values using Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {move: 0.0 for move in next_valid_moves}

        # Q-learning update
        current_q = self.q_table[state].get(action, 0.0)
        next_max_q = max(self.q_table[next_state].values()) if next_valid_moves else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q

    def save_q_table(self, filename: str) -> None:
        """Save Q-table to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename: str) -> None:
        """Load Q-table from file."""
        import pickle
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f) 