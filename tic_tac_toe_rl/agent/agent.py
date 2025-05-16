from typing import Tuple, List
import numpy as np
from .q_learning import QLearning
from game.board import Board

class Agent:
    def __init__(self, player: int, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 0.1):
        self.player = player  # 1 for X, -1 for O
        self.q_learning = QLearning(learning_rate, discount_factor, exploration_rate)

    def get_action(self, board: Board, training: bool = True) -> Tuple[int, int]:
        """Get the next action for the current board state."""
        state = self.q_learning.get_state_key(board.get_board())
        valid_moves = board.get_valid_moves()
        return self.q_learning.get_action(state, valid_moves, training)

    def update(self, board: Board, action: Tuple[int, int], reward: float) -> None:
        """Update the agent's Q-values based on the game outcome."""
        state = self.q_learning.get_state_key(board.get_board())
        next_state = self.q_learning.get_state_key(board.get_board())
        next_valid_moves = board.get_valid_moves()
        self.q_learning.update(state, action, reward, next_state, next_valid_moves)

    def save_model(self, filename: str) -> None:
        """Save the agent's Q-table to a file."""
        self.q_learning.save_q_table(filename)

    def load_model(self, filename: str) -> None:
        """Load the agent's Q-table from a file."""
        self.q_learning.load_q_table(filename) 