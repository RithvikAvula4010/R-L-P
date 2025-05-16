import numpy as np
from typing import Tuple, Optional, List

class Board:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.game_over = False
        self.winner = None

    def reset(self) -> None:
        """Reset the board to initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move at the specified position.
        Returns True if move was valid and successful, False otherwise.
        """
        if self.game_over or not self.is_valid_move(row, col):
            return False

        self.board[row, col] = self.current_player
        self.check_game_state()
        self.current_player *= -1
        return True

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if the move is valid."""
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of all valid moves."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def check_game_state(self) -> None:
        """Check if the game is over and determine the winner."""
        # Check rows and columns
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                self.game_over = True
                self.winner = self.current_player
                return

        # Check diagonals
        if abs(sum(np.diag(self.board))) == 3 or abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            self.game_over = True
            self.winner = self.current_player
            return

        # Check for draw
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0  # Draw

    def get_state(self) -> str:
        """Get string representation of the board state."""
        return str(self.board.tolist())

    def get_reward(self) -> float:
        """Get reward for the current state."""
        if not self.game_over:
            return 0
        if self.winner == 0:
            return 0.5  # Draw
        return 1.0 if self.winner == 1 else -1.0

    def get_board(self) -> np.ndarray:
        """Get the current board state."""
        return self.board.copy()

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_over 