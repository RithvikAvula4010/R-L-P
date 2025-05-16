import numpy as np

class Display:
    @staticmethod
    def print_board(board: np.ndarray) -> None:
        """Print the current state of the board."""
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        
        print("\n")
        for i in range(3):
            print("-------------")
            for j in range(3):
                print(f"| {symbols[board[i, j]]} ", end="")
            print("|")
        print("-------------")
        print("\n")

    @staticmethod
    def print_game_status(board: np.ndarray, game_over: bool, winner: int) -> None:
        """Print the current game status."""
        Display.print_board(board)
        
        if game_over:
            if winner == 0:
                print("Game Over! It's a draw!")
            else:
                winner_symbol = 'X' if winner == 1 else 'O'
                print(f"Game Over! {winner_symbol} wins!")
        else:
            current_player = 'X' if np.sum(board == 0) % 2 == 1 else 'O'
            print(f"Current player: {current_player}") 