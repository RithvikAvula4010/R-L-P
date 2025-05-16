import numpy as np
from typing import Tuple, List

def get_state_hash(board: np.ndarray) -> str:
    """Convert board state to a unique string hash."""
    return str(board.tolist())

def get_symmetrical_states(board: np.ndarray) -> List[np.ndarray]:
    """Get all symmetrical states of the current board."""
    states = [board]
    
    # Rotate 90 degrees
    states.append(np.rot90(board))
    # Rotate 180 degrees
    states.append(np.rot90(board, 2))
    # Rotate 270 degrees
    states.append(np.rot90(board, 3))
    
    # Flip horizontally
    states.append(np.fliplr(board))
    # Flip vertically
    states.append(np.flipud(board))
    
    # Flip and rotate combinations
    states.append(np.rot90(np.fliplr(board)))
    states.append(np.rot90(np.flipud(board)))
    
    return states

def get_symmetrical_action(action: Tuple[int, int], board_size: int = 3) -> List[Tuple[int, int]]:
    """Get all symmetrical actions for a given action."""
    row, col = action
    actions = [(row, col)]
    
    # Rotate 90 degrees
    actions.append((col, board_size - 1 - row))
    # Rotate 180 degrees
    actions.append((board_size - 1 - row, board_size - 1 - col))
    # Rotate 270 degrees
    actions.append((board_size - 1 - col, row))
    
    # Flip horizontally
    actions.append((row, board_size - 1 - col))
    # Flip vertically
    actions.append((board_size - 1 - row, col))
    
    # Flip and rotate combinations
    actions.append((col, row))
    actions.append((board_size - 1 - col, board_size - 1 - row))
    
    return actions 