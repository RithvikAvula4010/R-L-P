import numpy as np
from typing import Dict, Tuple, List, Deque
import random
from collections import deque
from utils.state_utils import get_symmetrical_states, get_symmetrical_action

class QLearning:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 0.3,
                 memory_size: int = 10000, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = exploration_rate
        self.exploration_rate = exploration_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.q_table: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.memory: Deque[Tuple[str, Tuple[int, int], float, str, List[Tuple[int, int]]]] = deque(maxlen=memory_size)
        self.training_steps = 0
        self.visited_states = set()

    def get_state_key(self, board: np.ndarray) -> str:
        """Convert board state to string key, using the canonical form."""
        # Get all symmetrical states
        symmetrical_states = get_symmetrical_states(board)
        # Convert all states to strings
        state_strings = [str(state.tolist()) for state in symmetrical_states]
        # Return the lexicographically smallest state (canonical form)
        return min(state_strings)

    def get_action(self, state: str, valid_moves: List[Tuple[int, int]], training: bool = True) -> Tuple[int, int]:
        """Choose an action using epsilon-greedy policy with decay and forced exploration."""
        if state not in self.q_table:
            self.q_table[state] = {move: 0.0 for move in valid_moves}

        # Decay exploration rate over time, but keep a minimum value
        if training:
            min_exploration_rate = 0.1
            decay_factor = 0.995
            self.exploration_rate = max(
                min_exploration_rate,
                self.initial_exploration_rate * (decay_factor ** (self.training_steps / 1000))
            )

            # Force exploration of unvisited states
            unvisited_moves = [move for move in valid_moves if (state, move) not in self.visited_states]
            if unvisited_moves and random.random() < 0.3:  # 30% chance to force exploration
                move = random.choice(unvisited_moves)
                self.visited_states.add((state, move))
                return move

            # Regular epsilon-greedy exploration
            if random.random() < self.exploration_rate:
                move = random.choice(valid_moves)
                self.visited_states.add((state, move))
                return move

        # Exploitation: choose the best known action
        state_actions = self.q_table[state]
        best_moves = [move for move, value in state_actions.items() 
                     if value == max(state_actions.values())]
        move = random.choice(best_moves)  # Randomly choose among best moves
        self.visited_states.add((state, move))
        return move

    def update(self, state: str, action: Tuple[int, int], reward: float, next_state: str, next_valid_moves: List[Tuple[int, int]]) -> None:
        """Update Q-values using Q-learning update rule and store experience in memory."""
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, next_valid_moves))
        
        # Initialize Q-values if not present
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {move: 0.0 for move in next_valid_moves}

        # Update Q-value for the current state-action pair
        current_q = self.q_table[state].get(action, 0.0)
        next_max_q = max(self.q_table[next_state].values()) if next_valid_moves else 0.0
        
        # Add a small bonus for exploring new states
        exploration_bonus = 0.1 if (state, action) not in self.visited_states else 0.0
        new_q = current_q + self.learning_rate * (reward + exploration_bonus + self.discount_factor * next_max_q - current_q)
        self.q_table[state][action] = new_q

        # Update symmetrical states
        self._update_symmetrical_states(state, action, new_q)

        # Experience replay
        if len(self.memory) >= self.batch_size:
            self._experience_replay()

        self.training_steps += 1

    def _update_symmetrical_states(self, state: str, action: Tuple[int, int], q_value: float) -> None:
        """Update Q-values for all symmetrical states and actions."""
        # Convert string state back to numpy array
        state_array = np.array(eval(state))
        
        # Get all symmetrical states and actions
        symmetrical_states = get_symmetrical_states(state_array)
        symmetrical_actions = get_symmetrical_action(action)
        
        # Update Q-values for all symmetrical states
        for sym_state, sym_action in zip(symmetrical_states, symmetrical_actions):
            sym_state_key = str(sym_state.tolist())
            if sym_state_key not in self.q_table:
                self.q_table[sym_state_key] = {}
            self.q_table[sym_state_key][sym_action] = q_value
            self.visited_states.add((sym_state_key, sym_action))

    def _experience_replay(self) -> None:
        """Learn from past experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, next_valid_moves in batch:
            if state not in self.q_table:
                self.q_table[state] = {action: 0.0}
            if next_state not in self.q_table:
                self.q_table[next_state] = {move: 0.0 for move in next_valid_moves}

            # Q-learning update
            current_q = self.q_table[state].get(action, 0.0)
            next_max_q = max(self.q_table[next_state].values()) if next_valid_moves else 0.0
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
            self.q_table[state][action] = new_q

            # Update symmetrical states
            self._update_symmetrical_states(state, action, new_q)

    def save_q_table(self, filename: str) -> None:
        """Save Q-table and training metadata to file."""
        import pickle
        save_data = {
            'q_table': self.q_table,
            'training_steps': self.training_steps,
            'visited_states': self.visited_states
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def load_q_table(self, filename: str) -> None:
        """Load Q-table and training metadata from file."""
        import pickle
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(save_data, dict):
                self.q_table = save_data.get('q_table', save_data)  # Try new format, fall back to old format
                self.training_steps = save_data.get('training_steps', 0)
                self.visited_states = save_data.get('visited_states', set())
            else:
                # Old format where the file directly contained the q_table
                self.q_table = save_data
                self.training_steps = 0
                self.visited_states = set() 