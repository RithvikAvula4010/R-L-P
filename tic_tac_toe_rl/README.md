# Tic-Tac-Toe with Reinforcement Learning

This project implements a Tic-Tac-Toe game with a reinforcement learning agent that learns to play optimally through self-play using Q-learning.

## Features

- Complete Tic-Tac-Toe game implementation
- Q-learning based reinforcement learning agent
- Self-play training capability
- Play against trained AI
- Visual game display
- State symmetry handling for faster learning

## Project Structure

```
tic_tac_toe_rl/
│
├── main.py                # Main entry point
├── game/
│   ├── __init__.py
│   ├── board.py           # Tic-Tac-Toe board implementation
│   └── display.py         # Visual representation of the game
│
├── agent/
│   ├── __init__.py
│   ├── q_learning.py      # Q-learning implementation
│   └── agent.py           # RL agent implementation
│
└── utils/
    ├── __init__.py
    └── state_utils.py     # Helper functions for state representation
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The program provides three options:
1. Train new agents - Trains two agents through self-play
2. Play against trained agent - Play against the trained AI
3. Exit - Quit the program

## How it Works

The reinforcement learning agent uses Q-learning to learn optimal strategies for playing Tic-Tac-Toe. The agent learns through self-play, where it plays against itself and updates its Q-values based on the outcomes of the games.

Key features of the implementation:
- Q-learning with epsilon-greedy exploration
- State symmetry handling for faster learning
- Reward system: +1 for win, -1 for loss, 0.5 for draw
- Visual display of the game state
- Save/load functionality for trained models

## Training

During training, the agent plays multiple episodes against itself, gradually improving its strategy. The training progress can be monitored through the visual display of games.

## Playing Against the AI

After training, you can play against the trained agent. The AI plays as 'O' and you play as 'X'. The game provides a visual display of the board and prompts for your moves. 