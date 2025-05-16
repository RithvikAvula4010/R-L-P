import numpy as np
from game.board import Board
from game.display import Display
from agent.agent import Agent
import time

def train_agents(episodes: int = 10000) -> None:
    """Train two agents through self-play."""
    board = Board()
    agent_x = Agent(player=1)
    agent_o = Agent(player=-1)
    display = Display()
    
    print("Training agents through self-play...")
    for episode in range(episodes):
        board.reset()
        while not board.is_game_over():
            current_agent = agent_x if board.current_player == 1 else agent_o
            
            # Get action from current agent
            action = current_agent.get_action(board, training=True)
            
            # Make move
            board.make_move(*action)
            
            # Update agent with reward
            reward = board.get_reward()
            current_agent.update(board, action, reward)
            
            if episode % 1000 == 0:
                display.print_board(board.get_board())
                time.sleep(0.1)
    
    # Save trained models
    agent_x.save_model("agent_x.pkl")
    agent_o.save_model("agent_o.pkl")
    print("Training completed!")

def play_against_agent() -> None:
    """Play against a trained agent."""
    board = Board()
    display = Display()
    
    # Load trained agent
    agent = Agent(player=-1)  # Agent plays as O
    agent.load_model("agent_o.pkl")
    
    print("Playing against AI (O). You are X.")
    while not board.is_game_over():
        display.print_game_status(board.get_board(), board.is_game_over(), board.winner)
        
        if board.current_player == 1:  # Human's turn
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                    if board.make_move(row, col):
                        break
                    print("Invalid move! Try again.")
                except ValueError:
                    print("Please enter numbers between 0 and 2!")
        else:  # Agent's turn
            print("AI is thinking...")
            time.sleep(0.5)
            action = agent.get_action(board, training=False)
            board.make_move(*action)
    
    display.print_game_status(board.get_board(), board.is_game_over(), board.winner)

def main():
    while True:
        print("\nTic-Tac-Toe with RL")
        print("1. Train new agents")
        print("2. Play against trained agent")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            episodes = int(input("Enter number of training episodes (default: 10000): ") or "10000")
            train_agents(episodes)
        elif choice == "2":
            play_against_agent()
        elif choice == "3":
            break
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main() 