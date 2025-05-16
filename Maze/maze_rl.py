import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import random
from Maze import maze, width, height

class MazeEnv:
    def __init__(self):
        self.maze = maze.copy()
        self.start = (0, 1)
        self.end = (height - 1, width - 2)
        self.current_pos = self.start
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.steps = 0
        self.max_steps = 100  # Reduced max steps for 8x8 maze
        
    def reset(self):
        self.current_pos = self.start
        self.steps = 0
        return self.current_pos
    
    def step(self, action):
        self.steps += 1
        next_pos = (self.current_pos[0] + self.actions[action][0],
                   self.current_pos[1] + self.actions[action][1])
        
        # Check if move is valid
        if (0 <= next_pos[0] < height and 0 <= next_pos[1] < width and 
            self.maze[next_pos[0]][next_pos[1]] == 0):
            self.current_pos = next_pos
            reward = 10 if next_pos == self.end else -0.1  # Increased reward for reaching goal
            done = next_pos == self.end or self.steps >= self.max_steps
        else:
            reward = -1  # Penalty for hitting wall
            done = self.steps >= self.max_steps
            
        return self.current_pos, reward, done
    
    def valid_actions(self, state):
        valid = []
        for i, (dx, dy) in enumerate(self.actions):
            nx, ny = state[0] + dx, state[1] + dy
            if 0 <= nx < height and 0 <= ny < width and self.maze[nx][ny] == 0:
                valid.append(i)
        return valid
    
    def render(self):
        plt.clf()
        plt.imshow(self.maze, cmap='binary')
        plt.plot(self.current_pos[1], self.current_pos[0], 'ro', markersize=10)
        plt.axis('off')
        plt.pause(0.01)

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.2, discount_factor=0.95, epsilon=1.0):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_size = action_size
        self.dead_ends = set()
        
    def get_state_key(self, state):
        return state
    
    def get_action(self, state, valid_actions):
        state_key = self.get_state_key(state)
        # Remove actions that lead to known dead ends
        filtered_actions = [a for a in valid_actions if (self.next_state(state, a) not in self.dead_ends)]
        if not filtered_actions:
            filtered_actions = valid_actions  # If all lead to dead ends, allow all
        if random.random() < self.epsilon:
            return random.choice(filtered_actions)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        # Pick best among filtered actions
        q_values = self.q_table[state_key]
        best_action = max(filtered_actions, key=lambda a: q_values[a])
        return best_action
    
    def next_state(self, state, action):
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return (state[0] + actions[action][0], state[1] + actions[action][1])
    
    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value
    
    def remember_dead_end(self, state, env):
        # Dead end: only one valid action (backwards)
        valid = env.valid_actions(state)
        if len(valid) <= 1 and state != env.end:
            self.dead_ends.add(state)

def train_agent(max_episodes=2000):
    env = MazeEnv()
    agent = QLearningAgent(state_size=(height, width), action_size=4)
    best_rewards = []
    best_episodes = []
    episode_paths = []
    success_count = 0
    required_successes = 3  # Number of successful episodes before stopping
    
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        path = [state]
        reached_goal = False
        
        while not done:
            valid_actions = env.valid_actions(state)
            action = agent.get_action(state, valid_actions)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            path.append(state)
            total_reward += reward
            
            # Check if we reached the goal
            if state == env.end:
                reached_goal = True
                success_count += 1
                print(f"\nSuccess! Episode {episode} reached the goal. Total successes: {success_count}")
            
            # Remember dead ends
            agent.remember_dead_end(state, env)
        
        # Store best episodes
        if len(best_rewards) < 3 or total_reward > min(best_rewards):
            if len(best_rewards) >= 3:
                idx = best_rewards.index(min(best_rewards))
                best_rewards.pop(idx)
                best_episodes.pop(idx)
                episode_paths.pop(idx)
            best_rewards.append(total_reward)
            best_episodes.append(episode)
            episode_paths.append(path)
        
        # Print progress every 50 episodes (more frequent for smaller maze)
        if episode % 50 == 0:
            print(f"Episode: {episode}, Successes: {success_count}, Epsilon: {agent.epsilon:.3f}")
        
        # Stop if we have enough successful episodes
        if success_count >= required_successes:
            print(f"\nTraining complete! Found {success_count} successful paths.")
            break
            
        agent.epsilon = max(0.01, agent.epsilon * 0.998)  # Slower epsilon decay for smaller maze
    
    return agent, best_episodes, episode_paths

def visualize_best_episodes(agent, best_episodes, episode_paths):
    env = MazeEnv()
    for i, episode in enumerate(best_episodes):
        path = episode_paths[i]
        plt.figure(figsize=(6, 6))  # Smaller figure size for 8x8 maze
        plt.imshow(env.maze, cmap='binary')
        y, x = zip(*path)
        plt.plot([p[1] for p in path], [p[0] for p in path], 'r-', linewidth=2, label='Path')
        plt.plot(x[0], y[0], 'go', markersize=8, label='Start')
        plt.plot(x[-1], y[-1], 'bo', markersize=8, label='End')
        plt.axis('off')
        plt.title(f'Top Episode {i+1} (Episode {episode})')
        plt.legend()
        plt.show()
        time.sleep(1)

if __name__ == "__main__":
    agent, best_episodes, episode_paths = train_agent()
    print("\nTop 3 episodes:", best_episodes)
    print("\nVisualizing best episodes...")
    visualize_best_episodes(agent, best_episodes, episode_paths) 