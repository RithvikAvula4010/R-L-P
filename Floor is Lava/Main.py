from Environment import FloorIsLavaEnv
from Agent import QLearningAgent
import tkinter as tk

CELL_SIZE = 60
EPISODES = 5000

env = FloorIsLavaEnv()
agent = QLearningAgent(state_size=(8, 8), action_size=8)

episode_stats = []  # To store all results

# ---------- Training ----------
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        steps += 1

    # Check if agent reached the destination
    reached_goal = state == (env.size - 1, env.size - 1)
    episode_stats.append((episode + 1, total_reward, steps, reached_goal))

    # Print key stats every 500 episodes
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1:4d} | Reward: {total_reward:5.1f} | Steps: {steps:3d} | {'Success' if reached_goal else 'Failed'}")

# ---------- Final Summary ----------
print("\n=== FINAL EPISODE STATS SUMMARY ===")
successes = [e for e in episode_stats if e[3]]
failures = [e for e in episode_stats if not e[3]]

print(f"Total Episodes: {EPISODES}")
print(f"Successes: {len(successes)} | Failures: {len(failures)}")
if successes:
    avg_reward_success = sum(e[1] for e in successes) / len(successes)
    print(f"Avg Reward (Successes): {avg_reward_success:.2f}")
if failures:
    avg_reward_fail = sum(e[1] for e in failures) / len(failures)
    print(f"Avg Reward (Failures): {avg_reward_fail:.2f}")

# ---------- GUI ----------
class LavaGameGUI:
    def __init__(self, master, env, agent):
        self.master = master
        self.env = env
        self.agent = agent
        self.canvas = tk.Canvas(master, width=env.size * CELL_SIZE, height=env.size * CELL_SIZE)
        self.canvas.pack()
        self.draw_grid()
        self.run_game()

    def draw_grid(self):
        self.canvas.delete("all")
        grid, agent_pos, visited = self.env.get_grid()

        for i in range(self.env.size):
            for j in range(self.env.size):
                x1, y1 = j * CELL_SIZE, i * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                tile = grid[i][j]
                color = {
                    'L': 'red',
                    'G': 'green',
                    'S': 'blue',
                    'D': 'gold'
                }.get(tile, 'gray')

                if (i, j) in visited and tile == 'G':
                    color = '#228B22'  # darker green for visited land

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')

        # Draw agent
        x, y = agent_pos
        x1, y1 = y * CELL_SIZE + 10, x * CELL_SIZE + 10
        x2, y2 = x1 + CELL_SIZE - 20, y1 + CELL_SIZE - 20
        self.canvas.create_oval(x1, y1, x2, y2, fill='white')

    def run_game(self):
        state = self.env.reset()
        self.update_gui()
        self.master.after(500, self.step, state)

    def step(self, state):
        action = self.agent.choose_action(state)
        next_state, reward, done = self.env.step(action)
        self.update_gui()
        if not done:
            self.master.after(500, self.step, next_state)

    def update_gui(self):
        self.draw_grid()
        self.master.update_idletasks()
        self.master.update()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Floor is Lava RL Agent")
    gui = LavaGameGUI(root, env, agent)
    root.mainloop()