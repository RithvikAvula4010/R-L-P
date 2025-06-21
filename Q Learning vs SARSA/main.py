import tkinter as tk
import time
from Environment import GridWorld
from Agent import Agent

def draw_agent(canvas, pos, color):
    block = 50
    x, y = pos
    canvas.create_oval(y * block + 10, x * block + 10, y * block + 40, x * block + 40, fill=color)

def run_agents(env, canvas, agent1, agent2, render=False):
    state1 = agent1.start
    state2 = agent2.start
    agent1.prev_state = None
    agent2.prev_state = None

    action1 = agent1.choose_action(state1)
    action2 = agent2.choose_action(state2)

    steps = 0
    while steps < 300:
        if render:
            canvas.delete("all")
            env.draw(canvas)
            draw_agent(canvas, state1, 'purple')   # Q-learning
            draw_agent(canvas, state2, 'orange')   # SARSA
            canvas.update()
            time.sleep(0.01)

        # Q-learning step
        next_state1, reward1 = agent1.step(state1, action1)
        next_action1 = agent1.choose_action(next_state1)
        agent1.learn(state1, action1, reward1, next_state1, next_action1 if agent1.use_sarsa else None)

        # SARSA step
        next_state2, reward2 = agent2.step(state2, action2)
        next_action2 = agent2.choose_action(next_state2)
        agent2.learn(state2, action2, reward2, next_state2, next_action2 if agent2.use_sarsa else None)

        state1, action1 = next_state1, next_action1
        state2, action2 = next_state2, next_action2

        if state1 == env.track1_end:
            return "Q-learning", steps
        if state2 == env.track2_end:
            return "SARSA", steps

        steps += 1

    return "None", steps

def main():
    root = tk.Tk()
    root.title("RL Racing: Q-Learning vs SARSA")
    canvas = tk.Canvas(root, width=500, height=500)
    canvas.pack()

    env = GridWorld()  # ✅ Only generate once and reuse

    agent1 = Agent(env, env.track1_start, env.track1_end, 'L1', track_range=(0, 4), use_sarsa=False)
    agent2 = Agent(env, env.track2_start, env.track2_end, 'L2', track_range=(5, 9), use_sarsa=True)

    q_wins = sarsa_wins = no_wins = 0

    for episode in range(1, 5001):
        winner, steps = run_agents(env, canvas, agent1, agent2, render=(episode % 500 == 0))  # show every 500th episode
        if winner == "Q-learning":
            q_wins += 1
        elif winner == "SARSA":
            sarsa_wins += 1
        else:
            no_wins += 1

        if episode % 500 == 0:
            print(f"Episode {episode}: Winner = {winner}, Steps = {steps}")

    print("\nTraining finished.")
    print(f"Q-learning Wins: {q_wins}")
    print(f"SARSA Wins: {sarsa_wins}")
    print(f"No Winner: {no_wins}")

    root.mainloop()

if __name__ == "__main__":
    main()
