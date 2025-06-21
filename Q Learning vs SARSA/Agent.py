import random

class Agent:
    def __init__(self, env, start, end, land_tag, track_range, use_sarsa=False, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.env = env
        self.start = start
        self.end = end
        self.land_tag = land_tag
        self.track_range = track_range  # valid rows
        self.use_sarsa = use_sarsa
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.prev_state = None
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-2, 0), (2, 0), (0, -2), (0, 2)]

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qs = [(self.get_q(state, a), a) for a in self.actions]
        return max(qs, key=lambda x: x[0])[1]

    def step(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])

        # Stay in track
        if not self.env.is_valid(next_state) or not (self.track_range[0] <= next_state[0] <= self.track_range[1]):
            return state, -10

        cell = self.env.get_state_type(next_state)
        reward = -1  # step

        if abs(action[0]) == 2 or abs(action[1]) == 2:
            reward -= 2  # jump cost

        if next_state == self.prev_state:
            reward -= 1
        elif cell.startswith(self.land_tag):
            if self.prev_state and self.env.get_state_type(self.prev_state).startswith(self.land_tag):
                reward += 10
            reward += 1
        elif cell == 'W':
            reward -= 10
        elif next_state == self.end:
            reward += 100

        self.prev_state = next_state
        return next_state, reward

    def learn(self, state, action, reward, next_state, next_action=None):
        q = self.get_q(state, action)
        if self.use_sarsa and next_action:
            next_q = self.get_q(next_state, next_action)
        else:
            next_q = max([self.get_q(next_state, a) for a in self.actions])
        self.q_table[(state, action)] = q + self.lr * (reward + self.gamma * next_q - q)